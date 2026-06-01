"""TC-QFormer encoder, SIFB-lite, and multi-quantile head.

Reproduces Guo et al. 2026, "Interval-based Tropical Cyclone Intensity Forecasting
with Spatiotemporal Transformers" (Remote Sensing 18(7):1069), adapted to
TyphoonUQ-Bench (single-channel IR frames, 8 metadata features, single-horizon
prediction).

Key adaptations (input/output contract bridge):
  * Frame channels C=1 (Digital Typhoon infrared only) instead of IR+WV.
  * Six 224x224 history frames downsampled to 128x128 (paper uses four 128x128
    frames).
  * Single forecast horizon Tout=1 per task (analysis-0h / forecast-6h /
    forecast-12h); each task trains an independent model.
  * Scalar branch consumes the 8-feature TyphoonUQ-Bench metadata sequence
    (lat, lon, motion_u, motion_v, motion_speed, quality_flag, month_sin,
    month_cos) instead of Vmax history.
  * Quantile head outputs {0.1, 0.25, 0.5, 0.75, 0.9}; q=0.5 -> point
    prediction, (q=0.1, q=0.9) -> 80% interval. The 90% interval needed by
    TyphoonUQ-Bench range_coverage_at_90 is produced post-hoc through residual
    conformal calibration on the validation split (sample symmetry around the
    median, identical to the built-in baselines).

NPU notes:
  * Uses torch.nn.MultiheadAttention with batch_first=True and a fixed
    sinusoidal positional embedding; both kernels are supported by torch_npu.
  * No SDPA flash kernels -- the explicit attention path keeps Ascend op
    coverage robust.
  * Patch embedding via Conv2d(stride=p) avoids einops dependency.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


PATCH_SIZE = 8
EMBED_DIM = 256
NUM_HEADS = 8
DEPTH = 6
MLP_RATIO = 4.0
DROPOUT = 0.0
DEFAULT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _sinusoidal_pe(num_tokens: int, dim: int) -> torch.Tensor:
    position = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(num_tokens, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class _MLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class _AttentionBlock(nn.Module):
    """Pre-norm transformer block with batched MHA (Ascend-friendly)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLPBlock(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PredFormerEncoder(nn.Module):
    """Encoder-only PredFormer with Quadruplet_TSST interleaving.

    The Quadruplet block alternates four attention sublayers operating on the
    (T, S) and (S, T) factorisations of the patch grid. Each layer reshapes the
    (B, T, P, D) tensor so MultiheadAttention contracts the desired axis while
    leaving the other batched alongside B.
    """

    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = PATCH_SIZE,
        in_channels: int = 1,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        depth: int = DEPTH,
        mlp_ratio: float = MLP_RATIO,
        dropout: float = DROPOUT,
        num_frames: int = 6,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} not divisible by patch {patch_size}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.register_buffer(
            "spatial_pe",
            _sinusoidal_pe(self.num_patches, embed_dim).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "temporal_pe",
            _sinusoidal_pe(num_frames, embed_dim).unsqueeze(0).unsqueeze(2),
            persistent=False,
        )

        # Four sub-blocks per Quadruplet layer (T, S, S, T) following the paper.
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        _AttentionBlock(embed_dim, num_heads, mlp_ratio, dropout),  # temporal
                        _AttentionBlock(embed_dim, num_heads, mlp_ratio, dropout),  # spatial
                        _AttentionBlock(embed_dim, num_heads, mlp_ratio, dropout),  # spatial
                        _AttentionBlock(embed_dim, num_heads, mlp_ratio, dropout),  # temporal
                    ]
                )
            )
        self.norm = nn.LayerNorm(embed_dim)

    def _patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(b, t, self.num_patches, self.embed_dim)
        return x

    def _apply_temporal(self, block: _AttentionBlock, x: torch.Tensor) -> torch.Tensor:
        b, t, p, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * p, t, d)
        x = block(x)
        x = x.reshape(b, p, t, d).permute(0, 2, 1, 3).contiguous()
        return x

    def _apply_spatial(self, block: _AttentionBlock, x: torch.Tensor) -> torch.Tensor:
        b, t, p, d = x.shape
        x = x.reshape(b * t, p, d)
        x = block(x)
        x = x.reshape(b, t, p, d).contiguous()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._patch_embed(x)
        z = z + self.spatial_pe + self.temporal_pe
        for blocks in self.layers:
            t_block, s_block_a, s_block_b, t_block_b = blocks
            z = self._apply_temporal(t_block, z)
            z = self._apply_spatial(s_block_a, z)
            z = self._apply_spatial(s_block_b, z)
            z = self._apply_temporal(t_block_b, z)
        z = self.norm(z)
        pooled = z.mean(dim=(1, 2))
        return pooled


class ScalarImageFusionBlock(nn.Module):
    """SIFB-lite: gated residual fusion of pooled latent and scalar features."""

    def __init__(self, embed_dim: int, scalar_dim: int) -> None:
        super().__init__()
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gate = nn.Linear(embed_dim, embed_dim)

    def forward(self, z: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        if scalars is None or scalars.numel() == 0:
            return z
        s_feat = self.scalar_mlp(scalars)
        gate = torch.sigmoid(self.gate(z))
        return z + gate * s_feat


class TCQFormer(nn.Module):
    """End-to-end TC-QFormer adapted for TyphoonUQ-Bench single-horizon UQ."""

    def __init__(
        self,
        scalar_dim: int,
        num_frames: int = 6,
        image_size: int = 128,
        patch_size: int = PATCH_SIZE,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        depth: int = DEPTH,
        mlp_ratio: float = MLP_RATIO,
        dropout: float = DROPOUT,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        horizon: int = 1,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.num_frames = num_frames
        self.quantiles = tuple(quantiles)
        self.horizon = horizon

        self.encoder = PredFormerEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            num_frames=num_frames,
        )
        self.sifb = ScalarImageFusionBlock(embed_dim=embed_dim, scalar_dim=scalar_dim)

        # Stage-1 head (median only) and stage-2 head (multi-quantile) live in
        # parallel so the stage transition just swaps `active_head`.
        self.median_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, horizon),
        )
        self.quantile_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, horizon * len(self.quantiles)),
        )
        self.active_head = "median"

    def set_stage(self, stage: int) -> None:
        if stage == 1:
            self.active_head = "median"
        elif stage == 2:
            self.active_head = "quantile"
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def _resize_frames(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, T, C, H, W).
        if images.size(-1) == self.image_size and images.size(-2) == self.image_size:
            return images
        b, t, c, h, w = images.shape
        flat = images.reshape(b * t, c, h, w)
        resized = F.interpolate(flat, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return resized.reshape(b, t, c, self.image_size, self.image_size)

    def forward(self, images: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        images = self._resize_frames(images)
        if images.dim() == 5 and images.size(2) != 1:
            images = images[:, :, :1]
        z = self.encoder(images)
        z = self.sifb(z, scalars)
        if self.active_head == "median":
            return self.median_head(z)
        out = self.quantile_head(z)
        return out.reshape(out.size(0), self.horizon, len(self.quantiles))


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    """Average pinball loss over horizons and quantiles.

    prediction: (B, H, |Q|), target: (B, H)
    """
    if prediction.dim() == 2:
        prediction = prediction.unsqueeze(-1)
    if target.dim() == 1:
        target = target.unsqueeze(-1)
    q = torch.tensor(quantiles, device=prediction.device, dtype=prediction.dtype).view(1, 1, -1)
    diff = target.unsqueeze(-1) - prediction
    loss = torch.maximum(q * diff, (q - 1.0) * diff)
    return loss.mean()


def median_l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.dim() == 2 and target.dim() == 1:
        prediction = prediction.squeeze(-1)
    return F.l1_loss(prediction, target)


def decode_quantile_outputs(
    prediction: torch.Tensor,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    target_coverage: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (point, lower, upper) at the requested nominal coverage.

    Quantile outputs are enforced monotonically non-decreasing per sample to
    avoid pathological crossings that would break in-range metrics.
    """
    sorted_pred, _ = torch.sort(prediction, dim=-1)
    qs = torch.tensor(quantiles, device=prediction.device, dtype=prediction.dtype)
    alpha = (1.0 - target_coverage) / 2.0
    lower_q = alpha
    upper_q = 1.0 - alpha
    lower_idx = int(torch.argmin(torch.abs(qs - lower_q)).item())
    upper_idx = int(torch.argmin(torch.abs(qs - upper_q)).item())
    median_idx = int(torch.argmin(torch.abs(qs - 0.5)).item())
    point = sorted_pred[..., median_idx]
    lower = sorted_pred[..., lower_idx]
    upper = sorted_pred[..., upper_idx]
    return point, lower, upper
