"""Torch baselines for image-only and multimodal experiments."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = None

try:
    from torchvision import models
except ImportError:  # pragma: no cover - optional dependency
    models = None


if torch is not None:

    class TinyFrameEncoder(nn.Module):
        def __init__(self, out_dim: int = 128) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class ResNet18FrameEncoder(nn.Module):
        def __init__(self, out_dim: int = 128, pretrained: bool = False) -> None:
            super().__init__()
            if models is None:
                raise ImportError("torchvision is required to use the resnet18 backbone")

            backbone = self._build_backbone(pretrained=pretrained)
            original_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                1,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None,
            )
            with torch.no_grad():
                if pretrained:
                    backbone.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
                else:
                    nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
                if original_conv.bias is not None and backbone.conv1.bias is not None:
                    backbone.conv1.bias.copy_(original_conv.bias)
            in_features = backbone.fc.in_features
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            self.proj = nn.Linear(in_features, out_dim)

        @staticmethod
        def _build_backbone(pretrained: bool):
            try:
                if pretrained:
                    return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                return models.resnet18(weights=None)
            except AttributeError:  # pragma: no cover - older torchvision fallback
                return models.resnet18(pretrained=pretrained)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.features(x).flatten(1)
            return self.proj(features)


    def _build_frame_encoder(backbone: str, hidden_dim: int, pretrained: bool) -> nn.Module:
        if backbone == "tiny":
            return TinyFrameEncoder(out_dim=hidden_dim)
        if backbone == "resnet18":
            return ResNet18FrameEncoder(out_dim=hidden_dim, pretrained=pretrained)
        raise ValueError(f"Unsupported backbone: {backbone}")


    class TemporalConvAggregator(nn.Module):
        def __init__(self, hidden_dim: int = 128, kernel_size: int = 3, dropout: float = 0.1) -> None:
            super().__init__()
            padding = kernel_size // 2
            self.net = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
            )

        def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
            temporal_features = frame_features.transpose(1, 2)
            temporal_out = self.net(temporal_features).transpose(1, 2)
            temporal_out = temporal_out + frame_features
            return temporal_out[:, -1, :]


    def _build_temporal_aggregator(hidden_dim: int, temporal_backend: str) -> nn.Module | None:
        if temporal_backend == "gru":
            return nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        if temporal_backend == "tcn":
            return TemporalConvAggregator(hidden_dim=hidden_dim)
        if temporal_backend == "mean":
            return None
        raise ValueError(f"Unsupported temporal backend: {temporal_backend}")


    def _aggregate_temporal_features(
        frame_features: torch.Tensor,
        temporal_backend: str,
        temporal: nn.Module | None,
    ) -> torch.Tensor:
        if temporal_backend == "gru":
            if temporal is None:
                raise ValueError("GRU temporal backend requires a recurrent module")
            frame_features = frame_features.to(temporal.weight_ih_l0.dtype)
            _, hidden = temporal(frame_features)
            return hidden[-1]
        if temporal_backend == "tcn":
            if temporal is None:
                raise ValueError("TCN temporal backend requires a temporal aggregator")
            return temporal(frame_features)
        if temporal_backend == "mean":
            return frame_features.mean(dim=1)
        raise ValueError(f"Unsupported temporal backend: {temporal_backend}")


    class ImageOnlyGRURegressor(nn.Module):
        def __init__(
            self,
            hidden_dim: int = 128,
            probabilistic: str = "gaussian",
            backbone: str = "tiny",
            pretrained: bool = False,
            temporal_backend: str = "gru",
        ) -> None:
            super().__init__()
            self.encoder = _build_frame_encoder(backbone=backbone, hidden_dim=hidden_dim, pretrained=pretrained)
            self.temporal_backend = temporal_backend
            self.temporal = _build_temporal_aggregator(hidden_dim=hidden_dim, temporal_backend=temporal_backend)
            out_dim = 2 if probabilistic == "gaussian" else 3
            self.head = nn.Linear(hidden_dim, out_dim)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            batch, time_steps, _, _, _ = images.shape
            frame_features = self.encoder(images.view(batch * time_steps, 1, images.size(-2), images.size(-1)))
            frame_features = frame_features.view(batch, time_steps, -1)
            temporal_out = _aggregate_temporal_features(
                frame_features,
                temporal_backend=self.temporal_backend,
                temporal=self.temporal,
            )
            return self.head(temporal_out.to(self.head.weight.dtype))


    class MultimodalGRURegressor(nn.Module):
        def __init__(
            self,
            tabular_dim: int,
            hidden_dim: int = 128,
            probabilistic: str = "gaussian",
            backbone: str = "tiny",
            pretrained: bool = False,
            temporal_backend: str = "gru",
        ) -> None:
            super().__init__()
            self.encoder = _build_frame_encoder(backbone=backbone, hidden_dim=hidden_dim, pretrained=pretrained)
            self.temporal_backend = temporal_backend
            self.temporal = _build_temporal_aggregator(hidden_dim=hidden_dim, temporal_backend=temporal_backend)
            self.tabular = nn.Sequential(
                nn.Linear(tabular_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(hidden_dim, hidden_dim),
            )
            out_dim = 2 if probabilistic == "gaussian" else 3
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
            batch, time_steps, _, _, _ = images.shape
            frame_features = self.encoder(images.view(batch * time_steps, 1, images.size(-2), images.size(-1)))
            frame_features = frame_features.view(batch, time_steps, -1)
            temporal_out = _aggregate_temporal_features(
                frame_features,
                temporal_backend=self.temporal_backend,
                temporal=self.temporal,
            )
            tabular_embed = self.tabular(tabular).to(temporal_out.dtype)
            fused = torch.cat([temporal_out, tabular_embed], dim=-1).to(self.head[0].weight.dtype)
            return self.head(fused)


    def gaussian_nll(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean = output[:, 0]
        log_var = output[:, 1].clamp(min=-6.0, max=6.0)
        precision = torch.exp(-log_var)
        return torch.mean(0.5 * (log_var + precision * (target - mean) ** 2))


    def quantile_loss(
        output: torch.Tensor,
        target: torch.Tensor,
        lower_q: float = 0.1,
        upper_q: float = 0.9,
    ) -> torch.Tensor:
        center = output[:, 0]
        lower = output[:, 1]
        upper = output[:, 2]
        diff_center = target - center
        diff_lower = target - lower
        diff_upper = target - upper
        center_loss = torch.mean(diff_center ** 2)
        lower_loss = torch.mean(torch.maximum(lower_q * diff_lower, (lower_q - 1.0) * diff_lower))
        upper_loss = torch.mean(torch.maximum(upper_q * diff_upper, (upper_q - 1.0) * diff_upper))
        order_penalty = torch.mean(torch.relu(lower - upper))
        return center_loss + lower_loss + upper_loss + order_penalty


    def decode_output(output: torch.Tensor, mode: str = "gaussian") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode == "gaussian":
            mean = output[:, 0]
            std = torch.exp(0.5 * output[:, 1].clamp(min=-6.0, max=6.0))
            z = 1.2815515655446004
            return mean, mean - z * std, mean + z * std
        return output[:, 0], output[:, 1], output[:, 2]

else:

    class ImageOnlyGRURegressor:  # pragma: no cover - optional dependency
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("torch is required to use ImageOnlyGRURegressor")


    class MultimodalGRURegressor:  # pragma: no cover - optional dependency
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("torch is required to use MultimodalGRURegressor")
