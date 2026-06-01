from __future__ import annotations

from pathlib import Path
import math

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_PATH = Path(__file__).resolve()
DATA_ROOT = SCRIPT_PATH.parent.parent.parent / "full"
FORMAL_RUN = DATA_ROOT / "outputs" / "formal_2026-04-24_202126"
MODEL_KEY = "torch_multimodal_resnet18_ascend"
MODEL_LABEL = "Multimodal (ResNet18)"
MODEL_DIR = FORMAL_RUN / MODEL_KEY
OUTPUT_DIR = SCRIPT_PATH.parent / "figures_png"

TASK_ORDER = ["analysis-0h", "forecast-6h", "forecast-12h"]
TASK_LABELS = {
    "analysis-0h": "Analysis (0 h)",
    "forecast-6h": "Forecast (+6 h)",
    "forecast-12h": "Forecast (+12 h)",
}
TASK_TO_HORIZON_HOURS = {
    "analysis-0h": 0,
    "forecast-6h": 6,
    "forecast-12h": 12,
}
OUTPUT_STEMS = {
    "analysis-0h": "gaussian_analysis_0h",
    "forecast-6h": "gaussian_forecast_6h",
    "forecast-12h": "gaussian_forecast_12h",
}
AGENCY_COLUMNS = ["pressure_jma", "pressure_jtwc", "pressure_cma", "pressure_hko"]

MM_TO_IN = 1 / 25.4
Z_80 = 1.2815515655446004

TARGET_STORM_ID = "201211"
INITIALIZATION_TIMESTAMP = pd.Timestamp("2012-08-06 00:00:00")

TARGET_COLOR = "#1F2F3B"
TARGET_FILL = "#E6ECEF"
MODEL_COLOR = "#4C8C68"
MODEL_FILL = "#B5D8C0"
TEXT_COLOR = "#43515A"
GRID_COLOR = "#E5EBEF"
SPINE_COLOR = "#B8C4CC"

FIGURE_WIDTH_MM = 88
FIGURE_HEIGHT_MM = 76


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )
    sns.set_style("whitegrid", {"grid.color": GRID_COLOR, "grid.linewidth": 0.6})


def mm_size(width_mm: float, height_mm: float) -> tuple[float, float]:
    return width_mm * MM_TO_IN, height_mm * MM_TO_IN


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError(f"Gaussian sigma must be positive, got {sigma}.")
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def load_prediction_rows() -> dict[str, pd.Series]:
    rows: dict[str, pd.Series] = {}
    for task in TASK_ORDER:
        predictions_path = MODEL_DIR / task / "predictions.parquet"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Required prediction parquet file not found: {predictions_path}")

        predictions = pd.read_parquet(predictions_path)
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
        valid_timestamp = INITIALIZATION_TIMESTAMP + pd.Timedelta(hours=TASK_TO_HORIZON_HOURS[task])
        target_row = predictions.loc[
            (predictions["storm_id"].astype(str) == TARGET_STORM_ID)
            & (predictions["timestamp"] == valid_timestamp)
        ]
        if target_row.empty:
            raise ValueError(
                f"Could not find storm {TARGET_STORM_ID} at {valid_timestamp} in {predictions_path}."
            )
        rows[task] = target_row.iloc[0]
    return rows


def target_distribution(row: pd.Series) -> tuple[float, float, np.ndarray]:
    target = float(row["target"])
    agencies = row[AGENCY_COLUMNS].astype(float).dropna().to_numpy()

    target_lower = row.get("target_lower")
    target_upper = row.get("target_upper")
    if pd.notna(target_lower) and pd.notna(target_upper) and float(target_upper) > float(target_lower):
        sigma = (float(target_upper) - float(target_lower)) / (2.0 * Z_80)
        return target, sigma, agencies

    if len(agencies) >= 2 and np.std(agencies) > 0:
        return float(np.mean(agencies)), float(np.std(agencies)), agencies

    return target, 1.0, agencies


def prediction_sigma(row: pd.Series) -> float:
    return (float(row["upper"]) - float(row["lower"])) / (2.0 * Z_80)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(0.8)


def plot_task_distribution(
    task: str,
    row: pd.Series,
    x: np.ndarray,
    target_y: np.ndarray,
    target_mu: float,
    target_sigma: float,
    y_limit: float,
) -> Path:
    prediction_mu = float(row["prediction"])
    prediction_sd = prediction_sigma(row)
    prediction_y = gaussian_pdf(x, prediction_mu, prediction_sd)

    fig, ax = plt.subplots(figsize=mm_size(FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM))
    ax.fill_between(x, target_y, color=TARGET_FILL, alpha=1.0, linewidth=0)
    ax.plot(x, target_y, color=TARGET_COLOR, linewidth=1.5)
    ax.fill_between(x, prediction_y, color=MODEL_FILL, alpha=0.55, linewidth=0)
    ax.plot(x, prediction_y, color=MODEL_COLOR, linewidth=1.5)

    ax.axvline(target_mu, color=TARGET_COLOR, linewidth=0.8, linestyle="--", alpha=0.65)
    ax.axvline(prediction_mu, color=MODEL_COLOR, linewidth=0.8, linestyle="--", alpha=0.65)

    handles = [
        Line2D([0], [0], color=TARGET_COLOR, lw=1.5, label="Target consensus"),
        Patch(facecolor=MODEL_FILL, edgecolor=MODEL_COLOR, linewidth=0.8, label=MODEL_LABEL),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        handlelength=1.8,
        borderaxespad=0.0,
    )

    ax.set_xlabel("Minimum sea-level pressure (hPa)", labelpad=7)
    ax.set_ylabel("Probability density", labelpad=8)
    ax.set_xlim(880, 1060)
    ax.set_ylim(0, y_limit)
    ax.set_xticks(np.arange(880, 1061, 40))
    style_axes(ax)

    fig.subplots_adjust(left=0.23, right=0.91, top=0.78, bottom=0.24)
    output_stem = OUTPUT_DIR / OUTPUT_STEMS[task]
    save_figure(fig, output_stem)
    plt.close(fig)
    return output_stem.with_suffix(".png")


def main() -> None:
    print("=== Loading probabilistic predictions from stv1/full ===")
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading predictions from {MODEL_DIR.resolve()}")
    print(f"Initialization timestamp: {INITIALIZATION_TIMESTAMP}")
    rows = load_prediction_rows()

    x = np.linspace(880, 1060, 1000)
    target_params = {
        task: target_distribution(rows[task])
        for task in TASK_ORDER
    }
    target_y_values = {
        task: gaussian_pdf(x, target_params[task][0], target_params[task][1])
        for task in TASK_ORDER
    }
    model_y_values = {
        task: gaussian_pdf(x, float(rows[task]["prediction"]), prediction_sigma(rows[task]))
        for task in TASK_ORDER
    }
    y_limit = float(
        max(
            *(curve.max() for curve in target_y_values.values()),
            *(curve.max() for curve in model_y_values.values()),
        )
        * 1.15
    )

    for task in TASK_ORDER:
        row = rows[task]
        valid_timestamp = INITIALIZATION_TIMESTAMP + pd.Timedelta(hours=TASK_TO_HORIZON_HOURS[task])
        target_mu, target_sigma, agencies = target_params[task]
        sigma = prediction_sigma(row)
        print(
            f"{task}: valid={valid_timestamp}, "
            f"target_mean={target_mu:.4f}, target_std={target_sigma:.4f}, "
            f"pred_mean={float(row['prediction']):.4f}, pred_std={sigma:.4f}, "
            f"agencies={', '.join(f'{value:.1f}' for value in agencies)}"
        )
        output_path = plot_task_distribution(
            task,
            row,
            x,
            target_y_values[task],
            target_mu,
            target_sigma,
            y_limit,
        )
        print(f"Saved {output_path}")

    print("=== Probabilistic distribution plots created successfully ===")


if __name__ == "__main__":
    main()
