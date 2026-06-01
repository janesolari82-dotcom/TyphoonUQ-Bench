from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.special import erfinv


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR.parent.parent / "full"
OUTPUT_DIR = SCRIPT_DIR / "figures_png"
PREDICTION_FILE = (
    INPUT_DIR
    / "outputs"
    / "formal_2026-04-24_202126"
    / "torch_multimodal_resnet18_ascend"
    / "forecast-12h"
    / "predictions.parquet"
)
METADATA_BASELINE_FILE = (
    INPUT_DIR
    / "outputs"
    / "formal_2026-04-24_202126"
    / "metadata"
    / "forecast-12h"
    / "predictions.parquet"
)
OUTPUT_FILES = {
    "A": OUTPUT_DIR / "haikui_uq_metrics_real_A_point_accuracy.png",
    "B": OUTPUT_DIR / "haikui_uq_metrics_real_B_interval_calibration.png",
    "C": OUTPUT_DIR / "haikui_uq_metrics_real_C_range_aware_disagreement.png",
}
STORM_ID = "201211"

BASE_FONT_SIZE = 10
TITLE_FONT_SIZE = 14
SUBTITLE_FONT_SIZE = 12
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 10.5
LEGEND_FONT_SIZE = 10
ANNOTATION_FONT_SIZE = 10


COLORS = {
    "blue": "#2E6F95",
    "green": "#4C8C68",
    "yellow": "#D17C45",
    "red": "#C65D4B",
    "darkblue": "#20313F",
    "fill": "#E6ECEF",
    "grid": "#E5EBEF",
    "axis": "#AEB7BF",
    "text": "#262626",
    "note": "#43515A",
}


def setup_style() -> None:
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": SUBTITLE_FONT_SIZE,
            "axes.labelsize": LABEL_FONT_SIZE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    if grid:
        ax.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.6)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=COLORS["text"], length=3, width=0.8)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])


def load_haikui(prediction_file: Path) -> pd.DataFrame:
    if not prediction_file.exists():
        raise FileNotFoundError(f"Missing prediction file: {prediction_file}")

    df = pd.read_parquet(prediction_file)
    df["storm_id"] = df["storm_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hk = df[df["storm_id"] == STORM_ID].sort_values("timestamp").copy()
    if hk.empty:
        raise ValueError(f"No records found for storm_id={STORM_ID} in {prediction_file}")
    return hk


def calibration_curve(hk: pd.DataFrame, expected: np.ndarray) -> list[float]:
    hk = hk.copy()
    hk["error"] = np.abs(hk["target"] - hk["prediction"])
    hk["sigma"] = (hk["upper"] - hk["lower"]) / (2.0 * 1.28155)

    observed = []
    for coverage in expected:
        if coverage == 0:
            observed.append(0.0)
        elif coverage == 1:
            observed.append(1.0)
        else:
            z_score = math.sqrt(2.0) * erfinv(coverage)
            width = z_score * hk["sigma"]
            observed.append(float(np.mean(hk["error"] <= width)))
    return observed


def draw_point_accuracy(ax_a: plt.Axes, hk: pd.DataFrame) -> None:
    hk_head = hk.head(4).reset_index(drop=True)
    y_pos = [3, 2, 1, 0]
    y_labels = [ts.strftime("%m-%d %Hh") for ts in hk_head["timestamp"]]

    for idx, row in hk_head.iterrows():
        err_low = row["prediction"] - row["lower"]
        err_up = row["upper"] - row["prediction"]

        ax_a.errorbar(
            row["prediction"],
            y_pos[idx],
            xerr=[[err_low], [err_up]],
            fmt="o",
            color=COLORS["blue"],
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=4.2,
            capsize=3,
            elinewidth=1.1,
            label="model interval" if idx == 3 else "",
            zorder=2,
        )
        ax_a.plot(
            row["target"],
            y_pos[idx],
            "D",
            color=COLORS["green"],
            markersize=5.0,
            label="Obs. Consensus(True)" if idx == 3 else "",
            zorder=3,
        )

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(y_labels)
    ax_a.set_ylabel("Timestamp")
    ax_a.set_xlabel("Central Pressure (hPa)")
    style_axis(ax_a)
    ax_a.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.42),
        ncol=2,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.4,
        labelspacing=0.8,
    )


def draw_interval_calibration(ax_b: plt.Axes, hk: pd.DataFrame, metadata_hk: pd.DataFrame) -> None:
    expected = np.linspace(0, 1, 11)
    observed_red = calibration_curve(hk, expected)
    observed_blue = calibration_curve(metadata_hk, expected)

    ax_b.plot(expected, expected, "--", color=COLORS["note"], label="Perfect Calibration", linewidth=1.1)

    x_smooth = np.linspace(0, 1, 100)
    spl_red = make_interp_spline(expected, observed_red, k=3)
    spl_blue = make_interp_spline(expected, observed_blue, k=3)
    y_smooth_red = np.clip(spl_red(x_smooth), 0, 1)
    y_smooth_blue = np.clip(spl_blue(x_smooth), 0, 1)

    ax_b.plot(x_smooth, y_smooth_blue, "-", color=COLORS["darkblue"], linewidth=1.4, label="Metadata baseline")
    ax_b.plot(x_smooth, y_smooth_red, "-", color=COLORS["red"], linewidth=1.4, label="Multi-modal ResNet-18")
    ax_b.fill_between(x_smooth, x_smooth, y_smooth_red, color=COLORS["fill"], alpha=1.0, linewidth=0)

    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.set_xlabel("Expected Coverage")
    ax_b.set_ylabel("Observed Coverage")
    style_axis(ax_b)
    ax_b.legend(loc="lower right", handlelength=1.7, labelspacing=0.45)


def prepare_range_data(
    hk: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hk_sub = hk.iloc[10:30].reset_index(drop=True)
    x_cov = np.arange(len(hk_sub))

    pred_lower = hk_sub["lower"].values
    pred_upper = hk_sub["upper"].values
    prediction = hk_sub["prediction"].values
    agency_lower = hk_sub["target_lower"].values
    agency_upper = hk_sub["target_upper"].values
    distance_to_range = np.maximum.reduce(
        [
            agency_lower - prediction,
            prediction - agency_upper,
            np.zeros_like(prediction),
        ]
    )

    x_smooth_c = np.linspace(0, len(hk_sub) - 1, 100)
    spl_p_lower = make_interp_spline(x_cov, pred_lower, k=3)
    spl_p_upper = make_interp_spline(x_cov, pred_upper, k=3)
    spl_a_lower = make_interp_spline(x_cov, agency_lower, k=3)
    spl_a_upper = make_interp_spline(x_cov, agency_upper, k=3)

    return (
        x_smooth_c,
        spl_p_lower(x_smooth_c),
        spl_p_upper(x_smooth_c),
        spl_a_lower(x_smooth_c),
        spl_a_upper(x_smooth_c),
        x_cov,
        distance_to_range,
    )


def draw_interagency_range(
    ax_c1: plt.Axes,
    x_smooth_c: np.ndarray,
    p_low_s: np.ndarray,
    p_up_s: np.ndarray,
    a_low_s: np.ndarray,
    a_up_s: np.ndarray,
) -> None:
    ax_c1.set_title("Inter-agency range", loc="left", fontweight="bold", fontsize=SUBTITLE_FONT_SIZE, pad=12)
    ax_c1.fill_between(
        x_smooth_c,
        p_low_s,
        p_up_s,
        color=COLORS["blue"],
        alpha=0.14,
        linewidth=0,
        label="Predicted Interval",
    )
    ax_c1.fill_between(
        x_smooth_c,
        a_low_s,
        a_up_s,
        color=COLORS["yellow"],
        alpha=0.22,
        linewidth=0,
        label="Inter-agency Range",
    )

    ax_c1.set_xticks([])
    ax_c1.set_ylabel("Central Pressure")
    style_axis(ax_c1)
    ax_c1.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.48),
        ncol=1,
        handlelength=1.6,
        handletextpad=0.5,
        labelspacing=0.8,
    )


def draw_distance_to_range(
    ax_c2: plt.Axes,
    x_cov: np.ndarray,
    distance_to_range: np.ndarray,
) -> None:
    ax_c2.set_title("Distance-to-range", loc="left", fontweight="bold", fontsize=SUBTITLE_FONT_SIZE, pad=10)
    ax_c2.fill_between(
        x_cov,
        0,
        distance_to_range,
        color=COLORS["yellow"],
        alpha=0.24,
        linewidth=0,
        label="Distance outside range",
    )
    ax_c2.plot(
        x_cov,
        distance_to_range,
        color=COLORS["yellow"],
        linewidth=1.6,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    ax_c2.axhline(0, color=COLORS["note"], linewidth=1.0, alpha=0.9)

    max_distance = float(np.max(distance_to_range))
    ax_c2.set_ylim(0, max(1.0, max_distance * 1.25))
    ax_c2.set_xlim(float(x_cov.min()), float(x_cov.max()))

    ax_c2.text(
        0.55,
        1.10,
        "distance from prediction\nto inter-agency range",
        transform=ax_c2.transAxes,
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_FONT_SIZE,
        color=COLORS["note"],
        fontweight="bold",
    )

    ax_c2.set_xticks([0, 5, 10, 15, 19])
    ax_c2.set_xlabel("Forecast step within selected window")
    ax_c2.set_ylabel("Distance (hPa)")
    style_axis(ax_c2)


def save_png(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.2, transparent=False)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.2, transparent=True)
    plt.close(fig)


def plot_real_haikui() -> None:
    setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hk = load_haikui(PREDICTION_FILE)
    metadata_hk = load_haikui(METADATA_BASELINE_FILE)

    fig_a, ax_a = plt.subplots(figsize=(6, 3.0))
    draw_point_accuracy(ax_a, hk)
    fig_a.subplots_adjust(left=0.16, right=0.97, top=0.66, bottom=0.24)
    save_png(fig_a, OUTPUT_FILES["A"])

    fig_b, ax_b = plt.subplots(figsize=(6, 3.25))
    draw_interval_calibration(ax_b, hk, metadata_hk)
    fig_b.subplots_adjust(left=0.24, right=0.88, top=0.88, bottom=0.28)
    save_png(fig_b, OUTPUT_FILES["B"])

    x_smooth_c, p_low_s, p_up_s, a_low_s, a_up_s, x_cov, distance_to_range = prepare_range_data(hk)
    fig_c = plt.figure(figsize=(6, 5.2))
    gs_c = fig_c.add_gridspec(2, 1, hspace=1.10)
    ax_c1 = fig_c.add_subplot(gs_c[0])
    ax_c2 = fig_c.add_subplot(gs_c[1])
    draw_interagency_range(ax_c1, x_smooth_c, p_low_s, p_up_s, a_low_s, a_up_s)
    draw_distance_to_range(ax_c2, x_cov, distance_to_range)
    fig_c.subplots_adjust(left=0.16, right=0.97, top=0.82, bottom=0.16)
    save_png(fig_c, OUTPUT_FILES["C"])

    print(f"Input folder: {INPUT_DIR}")
    for output_file in OUTPUT_FILES.values():
        print(f"Saved single figure to: {output_file}")
        print(f"Saved SVG to: {output_file.with_suffix('.svg')}")


if __name__ == "__main__":
    plot_real_haikui()
