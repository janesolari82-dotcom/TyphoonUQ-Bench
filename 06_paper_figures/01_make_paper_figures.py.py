from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent.parent.parent / "full"
FORMAL_RUN = ROOT / "outputs" / "formal_2026-04-24_202126"
FIGURE_DIR = Path(__file__).resolve().parent / "figures_png"
SINGLE_FIGURE_DIR = FIGURE_DIR / "single_panels"
SOURCE_DATA_DIR = ROOT / "paper" / "nature_figures" / "source_data"

MM_TO_IN = 1 / 25.4
SINGLE_PANEL_TEXT_SCALE = 1.22
SINGLE_PANEL_AXES_WIDTH_SCALE = 0.90
SINGLE_PANEL_AXES_HEIGHT_SCALE = 0.88
FIGURE3_SINGLE_PANEL_TEXT_SCALE = 1.34
FIGURE3_SINGLE_PANEL_AXES_WIDTH_SCALE = 0.90
FIGURE3_SINGLE_PANEL_AXES_HEIGHT_SCALE = 0.88

MODEL_SPECS = [
    {
        "key": "metadata",
        "label": "Metadata baseline",
        "color": "#7C8894",
        "prediction_rel": None,
    },
    {
        "key": "torch_image_ascend",
        "label": "Image-only (tiny)",
        "color": "#A7D0E8",
        "prediction_rel": "torch_image_ascend",
    },
    {
        "key": "torch_image_resnet18_ascend",
        "label": "Image-only (ResNet18)",
        "color": "#2E6F95",
        "prediction_rel": "torch_image_resnet18_ascend",
    },
    {
        "key": "torch_multimodal_ascend",
        "label": "Multimodal (tiny)",
        "color": "#B5D8C0",
        "prediction_rel": "torch_multimodal_ascend",
    },
    {
        "key": "torch_multimodal_resnet18_ascend",
        "label": "Multimodal (ResNet18)",
        "color": "#4C8C68",
        "prediction_rel": "torch_multimodal_resnet18_ascend",
    },
]

MODEL_LABEL_MAP = {spec["key"]: spec["label"] for spec in MODEL_SPECS}
MODEL_COLOR_MAP = {spec["key"]: spec["color"] for spec in MODEL_SPECS}

TASK_ORDER = ["analysis-0h", "forecast-6h", "forecast-12h"]
TASK_LABELS = {
    "analysis-0h": "Analysis (0 h)",
    "forecast-6h": "Forecast (+6 h)",
    "forecast-12h": "Forecast (+12 h)",
}
TASK_MARKERS = {
    "analysis-0h": "o",
    "forecast-6h": "s",
    "forecast-12h": "^",
}

AGENCY_COLUMNS = {
    "JMA": "pressure_jma",
    "JTWC": "pressure_jtwc",
    "CMA": "pressure_cma",
    "HKO": "pressure_hko",
}
AGENCY_COLORS = {
    "JMA": "#3C6997",
    "JTWC": "#9C6FB0",
    "CMA": "#D17C45",
    "HKO": "#4F8A5B",
}


def setup_style() -> None:
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 6.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    sns.set_style("whitegrid", {"grid.color": "#E5EBEF", "grid.linewidth": 0.6})


def mm_size(width_mm: float, height_mm: float) -> tuple[float, float]:
    return width_mm * MM_TO_IN, height_mm * MM_TO_IN


def save_pub_figure(fig: plt.Figure, stem: Path, width_mm: float, height_mm: float) -> None:
    fig.set_size_inches(*mm_size(width_mm, height_mm))
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def emphasize_single_panel(fig: plt.Figure) -> None:
    if getattr(fig, "_single_panel_emphasis_applied", False):
        return

    axes_width_scale = getattr(fig, "_single_panel_axes_width_scale", SINGLE_PANEL_AXES_WIDTH_SCALE)
    axes_height_scale = getattr(fig, "_single_panel_axes_height_scale", SINGLE_PANEL_AXES_HEIGHT_SCALE)
    text_scale = getattr(fig, "_single_panel_text_scale", SINGLE_PANEL_TEXT_SCALE)

    for text in fig.findobj(match=Text):
        text.set_fontsize(text.get_fontsize() * text_scale)

    for ax in fig.axes:
        pos = ax.get_position()
        width = pos.width * axes_width_scale
        height = pos.height * axes_height_scale
        ax.set_position(
            [
                pos.x0 + (pos.width - width) / 2,
                pos.y0 + (pos.height - height),
                width,
                height,
            ]
        )

    fig._single_panel_emphasis_applied = True


def save_single_figure(fig: plt.Figure, stem: Path, width_mm: float, height_mm: float) -> None:
    fig.set_size_inches(*mm_size(width_mm, height_mm))
    emphasize_single_panel(fig)
    transparent_png = getattr(fig, "_single_panel_transparent_png", False)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", transparent=transparent_png)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def label_model_name(model_key: str) -> str:
    return MODEL_LABEL_MAP.get(model_key, model_key)


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_parquet(ROOT / "data" / "processed" / "benchmark" / "benchmark_manifest.parquet")
    manifest["timestamp"] = pd.to_datetime(manifest["timestamp"])
    manifest["observed_agencies"] = manifest[list(AGENCY_COLUMNS.values())].notna().sum(axis=1)
    manifest["has_any_image"] = manifest["has_png"].fillna(False) | manifest["has_h5"].fillna(False)
    return manifest


def build_dataset_tables(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    labeled = manifest.dropna(subset=["pressure_consensus_median"]).copy()
    yearly = (
        labeled.groupby(["year", "observed_agencies"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[1, 2, 3, 4], fill_value=0)
        .reset_index()
    )
    totals = labeled.groupby("year").size().rename("labeled_timestamps").reset_index()
    image_ready = (
        labeled.groupby("year")["has_any_image"].mean().rename("image_ready_fraction").reset_index()
    )
    yearly = yearly.merge(totals, on="year").merge(image_ready, on="year")

    agency_coverage = pd.DataFrame(
        {
            "agency": list(AGENCY_COLUMNS.keys()),
            "coverage_fraction": [labeled[column].notna().mean() for column in AGENCY_COLUMNS.values()],
        }
    )
    agency_coverage["coverage_percent"] = agency_coverage["coverage_fraction"] * 100

    range_by_agency_count = labeled.loc[labeled["observed_agencies"] >= 2, ["pressure_range", "observed_agencies"]].copy()
    range_by_agency_count["observed_agencies"] = range_by_agency_count["observed_agencies"].astype(int)

    extremes = (
        labeled.loc[labeled["observed_agencies"] >= 2, ["storm_name", "timestamp", "pressure_range"] + list(AGENCY_COLUMNS.values())]
        .sort_values("pressure_range", ascending=False)
        .head(8)
        .copy()
    )
    extremes["event_label"] = extremes.apply(
        lambda row: f"{row['storm_name']}\n{row['timestamp']:%m-%d %H:%M}", axis=1
    )
    extremes["short_label"] = extremes["event_label"]
    extremes = extremes.sort_values("pressure_range", ascending=True)

    summary = pd.DataFrame(
        [
            {
                "metric": "benchmark_storms",
                "value": labeled["storm_id"].nunique(),
            },
            {
                "metric": "labeled_timestamps",
                "value": len(labeled),
            },
            {
                "metric": "full_manifest_timestamps",
                "value": len(manifest),
            },
            {
                "metric": "image_ready_fraction",
                "value": labeled["has_any_image"].mean(),
            },
            {
                "metric": "pressure_range_median",
                "value": labeled["pressure_range"].median(),
            },
            {
                "metric": "pressure_range_p90",
                "value": labeled["pressure_range"].quantile(0.90),
            },
            {
                "metric": "pressure_range_p99",
                "value": labeled["pressure_range"].quantile(0.99),
            },
            {
                "metric": "pressure_range_max",
                "value": labeled["pressure_range"].max(),
            },
        ]
    )

    return {
        "yearly": yearly,
        "agency_coverage": agency_coverage,
        "range_by_agency_count": range_by_agency_count,
        "extremes": extremes,
        "summary": summary,
    }


def build_results_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in MODEL_SPECS:
        for task in TASK_ORDER:
            metrics_path = FORMAL_RUN / spec["key"] / task / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            test_metrics = metrics["metrics_by_split"]["test"]
            high_disagreement = test_metrics["breakdowns"]["high_disagreement"]
            rows.append(
                {
                    "model_key": spec["key"],
                    "model_label": spec["label"],
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "mae": test_metrics["mae"],
                    "rmse": test_metrics["rmse"],
                    "interval_coverage": test_metrics["interval_coverage"],
                    "interval_width": test_metrics["interval_width"],
                    "range_coverage_at_90": test_metrics.get("range_coverage_at_90"),
                    "high_disagreement_mae": high_disagreement["mae"],
                    "high_disagreement_coverage": high_disagreement["interval_coverage"],
                    "high_disagreement_count": high_disagreement["count"],
                    "mae_penalty": high_disagreement["mae"] - test_metrics["mae"],
                    "model_color": spec["color"],
                }
            )
    results = pd.DataFrame(rows)
    metadata_reference = (
        results.loc[results["model_key"] == "metadata", ["task", "mae"]]
        .rename(columns={"mae": "metadata_mae"})
    )
    results = results.merge(metadata_reference, on="task", how="left")
    results["mae_gain_vs_metadata"] = results["metadata_mae"] - results["mae"]
    return results


def load_prediction_slice(model_key: str, task: str, storm_id: str) -> pd.DataFrame:
    predictions_path = FORMAL_RUN / model_key / task / "predictions.parquet"
    predictions = pd.read_parquet(predictions_path)
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    predictions = predictions.loc[predictions["storm_id"].astype(str) == storm_id].copy()
    predictions["model_key"] = model_key
    predictions["model_label"] = label_model_name(model_key)
    predictions["model_color"] = MODEL_COLOR_MAP[model_key]
    return predictions


def build_haiyan_tables(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    haiyan_manifest = (
        manifest.loc[manifest["storm_id"].astype(str) == "201330", ["storm_id", "storm_name", "timestamp"] + list(AGENCY_COLUMNS.values()) + ["pressure_consensus_median", "pressure_min", "pressure_max", "pressure_range"]]
        .copy()
    )
    haiyan_manifest = haiyan_manifest.dropna(subset=["pressure_consensus_median"]).sort_values("timestamp")

    prediction_frames = []
    selected_models = ["metadata", "torch_image_resnet18_ascend", "torch_multimodal_resnet18_ascend"]
    for task in TASK_ORDER:
        for model_key in selected_models:
            prediction_frames.append(load_prediction_slice(model_key, task, "201330"))
    haiyan_predictions = pd.concat(prediction_frames, ignore_index=True)

    return {
        "manifest": haiyan_manifest,
        "predictions": haiyan_predictions,
    }


def export_source_data(dataset_tables: dict[str, pd.DataFrame], results: pd.DataFrame, haiyan: dict[str, pd.DataFrame]) -> None:
    for name, table in dataset_tables.items():
        table.to_csv(SOURCE_DATA_DIR / f"figure1_{name}.csv", index=False)
    results.to_csv(SOURCE_DATA_DIR / "figure2_model_metrics.csv", index=False)
    haiyan["manifest"].to_csv(SOURCE_DATA_DIR / "figure3_haiyan_manifest.csv", index=False)
    haiyan["predictions"].to_csv(SOURCE_DATA_DIR / "figure3_haiyan_predictions.csv", index=False)


def plot_dataset_figure(dataset_tables: dict[str, pd.DataFrame]) -> None:
    yearly = dataset_tables["yearly"]
    agency_coverage = dataset_tables["agency_coverage"]
    range_by_agency_count = dataset_tables["range_by_agency_count"]
    extremes = dataset_tables["extremes"]
    summary = dataset_tables["summary"].set_index("metric")["value"]

    fig = plt.figure(figsize=mm_size(183, 134))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.22, 1.0], width_ratios=[1.25, 0.95, 1.35], hspace=0.55, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, :])
    observed_colors = {1: "#DCE5EB", 2: "#AFC3CF", 3: "#6F93A7", 4: "#284B63"}
    bottom = np.zeros(len(yearly))
    for count in [1, 2, 3, 4]:
        ax_a.bar(
            yearly["year"],
            yearly[count],
            bottom=bottom,
            color=observed_colors[count],
            edgecolor="white",
            linewidth=0.2,
            width=0.86,
            label=f"{count} agency" if count == 1 else f"{count} agencies",
        )
        bottom += yearly[count].to_numpy()
    ax_a.set_title("Benchmark support is both long-term and multi-agency", pad=8, loc="left", fontweight="bold")
    ax_a.set_xlabel("Season")
    ax_a.set_ylabel("Labeled timestamps")
    ax_a.set_xlim(yearly["year"].min() - 0.7, yearly["year"].max() + 0.7)
    ax_a.set_xticks(np.arange(1988, 2024, 4))
    ax_a.tick_params(axis="x", rotation=0)
    ax_a.grid(axis="y", alpha=0.6)

    ax_a2 = ax_a.twinx()
    ax_a2.plot(
        yearly["year"],
        yearly["image_ready_fraction"] * 100,
        color="#C05A36",
        linewidth=1.5,
        marker="o",
        markersize=2.4,
    )
    ax_a2.set_ylim(85, 101)
    ax_a2.set_ylabel("PNG/H5 ready (%)", color="#C05A36")
    ax_a2.tick_params(axis="y", colors="#C05A36")
    ax_a2.spines["top"].set_visible(False)

    summary_text = (
        f"{int(summary['benchmark_storms'])} storms   "
        f"{int(summary['labeled_timestamps']):,} labeled timestamps   "
        f"{summary['image_ready_fraction'] * 100:.1f}% image-ready"
    )
    ax_a.text(
        0.01,
        1.02,
        summary_text,
        transform=ax_a.transAxes,
        fontsize=6.4,
        color="#43515A",
        ha="left",
        va="bottom",
    )
    handles = [
        Patch(facecolor=observed_colors[count], edgecolor="none", label=f"{count} observed agency" if count == 1 else f"{count} observed agencies")
        for count in [1, 2, 3, 4]
    ]
    handles.append(Line2D([0], [0], color="#C05A36", marker="o", lw=1.5, markersize=3, label="PNG/H5 ready share"))
    ax_a.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.53, -0.17), columnspacing=1.0, handlelength=1.6)
    add_panel_label(ax_a, "A")

    ax_b = fig.add_subplot(gs[1, 0])
    agency_coverage = agency_coverage.sort_values("coverage_percent")
    ax_b.barh(
        agency_coverage["agency"],
        agency_coverage["coverage_percent"],
        color=[AGENCY_COLORS[agency] for agency in agency_coverage["agency"]],
        edgecolor="none",
        height=0.6,
    )
    for _, row in agency_coverage.iterrows():
        ax_b.text(row["coverage_percent"] + 1.2, row["agency"], f"{row['coverage_percent']:.1f}%", va="center", fontsize=6.2)
    ax_b.set_xlim(0, 110)
    ax_b.set_xlabel("Coverage among labeled timestamps")
    ax_b.set_title("Per-agency label support", loc="left", fontweight="bold")
    ax_b.grid(axis="x", alpha=0.6)
    add_panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[1, 1])
    sns.violinplot(
        data=range_by_agency_count,
        x="observed_agencies",
        y="pressure_range",
        hue="observed_agencies",
        order=[2, 3, 4],
        palette=[observed_colors[2], observed_colors[3], observed_colors[4]],
        inner=None,
        cut=0,
        linewidth=0.7,
        legend=False,
        ax=ax_c,
    )
    sns.boxplot(
        data=range_by_agency_count,
        x="observed_agencies",
        y="pressure_range",
        order=[2, 3, 4],
        width=0.28,
        showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": "#33444F", "linewidth": 0.7},
        whiskerprops={"color": "#33444F", "linewidth": 0.7},
        medianprops={"color": "#33444F", "linewidth": 0.9},
        flierprops={"marker": "", "markersize": 0},
        ax=ax_c,
    )
    ax_c.set_xlabel("Observed agencies at target time")
    ax_c.set_ylabel("Pressure disagreement range (hPa)")
    ax_c.set_title("Disagreement by agency count", loc="left", fontweight="bold")
    ax_c.set_ylim(0, max(24, range_by_agency_count["pressure_range"].quantile(0.995)))
    ax_c.text(
        0.98,
        0.98,
        f"Median = {summary['pressure_range_median']:.1f} hPa\nP90 = {summary['pressure_range_p90']:.1f} hPa\nP99 = {summary['pressure_range_p99']:.1f} hPa",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=6.1,
        color="#43515A",
    )
    add_panel_label(ax_c, "C")

    ax_d = fig.add_subplot(gs[1, 2])
    y_positions = np.arange(len(extremes))
    ax_d.hlines(y_positions, extremes[list(AGENCY_COLUMNS.values())].min(axis=1), extremes[list(AGENCY_COLUMNS.values())].max(axis=1), color="#CBD5DB", linewidth=2.0)
    for agency, column in AGENCY_COLUMNS.items():
        ax_d.scatter(
            extremes[column],
            y_positions,
            s=20,
            color=AGENCY_COLORS[agency],
            label=agency,
            zorder=3,
            edgecolor="white",
            linewidth=0.3,
        )
    for idx, (_, row) in enumerate(extremes.iterrows()):
        ax_d.text(
            row[list(AGENCY_COLUMNS.values())].max() + 1.4,
            idx,
            f"range {row['pressure_range']:.0f}",
            va="center",
            fontsize=6.0,
            color="#43515A",
        )
    ax_d.set_yticks(y_positions)
    ax_d.set_yticklabels(extremes["short_label"], fontsize=5.7)
    ax_d.tick_params(axis="y", pad=1)
    ax_d.set_xlabel("Agency-reported pressure (hPa)")
    ax_d.set_title("Largest disagreement cases", loc="left", fontweight="bold")
    pressure_values = extremes[list(AGENCY_COLUMNS.values())]
    ax_d.set_xlim(pressure_values.min().min() - 3, pressure_values.max().max() + 10)
    ax_d.grid(axis="x", alpha=0.6)
    ax_d.legend(loc="lower right", ncol=2, columnspacing=0.8, handletextpad=0.3)
    add_panel_label(ax_d, "D")

    fig.suptitle(
        "Figure 1 | TyphoonUQ-Bench preserves long-term satellite coverage while keeping multi-agency pressure disagreement explicit.",
        x=0.01,
        y=1.02,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_pub_figure(fig, FIGURE_DIR / "figure1_dataset_overview", 183, 126)
    plt.close(fig)


def plot_results_figure(results: pd.DataFrame) -> None:
    fig = plt.figure(figsize=mm_size(183, 136))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.08, 1.0], width_ratios=[1, 1, 1], hspace=0.42, wspace=0.32)

    model_order = (
        results.groupby("model_key")["mae"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    y_positions = np.arange(len(model_order))

    top_axes = [fig.add_subplot(gs[0, idx]) for idx in range(3)]
    for idx, task in enumerate(TASK_ORDER):
        ax = top_axes[idx]
        task_df = (
            results.loc[results["task"] == task, ["model_key", "mae", "model_label", "model_color"]]
            .set_index("model_key")
            .loc[model_order]
            .reset_index()
        )
        for row_idx, (_, row) in enumerate(task_df.iterrows()):
            ax.hlines(row_idx, 0, row["mae"], color="#E2E8ED", linewidth=1.8)
            ax.scatter(row["mae"], row_idx, s=34, color=row["model_color"], edgecolor="white", linewidth=0.4, zorder=3)
            ax.text(row["mae"] + 0.45, row_idx, f"{row['mae']:.2f}", va="center", fontsize=6.2, color="#43515A")
        ax.set_title(TASK_LABELS[task], loc="left", fontweight="bold")
        ax.set_xlabel("Test MAE (hPa)")
        ax.set_xlim(0, results["mae"].max() * 1.12)
        ax.set_yticks(y_positions)
        if idx == 0:
            ax.set_yticklabels([label_model_name(model_key) for model_key in model_order])
            ax.set_ylabel("Model family")
        else:
            ax.set_yticklabels([])
        ax.grid(axis="x", alpha=0.6)
        add_panel_label(ax, chr(ord("A") + idx))

    ax_d = fig.add_subplot(gs[1, 0])
    for _, row in results.iterrows():
        ax_d.scatter(
            row["interval_width"],
            row["interval_coverage"],
            s=34,
            color=row["model_color"],
            marker=TASK_MARKERS[row["task"]],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    ax_d.axhline(0.8, color="#43515A", linestyle="--", linewidth=0.9)
    ax_d.set_xlabel("80% interval width (hPa)")
    ax_d.set_ylabel("Observed 80% coverage")
    ax_d.set_title("Coverage-width tradeoff", loc="left", fontweight="bold")
    ax_d.grid(alpha=0.6)
    model_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLOR_MAP[key], markeredgecolor="white", markeredgewidth=0.4, markersize=5.5, label=label_model_name(key))
        for key in model_order
    ]
    task_handles = [
        Line2D([0], [0], marker=TASK_MARKERS[task], color="#43515A", linestyle="none", markersize=5.0, label=TASK_LABELS[task])
        for task in TASK_ORDER
    ]
    ax_d.legend(handles=model_handles + task_handles, loc="lower right", fontsize=5.8, handletextpad=0.4)
    add_panel_label(ax_d, "D")

    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.plot([0, results["high_disagreement_mae"].max() * 1.05], [0, results["high_disagreement_mae"].max() * 1.05], color="#B8C4CC", linewidth=1.0, linestyle="--")
    for _, row in results.iterrows():
        ax_e.scatter(
            row["mae"],
            row["high_disagreement_mae"],
            s=34,
            color=row["model_color"],
            marker=TASK_MARKERS[row["task"]],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    ax_e.set_xlabel("Overall test MAE (hPa)")
    ax_e.set_ylabel("High-disagreement MAE (hPa)")
    ax_e.set_title("High-disagreement subset", loc="left", fontweight="bold")
    ax_e.grid(alpha=0.6)
    add_panel_label(ax_e, "E")

    ax_f = fig.add_subplot(gs[1, 2])
    heatmap_models = [key for key in model_order if key != "metadata"]
    heatmap = (
        results.loc[results["model_key"].isin(heatmap_models), ["model_key", "task", "mae_gain_vs_metadata"]]
        .pivot(index="model_key", columns="task", values="mae_gain_vs_metadata")
        .loc[heatmap_models, TASK_ORDER]
    )
    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".2f",
        cmap=sns.light_palette("#2E6F95", as_cmap=True),
        cbar_kws={"label": "MAE gain vs metadata (hPa)"},
        linewidths=0.6,
        linecolor="white",
        ax=ax_f,
    )
    ax_f.set_xticklabels([TASK_LABELS[task] for task in TASK_ORDER], rotation=20, ha="right")
    ax_f.set_yticklabels([label_model_name(model_key) for model_key in heatmap.index], rotation=0)
    ax_f.set_xlabel("")
    ax_f.set_ylabel("")
    ax_f.set_title("Gain over metadata", loc="left", fontweight="bold")
    add_panel_label(ax_f, "F")

    fig.suptitle(
        "Figure 2 | Under the formal benchmark run, the strongest image-only baseline dominates point accuracy across all three tasks.",
        x=0.01,
        y=1.02,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_pub_figure(fig, FIGURE_DIR / "figure2_benchmark_results", 183, 138)
    plt.close(fig)


def plot_haiyan_figure(haiyan: dict[str, pd.DataFrame]) -> None:
    manifest = haiyan["manifest"].copy()
    predictions = haiyan["predictions"].copy()
    predictions = predictions.sort_values("timestamp")

    fig = plt.figure(figsize=mm_size(183, 135))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.08, 1.0], width_ratios=[1, 1, 1], hspace=0.38, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, :])
    ax_a.fill_between(
        manifest["timestamp"],
        manifest["pressure_min"],
        manifest["pressure_max"],
        color="#E4ECEF",
        alpha=1.0,
        label="Agency min-max range",
    )
    ax_a.plot(
        manifest["timestamp"],
        manifest["pressure_consensus_median"],
        color="#20313F",
        linewidth=1.8,
        label="Consensus median",
    )
    for agency, column in AGENCY_COLUMNS.items():
        subset = manifest.dropna(subset=[column])
        ax_a.plot(
            subset["timestamp"],
            subset[column],
            color=AGENCY_COLORS[agency],
            linewidth=1.0,
            alpha=0.9,
            label=agency,
        )
    ax_a.invert_yaxis()
    ax_a.set_ylabel("Minimum sea-level pressure (hPa)")
    ax_a.set_title("Haiyan (2013): the benchmark keeps storm-level agency spread visible", loc="left", fontweight="bold")
    ax_a.xaxis.set_major_locator(mdates.DayLocator())
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_a.grid(alpha=0.6)
    ax_a.legend(loc="upper right", ncol=3, columnspacing=0.8, handlelength=1.8)
    add_panel_label(ax_a, "A")

    bottom_axes = [fig.add_subplot(gs[1, idx], sharey=ax_a) for idx in range(3)]
    task_to_letter = {"analysis-0h": "B", "forecast-6h": "C", "forecast-12h": "D"}
    selected_model_keys = ["metadata", "torch_image_resnet18_ascend", "torch_multimodal_resnet18_ascend"]

    for idx, task in enumerate(TASK_ORDER):
        ax = bottom_axes[idx]
        task_truth = predictions.loc[predictions["task"] == task, ["timestamp", "target", "target_lower", "target_upper"]].drop_duplicates().sort_values("timestamp")
        ax.fill_between(
            task_truth["timestamp"],
            task_truth["target_lower"],
            task_truth["target_upper"],
            color="#E6ECEF",
            alpha=1.0,
            zorder=0,
            label="Target disagreement range",
        )
        ax.plot(
            task_truth["timestamp"],
            task_truth["target"],
            color="#1F2F3B",
            linewidth=1.4,
            label="Consensus target",
            zorder=3,
        )
        for model_key in selected_model_keys:
            model_df = predictions.loc[(predictions["task"] == task) & (predictions["model_key"] == model_key)].copy()
            if model_df.empty:
                continue
            color = MODEL_COLOR_MAP[model_key]
            ax.fill_between(
                model_df["timestamp"],
                model_df["lower"],
                model_df["upper"],
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                model_df["timestamp"],
                model_df["prediction"],
                color=color,
                linewidth=1.2,
                label=label_model_name(model_key),
                zorder=2,
            )
            storm_mae = np.mean(np.abs(model_df["prediction"] - model_df["target"]))
            ax.text(
                0.02,
                0.08 + 0.08 * selected_model_keys.index(model_key),
                f"{label_model_name(model_key)} MAE = {storm_mae:.2f}",
                transform=ax.transAxes,
                fontsize=5.8,
                color=color,
                ha="left",
                va="bottom",
            )
        ax.invert_yaxis()
        ax.set_title(TASK_LABELS[task], loc="left", fontweight="bold")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.grid(alpha=0.6)
        if idx == 0:
            ax.set_ylabel("Pressure (hPa)")
        else:
            ax.set_ylabel("")
        add_panel_label(ax, task_to_letter[task])

    bottom_axes[1].set_xlabel("Timestamp")
    legend_handles = [
        Patch(facecolor="#E6ECEF", edgecolor="none", label="Target disagreement range"),
        Line2D([0], [0], color="#1F2F3B", lw=1.4, label="Consensus target"),
    ] + [
        Line2D([0], [0], color=MODEL_COLOR_MAP[key], lw=1.3, label=label_model_name(key))
        for key in selected_model_keys
    ]
    bottom_axes[2].legend(handles=legend_handles, loc="upper right", fontsize=5.8, handlelength=1.8)

    fig.suptitle(
        "Figure 3 | For Haiyan, model intervals can be read directly against the preserved agency disagreement envelope across analysis and forecast settings.",
        x=0.01,
        y=1.02,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_pub_figure(fig, FIGURE_DIR / "figure3_haiyan_case_study", 183, 135)
    plt.close(fig)


def standalone_title(fig: plt.Figure, text: str) -> None:
    fig.suptitle(text, x=0.01, y=0.98, ha="left", fontsize=10, fontweight="bold")


def result_model_order(results: pd.DataFrame) -> list[str]:
    return results.groupby("model_key")["mae"].mean().sort_values().index.tolist()


def build_model_handles(model_order: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=MODEL_COLOR_MAP[key],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=5.5,
            label=label_model_name(key),
        )
        for key in model_order
    ]


def build_task_handles(tasks: list[str] = TASK_ORDER) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=TASK_MARKERS[task],
            color="#43515A",
            linestyle="none",
            markersize=5.0,
            label=TASK_LABELS[task],
        )
        for task in tasks
    ]


def plot_single_figure1a(dataset_tables: dict[str, pd.DataFrame]) -> None:
    yearly = dataset_tables["yearly"]
    summary = dataset_tables["summary"].set_index("metric")["value"]

    fig, ax = plt.subplots(figsize=mm_size(183, 105))
    standalone_title(fig, "Figure 1A | Benchmark support is both long-term and multi-agency")

    observed_colors = {1: "#DCE5EB", 2: "#AFC3CF", 3: "#6F93A7", 4: "#284B63"}
    bottom = np.zeros(len(yearly))
    for count in [1, 2, 3, 4]:
        ax.bar(
            yearly["year"],
            yearly[count],
            bottom=bottom,
            color=observed_colors[count],
            edgecolor="white",
            linewidth=0.2,
            width=0.86,
            label=f"{count} agency" if count == 1 else f"{count} agencies",
        )
        bottom += yearly[count].to_numpy()

    ax.set_xlabel("Season")
    ax.set_ylabel("Labeled timestamps")
    ax.set_xlim(yearly["year"].min() - 0.7, yearly["year"].max() + 0.7)
    ax.set_xticks(np.arange(1988, 2024, 4))
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.plot(
        yearly["year"],
        yearly["image_ready_fraction"] * 100,
        color="#C05A36",
        linewidth=1.6,
        marker="o",
        markersize=2.6,
    )
    ax2.set_ylim(85, 101)
    ax2.set_ylabel("PNG/H5 ready (%)", color="#C05A36")
    ax2.tick_params(axis="y", colors="#C05A36")
    ax2.spines["top"].set_visible(False)

    summary_text = (
        f"{int(summary['benchmark_storms'])} storms   "
        f"{int(summary['labeled_timestamps']):,} labeled timestamps   "
        f"{summary['image_ready_fraction'] * 100:.1f}% image-ready"
    )
    ax.text(
        0.01,
        1.02,
        summary_text,
        transform=ax.transAxes,
        fontsize=6.5,
        color="#43515A",
        ha="left",
        va="bottom",
    )

    handles = [
        Patch(facecolor=observed_colors[count], edgecolor="none", label=f"{count} observed agency" if count == 1 else f"{count} observed agencies")
        for count in [1, 2, 3, 4]
    ]
    handles.append(Line2D([0], [0], color="#C05A36", marker="o", lw=1.6, markersize=3, label="PNG/H5 ready share"))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        fontsize=6.2,
        columnspacing=1.0,
        handlelength=1.8,
    )

    fig.subplots_adjust(left=0.08, right=0.89, top=0.78, bottom=0.22)
    save_single_figure(fig, SINGLE_FIGURE_DIR / "figure1A_dataset_overview", 183, 105)
    plt.close(fig)


def plot_single_figure1b(dataset_tables: dict[str, pd.DataFrame]) -> None:
    agency_coverage = dataset_tables["agency_coverage"].copy().sort_values("coverage_percent")

    fig, ax = plt.subplots(figsize=mm_size(102, 92))
    standalone_title(fig, "Figure 1B | Per-agency label support")

    ax.barh(
        agency_coverage["agency"],
        agency_coverage["coverage_percent"],
        color=[AGENCY_COLORS[agency] for agency in agency_coverage["agency"]],
        edgecolor="none",
        height=0.6,
    )
    for _, row in agency_coverage.iterrows():
        ax.text(row["coverage_percent"] + 1.2, row["agency"], f"{row['coverage_percent']:.1f}%", va="center", fontsize=6.2)

    ax.set_xlim(0, 110)
    ax.set_xlabel("Coverage among labeled timestamps")
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.22, right=0.95, top=0.84, bottom=0.16)
    save_single_figure(fig, SINGLE_FIGURE_DIR / "figure1B_agency_coverage", 102, 92)
    plt.close(fig)


def plot_single_figure1c(dataset_tables: dict[str, pd.DataFrame]) -> None:
    range_by_agency_count = dataset_tables["range_by_agency_count"].copy()
    summary = dataset_tables["summary"].set_index("metric")["value"]

    fig, ax = plt.subplots(figsize=mm_size(126, 95))
    standalone_title(fig, "Figure 1C | Disagreement grows with the number of observed agencies")

    observed_colors = {2: "#AFC3CF", 3: "#6F93A7", 4: "#284B63"}
    sns.violinplot(
        data=range_by_agency_count,
        x="observed_agencies",
        y="pressure_range",
        hue="observed_agencies",
        order=[2, 3, 4],
        palette=[observed_colors[2], observed_colors[3], observed_colors[4]],
        inner=None,
        cut=0,
        linewidth=0.7,
        legend=False,
        ax=ax,
    )
    sns.boxplot(
        data=range_by_agency_count,
        x="observed_agencies",
        y="pressure_range",
        order=[2, 3, 4],
        width=0.28,
        showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": "#33444F", "linewidth": 0.7},
        whiskerprops={"color": "#33444F", "linewidth": 0.7},
        medianprops={"color": "#33444F", "linewidth": 0.9},
        flierprops={"marker": "", "markersize": 0},
        ax=ax,
    )
    ax.set_xlabel("Observed agencies at target time")
    ax.set_ylabel("Pressure disagreement range (hPa)")
    ax.set_ylim(0, max(24, range_by_agency_count["pressure_range"].quantile(0.995)))
    ax.set_xlim(-0.5, 3.75)
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    ax.text(
        0.985,
        0.965,
        f"Median = {summary['pressure_range_median']:.1f} hPa\nP90 = {summary['pressure_range_p90']:.1f} hPa\nP99 = {summary['pressure_range_p99']:.1f} hPa",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.3,
        color="#43515A",
    )
    legend_handles = [
        Patch(facecolor=observed_colors[count], edgecolor="none", label=f"{count} observed agencies")
        for count in [2, 3, 4]
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.045),
        frameon=False,
        fontsize=6.2,
        handlelength=1.6,
        handletextpad=0.5,
        labelspacing=0.45,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.14, right=0.97, top=0.84, bottom=0.15)
    save_single_figure(fig, SINGLE_FIGURE_DIR / "figure1C_disagreement_by_agency_count", 126, 95)
    plt.close(fig)


def plot_single_figure1d(dataset_tables: dict[str, pd.DataFrame]) -> None:
    extremes = dataset_tables["extremes"].copy()

    fig, ax = plt.subplots(figsize=mm_size(162, 100))
    standalone_title(fig, "Figure 1D | Largest disagreement cases")

    y_positions = np.arange(len(extremes))
    ax.hlines(
        y_positions,
        extremes[list(AGENCY_COLUMNS.values())].min(axis=1),
        extremes[list(AGENCY_COLUMNS.values())].max(axis=1),
        color="#CBD5DB",
        linewidth=2.0,
    )
    for agency, column in AGENCY_COLUMNS.items():
        ax.scatter(
            extremes[column],
            y_positions,
            s=20,
            color=AGENCY_COLORS[agency],
            label=agency,
            zorder=3,
            edgecolor="white",
            linewidth=0.3,
        )
    for idx, (_, row) in enumerate(extremes.iterrows()):
        ax.text(
            row[list(AGENCY_COLUMNS.values())].max() + 1.4,
            idx,
            f"range {row['pressure_range']:.0f}",
            va="center",
            fontsize=6.0,
            color="#43515A",
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(extremes["short_label"], fontsize=5.8)
    ax.tick_params(axis="y", pad=1)
    ax.set_xlabel("Agency-reported pressure (hPa)")
    pressure_values = extremes[list(AGENCY_COLUMNS.values())]
    ax.set_xlim(pressure_values.min().min() - 3, pressure_values.max().max() + 10)
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.58, 0.02),
        ncol=4,
        columnspacing=1.2,
        handletextpad=0.4,
        frameon=False,
        fontsize=6,
    )
    fig.subplots_adjust(left=0.26, right=0.975, top=0.84, bottom=0.22)
    save_single_figure(fig, SINGLE_FIGURE_DIR / "figure1D_largest_disagreement_cases", 162, 100)
    plt.close(fig)


def plot_single_figure2_task(results: pd.DataFrame, task: str) -> None:
    model_order = result_model_order(results)
    task_df = (
        results.loc[results["task"] == task, ["model_key", "mae", "model_label", "model_color"]]
        .set_index("model_key")
        .loc[model_order]
        .reset_index()
    )
    y_positions = np.arange(len(model_order))
    task_letter = {"analysis-0h": "A", "forecast-6h": "B", "forecast-12h": "C"}[task]

    fig, ax = plt.subplots(figsize=mm_size(112, 104))
    standalone_title(fig, f"Figure 2{task_letter} | {TASK_LABELS[task]} test MAE by model family")

    for row_idx, (_, row) in enumerate(task_df.iterrows()):
        ax.hlines(row_idx, 0, row["mae"], color="#E2E8ED", linewidth=1.8)
        ax.scatter(row["mae"], row_idx, s=34, color=row["model_color"], edgecolor="white", linewidth=0.4, zorder=3)
        ax.text(row["mae"] + 0.45, row_idx, f"{row['mae']:.2f}", va="center", fontsize=6.3, color="#43515A")

    ax.set_xlabel("Test MAE (hPa)")
    ax.set_ylabel("Model family")
    ax.set_xlim(0, results["mae"].max() * 1.12)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label_model_name(model_key) for model_key in model_order])
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.24, right=0.95, top=0.84, bottom=0.14)
    save_single_figure(fig, SINGLE_FIGURE_DIR / f"figure2{task_letter}_model_mae", 112, 104)
    plt.close(fig)


def plot_single_figure2d(results: pd.DataFrame) -> None:
    model_order = result_model_order(results)
    model_handles = build_model_handles(model_order)
    task_handles = build_task_handles()

    fig, ax = plt.subplots(figsize=mm_size(130, 106))
    standalone_title(fig, "Figure 2D | Coverage-width tradeoff")

    for _, row in results.iterrows():
        ax.scatter(
            row["interval_width"],
            row["interval_coverage"],
            s=34,
            color=row["model_color"],
            marker=TASK_MARKERS[row["task"]],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    ax.axhline(0.8, color="#43515A", linestyle="--", linewidth=0.9)
    ax.set_xlabel("80% interval width (hPa)")
    ax.set_ylabel("Observed 80% coverage")
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    fig.legend(
        handles=model_handles + task_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        fontsize=5.8,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.14, right=0.97, top=0.84, bottom=0.26)
    save_single_figure(fig, SINGLE_FIGURE_DIR / "figure2D_coverage_width_tradeoff", 130, 106)
    plt.close(fig)


def plot_single_figure2e(results: pd.DataFrame) -> None:
    model_order = result_model_order(results)
    model_handles = build_model_handles(model_order)
    task_handles = build_task_handles()

    fig, ax = plt.subplots(figsize=mm_size(130, 106))
    standalone_title(fig, "Figure 2E | High-disagreement subset")

    limit = max(results["high_disagreement_mae"].max(), results["mae"].max()) * 1.05
    ax.plot([0, limit], [0, limit], color="#B8C4CC", linewidth=1.0, linestyle="--")
    for _, row in results.iterrows():
        ax.scatter(
            row["mae"],
            row["high_disagreement_mae"],
            s=34,
            color=row["model_color"],
            marker=TASK_MARKERS[row["task"]],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    ax.set_xlabel("Overall test MAE (hPa)")
    ax.set_ylabel("High-disagreement MAE (hPa)")
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    fig.legend(
        handles=model_handles + task_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        fontsize=5.8,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.14, right=0.97, top=0.84, bottom=0.26)
    save_single_figure(fig, SINGLE_FIGURE_DIR / "figure2E_high_disagreement_subset", 130, 106)
    plt.close(fig)


def plot_single_figure2f(results: pd.DataFrame) -> None:
    model_order = result_model_order(results)
    heatmap_models = [key for key in model_order if key != "metadata"]
    heatmap = (
        results.loc[results["model_key"].isin(heatmap_models), ["model_key", "task", "mae_gain_vs_metadata"]]
        .pivot(index="model_key", columns="task", values="mae_gain_vs_metadata")
        .loc[heatmap_models, TASK_ORDER]
    )

    fig, ax = plt.subplots(figsize=mm_size(132, 96))
    standalone_title(fig, "Figure 2F | Gain over metadata")

    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".2f",
        cmap=sns.light_palette("#2E6F95", as_cmap=True),
        cbar_kws={"label": "MAE gain vs metadata (hPa)"},
        linewidths=0.6,
        linecolor="white",
        ax=ax,
    )
    ax.set_xticklabels([TASK_LABELS[task] for task in TASK_ORDER], rotation=20, ha="right")
    ax.set_yticklabels([label_model_name(model_key) for model_key in heatmap.index], rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    fig.subplots_adjust(left=0.18, right=0.86, top=0.84, bottom=0.18)
    save_single_figure(fig, SINGLE_FIGURE_DIR / "figure2F_gain_over_metadata", 132, 96)
    plt.close(fig)


def plot_single_figure3a(haiyan: dict[str, pd.DataFrame]) -> None:
    manifest = haiyan["manifest"].copy()

    fig, ax = plt.subplots(figsize=mm_size(183, 100))
    fig._single_panel_transparent_png = True
    fig._single_panel_text_scale = FIGURE3_SINGLE_PANEL_TEXT_SCALE
    fig._single_panel_axes_width_scale = FIGURE3_SINGLE_PANEL_AXES_WIDTH_SCALE
    fig._single_panel_axes_height_scale = FIGURE3_SINGLE_PANEL_AXES_HEIGHT_SCALE
    standalone_title(fig, "Haiyan (2013): storm-level agency spread remains visible")

    ax.fill_between(
        manifest["timestamp"],
        manifest["pressure_min"],
        manifest["pressure_max"],
        color="#E4ECEF",
        alpha=1.0,
        label="Agency min-max range",
    )
    ax.plot(
        manifest["timestamp"],
        manifest["pressure_consensus_median"],
        color="#20313F",
        linewidth=1.8,
        label="Consensus median",
    )
    for agency, column in AGENCY_COLUMNS.items():
        subset = manifest.dropna(subset=[column])
        ax.plot(
            subset["timestamp"],
            subset[column],
            color=AGENCY_COLORS[agency],
            linewidth=1.0,
            alpha=0.9,
            label=agency,
        )
    ax.invert_yaxis()
    ax.set_ylabel("Minimum sea-level pressure (hPa)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, columnspacing=0.9, handlelength=1.8, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.91, bottom=0.20)
    save_single_figure(fig, SINGLE_FIGURE_DIR / "figure3A_haiyan_agency_track", 183, 100)
    plt.close(fig)


def plot_single_figure3_task(haiyan: dict[str, pd.DataFrame], task: str) -> None:
    predictions = haiyan["predictions"].copy().sort_values("timestamp")
    selected_model_keys = ["metadata", "torch_image_resnet18_ascend", "torch_multimodal_resnet18_ascend"]
    task_letter = {"analysis-0h": "B", "forecast-6h": "C", "forecast-12h": "D"}[task]

    fig, ax = plt.subplots(figsize=mm_size(132, 118))
    fig._single_panel_transparent_png = True
    fig._single_panel_text_scale = FIGURE3_SINGLE_PANEL_TEXT_SCALE
    fig._single_panel_axes_width_scale = FIGURE3_SINGLE_PANEL_AXES_WIDTH_SCALE
    fig._single_panel_axes_height_scale = FIGURE3_SINGLE_PANEL_AXES_HEIGHT_SCALE
    standalone_title(fig, f"{TASK_LABELS[task]} case study")

    task_truth = predictions.loc[predictions["task"] == task, ["timestamp", "target", "target_lower", "target_upper"]].drop_duplicates().sort_values("timestamp")
    ax.fill_between(
        task_truth["timestamp"],
        task_truth["target_lower"],
        task_truth["target_upper"],
        color="#E6ECEF",
        alpha=1.0,
        zorder=0,
        label="Target disagreement range",
    )
    ax.plot(
        task_truth["timestamp"],
        task_truth["target"],
        color="#1F2F3B",
        linewidth=1.4,
        label="Consensus target",
        zorder=3,
    )
    for model_key in selected_model_keys:
        model_df = predictions.loc[(predictions["task"] == task) & (predictions["model_key"] == model_key)].copy()
        if model_df.empty:
            continue
        color = MODEL_COLOR_MAP[model_key]
        ax.fill_between(
            model_df["timestamp"],
            model_df["lower"],
            model_df["upper"],
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            model_df["timestamp"],
            model_df["prediction"],
            color=color,
            linewidth=1.2,
            label=label_model_name(model_key),
            zorder=2,
        )
        storm_mae = np.mean(np.abs(model_df["prediction"] - model_df["target"]))
        ax.text(
            0.98,
            0.94 - 0.07 * selected_model_keys.index(model_key),
            f"{label_model_name(model_key)} MAE = {storm_mae:.2f}",
            transform=ax.transAxes,
            fontsize=5.1,
            color=color,
            ha="right",
            va="top",
        )
    ax.invert_yaxis()
    ax.set_ylabel("Pressure (hPa)")
    ax.set_xlabel("Timestamp")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.33),
        ncol=3,
        columnspacing=0.45,
        handlelength=1.5,
        handletextpad=0.4,
        labelspacing=0.55,
        frameon=False,
        fontsize=5.8,
    )
    fig.subplots_adjust(left=0.08, right=0.96, top=0.91, bottom=0.34)
    save_single_figure(fig, SINGLE_FIGURE_DIR / f"figure3{task_letter}_haiyan_case_study", 132, 118)
    plt.close(fig)


def export_single_panel_figures(dataset_tables: dict[str, pd.DataFrame], results: pd.DataFrame, haiyan: dict[str, pd.DataFrame]) -> None:
    SINGLE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_single_figure1a(dataset_tables)
    plot_single_figure1b(dataset_tables)
    plot_single_figure1c(dataset_tables)
    plot_single_figure1d(dataset_tables)

    for task in TASK_ORDER:
        plot_single_figure2_task(results, task)
    plot_single_figure2d(results)
    plot_single_figure2e(results)
    plot_single_figure2f(results)

    plot_single_figure3a(haiyan)
    for task in TASK_ORDER:
        plot_single_figure3_task(haiyan, task)


def main() -> None:
    setup_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SINGLE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    dataset_tables = build_dataset_tables(manifest)
    results = build_results_table()
    haiyan = build_haiyan_tables(manifest)
    export_source_data(dataset_tables, results, haiyan)

    plot_dataset_figure(dataset_tables)
    plot_results_figure(results)
    plot_haiyan_figure(haiyan)
    export_single_panel_figures(dataset_tables, results, haiyan)


if __name__ == "__main__":
    main()
