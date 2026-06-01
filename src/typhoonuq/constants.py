"""Project-wide constants."""

AGENCIES = ("jma", "jtwc", "cma", "hko")
IMAGE_BACKENDS = ("png", "h5")
TASKS = ("analysis-0h", "forecast-6h", "forecast-12h")
TASK_TO_HORIZON = {
    "analysis-0h": 0,
    "forecast-6h": 6,
    "forecast-12h": 12,
}
DEFAULT_FORWARD_SPLIT = {
    "train": (1988, 2015),
    "val": (2016, 2019),
    "test": (2020, 2023),
}
CONSENSUS_COLUMNS = (
    "pressure_consensus_median",
    "pressure_min",
    "pressure_max",
    "pressure_range",
    "pressure_std",
    "agency_count",
)
CORE_MANIFEST_COLUMNS = (
    "storm_id",
    "storm_name",
    "timestamp",
    "image_ref",
    "png_ref",
    "h5_ref",
    "quality_flag",
    "lat",
    "lon",
    "year",
    "month",
)
DT_STORM_INDEX_COLUMNS = (
    "storm_id",
    "storm_name",
    "season",
    "start",
    "end",
    "images",
    "normalized_name",
)
DT_FRAME_INDEX_COLUMNS = (
    "storm_id",
    "storm_name",
    "season",
    "timestamp",
    "lat",
    "lon",
    "pressure_dt",
    "wind_dt",
    "grade",
    "intp",
    "h5_ref",
    "png_ref",
    "has_h5",
    "has_png",
)
DEFAULT_FEATURE_COLUMNS = (
    "lat",
    "lon",
    "motion_u",
    "motion_v",
    "motion_speed",
    "quality_flag",
    "month_sin",
    "month_cos",
)
DEFAULT_DRIVE_LAYOUT = {
    "digital_typhoon_archive": "data/raw/digital_typhoon/archive",
    "ibtracs_csv": "data/raw/ibtracs/ibtracs.WP.list.v04r01.csv",
    "jma_best_track": "data/raw/jma/bst_all.txt",
    "dt_processed": "data/processed/digital_typhoon",
    "benchmark_processed": "data/processed/benchmark",
    "outputs": "outputs",
}
