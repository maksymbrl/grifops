# src/grifops/__main__.py

from pathlib import Path

from grifops.data_adapter import PandasDataFrameAdapter
from grifops.data_loader import CsvTimeSeriesLoader
from grifops.timeline_inspector import TimelineInspector


# The data values that will eventually be supplied by
# either the CLI or a parameter file.
TIMESTAMP_COLUMN = "utc_timestamp"

TARGET_COLUMN = "NO_load_actual_entsoe_transparency"

FREQUENCY = "1h"

DATA_PATH = Path(
    "data/raw/time_series_60min_singleindex.csv"
)


def main() -> None:
    print("GRIFOps")

    adapter = PandasDataFrameAdapter(
        timestamp_column=TIMESTAMP_COLUMN,
        target_column=TARGET_COLUMN,
        frequency=FREQUENCY,
        normalize_to_utc=True,
    )

    loader = CsvTimeSeriesLoader(
        adapter=adapter,
    )

    dataset = loader.load(
        DATA_PATH
    )

    print("\nDataset loaded successfully.")
    print(f"Rows: {len(dataset.data)}")
    print(f"Target: {dataset.target}")
    print(f"Frequency: {dataset.frequency}")
    print(f"Start: {dataset.data.index[0]}")
    print(f"End: {dataset.data.index[-1]}")
    print(
        "Missing target values: "
        f"{dataset.target_series.isna().sum()}"
    )

    inspector = TimelineInspector()
    gaps = inspector.inspect(dataset)

    print(f"Gaps are: {gaps}")


if __name__ == "__main__":
    main()
