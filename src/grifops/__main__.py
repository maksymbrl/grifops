# src/grifops/__main__.py

from pathlib import Path

import pandas as pd

from grifops.dataset import TimeSeriesDataset
from grifops.data_adapter import PandasDataFrameAdapter
from grifops.data_loader import CsvTimeSeriesLoader
from grifops.timeline_inspector import TimelineInspector
from grifops.timeline_repairer import (
    TimelineBoundaryType,
    TimelineGap,
)


TIMESTAMP_COLUMN = "utc_timestamp"
TARGET_COLUMN = "NO_load_actual_entsoe_transparency"
FREQUENCY = "1h"

DATA_PATH = Path(
    "data/raw/time_series_60min_singleindex.csv"
)


def reconstruct_timeline(
    dataset: TimeSeriesDataset,
    gaps: list[TimelineGap],
) -> TimeSeriesDataset:
    """
    Reconstruct the target timeline using linear interpolation.

    Boundary gaps are removed. Internal gaps are materialized on the
    expected timeline and filled using time-based interpolation.

    This is a temporary implementation that will later be replaced
    by TimelineRepairer and TimelineRepairStrategy.
    """

    offset = pd.tseries.frequencies.to_offset(
        dataset.frequency
    )

    start = dataset.data.index[0]
    end = dataset.data.index[-1]

    # Remove gaps at the boundaries because interpolation requires
    # valid observations on both sides.
    for gap in gaps:
        if gap.boundary is TimelineBoundaryType.START:
            start = gap.end + offset

        elif gap.boundary is TimelineBoundaryType.END:
            end = gap.start - offset

    expected_index = pd.date_range(
        start=start,
        end=end,
        freq=dataset.frequency,
        name=dataset.data.index.name,
    )

    # Reindexing also creates rows for completely missing timestamps.
    reconstructed_data = (
        dataset.data
        .reindex(expected_index)
        .copy()
    )

    reconstructed_data[dataset.target] = (
        reconstructed_data[dataset.target]
        .interpolate(
            method="time",
            limit_area="inside",
        )
    )

    return TimeSeriesDataset(
        data=reconstructed_data,
        target=dataset.target,
        frequency=dataset.frequency,
        expected_start=expected_index[0],
        expected_end=expected_index[-1],
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

    gaps = inspector.inspect(
        dataset
    )

    print("\nTimeline inspection")
    print("-------------------")
    print(f"Detected gaps: {len(gaps)}")

    for index, gap in enumerate(
        gaps,
        start=1,
    ):
        print(
            f"\nGap {index}: "
            f"{gap.start} -> {gap.end}"
        )

        print(
            f"  Length:   {gap.length}"
        )

        print(
            f"  Boundary: {gap.boundary.value}"
        )

        print(
            f"  Mixed:    {gap.is_mixed}"
        )

        for segment in gap.segments:
            print(
                "  Segment:  "
                f"{segment.start} -> {segment.end} "
                f"[{segment.defect_type.value}]"
            )



    cleaned_dataset = reconstruct_timeline(
        dataset=dataset,
        gaps=gaps,
    )

    remaining_gaps = inspector.inspect(
        cleaned_dataset
    )

    print("\nTimeline reconstruction")
    print("-----------------------")
    print(
        f"Rows: {len(cleaned_dataset.data)}"
    )
    print(
        f"Start: {cleaned_dataset.data.index[0]}"
    )
    print(
        f"End: {cleaned_dataset.data.index[-1]}"
    )
    print(
        "Missing target values: "
        f"{cleaned_dataset.target_series.isna().sum()}"
    )
    print(
        f"Remaining gaps: {len(remaining_gaps)}"
    )


if __name__ == "__main__":
    main()
