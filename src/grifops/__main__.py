# src/grifops/__main__.py

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
)

from grifops.dataset.adapter import (
    PandasDataFrameAdapter,
)
from grifops.dataset.loader import (
    CsvTimeSeriesLoader,
)
from grifops.timeline.evaluator import (
    TimelineRepairMethodEvaluator,
)
from grifops.timeline.inspector import (
    TimelineInspector,
)
from grifops.timeline.model import (
    TimelineBoundaryType,
)
from grifops.timeline.repairer import (
    LinearTimelineRepairStrategy,
    NextWeekTimelineRepairStrategy,
    PreviousWeekTimelineRepairStrategy,
    WeeklyAverageTimelineRepairStrategy,
)
from grifops.timeline.validation import (
    MatchingWeekdayHourValidationStrategy,
)


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
                f"{segment.start} -> "
                f"{segment.end} "
                f"[{segment.defect_type.value}]"
            )

    repair_strategies = (
        LinearTimelineRepairStrategy(),
        PreviousWeekTimelineRepairStrategy(),
        NextWeekTimelineRepairStrategy(),
        WeeklyAverageTimelineRepairStrategy(),
    )

    metrics = {
        "mae": MeanAbsoluteError(),
        "rmse": MeanSquaredError(
            square_root=True,
        ),
    }

    validation_strategy = (
        MatchingWeekdayHourValidationStrategy()
    )

    evaluator = TimelineRepairMethodEvaluator(
        strategies=repair_strategies,
        metrics=metrics,
    )

    evaluation_results = []

    print("\nRepair method evaluation")
    print("------------------------")

    for gap in gaps:
        if (
            gap.boundary
            is not TimelineBoundaryType.NONE
        ):
            continue

        validation_gaps = (
            validation_strategy.generate(
                series=dataset.target_series,
                gap=gap,
            )
        )

        print(
            f"\nGap: {gap.start} -> {gap.end}"
        )

        print(
            f"Length: {gap.length}"
        )

        print(
            "Historical validation intervals: "
            f"{len(validation_gaps)}"
        )

        results = evaluator.evaluate(
            dataset=dataset,
            gap=gap,
            validation_gaps=validation_gaps,
        )

        evaluation_results.extend(
            results
        )

    if not evaluation_results:
        print(
            "\nNo valid repair method "
            "evaluations were produced."
        )

        return

    results_df = pd.DataFrame(
        asdict(result)
        for result in evaluation_results
    )

    summary = (
        results_df
        .groupby(
            [
                "gap_start",
                "gap_end",
                "method",
                "metric",
            ]
        )["score"]
        .agg(
            [
                "mean",
                "median",
                "count",
            ]
        )
        .round(2)
    )

    print("\nEvaluation summary")
    print("------------------")
    print(summary)

    print("\nBest method by MAE")
    print("------------------")

    mae_results = (
        results_df[
            results_df["metric"] == "mae"
        ]
        .groupby(
            [
                "gap_start",
                "gap_end",
                "method",
            ],
            as_index=False,
        )["score"]
        .mean()
    )

    for (
        gap_start,
        gap_end,
    ), group in mae_results.groupby(
        [
            "gap_start",
            "gap_end",
        ]
    ):
        best = group.loc[
            group["score"].idxmin()
        ]

        print(
            f"{gap_start} -> {gap_end}"
        )

        print(
            f"  Method: {best['method']}"
        )

        print(
            f"  Mean MAE: {best['score']:.2f}"
        )


if __name__ == "__main__":
    main()
