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
    TimelineRepairer,
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

    #
    # Load dataset
    #

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

    #
    # Inspect timeline
    #

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

    #
    # Configure available repair methods
    #

    repair_strategies = {
        strategy.name: strategy
        for strategy in (
            LinearTimelineRepairStrategy(),
            PreviousWeekTimelineRepairStrategy(),
            NextWeekTimelineRepairStrategy(),
            WeeklyAverageTimelineRepairStrategy(),
        )
    }

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
        strategies=repair_strategies.values(),
        metrics=metrics,
    )

    #
    # Evaluate repair methods
    #

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

    #
    # Summarize evaluation results
    #

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

    #
    # Calculate mean MAE and RMSE
    #

    mean_scores = (
        results_df
        .groupby(
            [
                "gap_start",
                "gap_end",
                "method",
                "metric",
            ],
            as_index=False,
        )["score"]
        .mean()
    )

    #
    # Build lookup for real internal gaps
    #

    internal_gaps = {
        (
            gap.start,
            gap.end,
        ): gap
        for gap in gaps
        if (
            gap.boundary
            is TimelineBoundaryType.NONE
        )
    }

    #
    # Select repair strategy for each internal gap
    #

    selected_strategies = {}

    print("\nBest repair methods")
    print("-------------------")

    for (
        gap_start,
        gap_end,
    ), gap_scores in mean_scores.groupby(
        [
            "gap_start",
            "gap_end",
        ]
    ):
        print(
            f"\n{gap_start} -> {gap_end}"
        )

        mae_scores = gap_scores[
            gap_scores["metric"] == "mae"
        ]

        rmse_scores = gap_scores[
            gap_scores["metric"] == "rmse"
        ]

        if (
            mae_scores.empty
            or rmse_scores.empty
        ):
            print(
                "  Cannot select method: "
                "MAE or RMSE results are missing."
            )
            continue

        minimum_mae = mae_scores[
            "score"
        ].min()

        minimum_rmse = rmse_scores[
            "score"
        ].min()

        best_mae_methods = set(
            mae_scores.loc[
                mae_scores["score"] == minimum_mae,
                "method",
            ]
        )

        best_rmse_methods = set(
            rmse_scores.loc[
                rmse_scores["score"] == minimum_rmse,
                "method",
            ]
        )

        #
        # A method wins only if both MAE and RMSE
        # identify it as a best-performing method.
        #

        best_methods = (
            best_mae_methods
            & best_rmse_methods
        )

        if not best_methods:
            print(
                "  No unambiguous best method."
            )

            print(
                "  Lowest MAE: "
                f"{', '.join(sorted(best_mae_methods))} "
                f"({minimum_mae:.2f})"
            )

            print(
                "  Lowest RMSE: "
                f"{', '.join(sorted(best_rmse_methods))} "
                f"({minimum_rmse:.2f})"
            )

            continue

        if len(best_methods) > 1:
            print(
                "  Multiple methods share the lowest "
                "MAE and RMSE:"
            )

            for method in sorted(
                best_methods
            ):
                print(
                    f"    {method}"
                )

            continue

        best_method = next(
            iter(best_methods)
        )

        gap = internal_gaps[
            (
                gap_start,
                gap_end,
            )
        ]

        selected_strategies[
            gap
        ] = repair_strategies[
            best_method
        ]

        print(
            f"  Method: {best_method}"
        )

        print(
            f"  Mean MAE: {minimum_mae:.2f}"
        )

        print(
            f"  Mean RMSE: {minimum_rmse:.2f}"
        )

    #
    # Require a strategy for every internal gap
    #

    if (
        len(selected_strategies)
        != len(internal_gaps)
    ):
        print(
            "\nTimeline repair aborted: "
            "not every internal gap has an "
            "unambiguous repair method."
        )

        return

    #
    # Repair timeline
    #

    print("\nApplied repair methods")
    print("----------------------")

    for gap, strategy in (
        selected_strategies.items()
    ):
        print(
            f"{gap.start} -> {gap.end}: "
            f"{strategy.name}"
        )

    repairer = TimelineRepairer()

    cleaned_dataset = repairer.repair(
        dataset=dataset,
        gaps=gaps,
        strategies=selected_strategies,
    )

    #
    # Validate repaired dataset
    #

    remaining_gaps = inspector.inspect(
        cleaned_dataset
    )

    print("\nTimeline repair")
    print("---------------")

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
