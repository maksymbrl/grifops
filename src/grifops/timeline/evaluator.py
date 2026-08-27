# src/grifops/timeline/evaluator.py

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import pandas as pd

from grifops.dataset import TimeSeriesDataset
from grifops.timeline.model import (
    TimelineBoundaryType,
    TimelineGap,
)
from grifops.timeline.repairer import (
    TimelineRepairStrategy,
)


ForecastMetric = Callable[
    [pd.Series, pd.Series],
    object,
]


@dataclass(frozen=True)
class TimelineRepairMethodEvaluationResult:
    """
    Result of evaluating one timeline repair method on one
    historical validation interval.
    """

    gap_start: pd.Timestamp
    gap_end: pd.Timestamp

    validation_start: pd.Timestamp
    validation_end: pd.Timestamp

    method: str
    metric: str
    score: float


class TimelineRepairMethodEvaluator:
    """
    Evaluates timeline repair methods on known historical data.

    Each validation interval is temporarily masked and reconstructed
    using every supplied repair strategy. The reconstructed values are
    then compared with the known observations using the configured
    evaluation metrics.

    Validation interval generation is intentionally handled elsewhere.
    """

    def __init__(
        self,
        strategies: Iterable[TimelineRepairStrategy],
        metrics: Mapping[str, ForecastMetric],
    ) -> None:
        self._strategies = tuple(
            strategies
        )

        self._metrics = dict(
            metrics
        )

        if not self._strategies:
            raise ValueError(
                "At least one timeline repair strategy is required."
            )

        if not self._metrics:
            raise ValueError(
                "At least one evaluation metric is required."
            )

    def evaluate(
        self,
        dataset: TimeSeriesDataset,
        gap: TimelineGap,
        validation_gaps: Iterable[TimelineGap],
    ) -> list[TimelineRepairMethodEvaluationResult]:
        """
        Evaluate every configured repair method against the supplied
        historical validation gaps.
        """

        self._validate_gap(
            gap
        )

        results: list[
            TimelineRepairMethodEvaluationResult
        ] = []

        for validation_gap in validation_gaps:
            results.extend(
                self._evaluate_validation_gap(
                    dataset=dataset,
                    original_gap=gap,
                    validation_gap=validation_gap,
                )
            )

        return results

    def _evaluate_validation_gap(
        self,
        dataset: TimeSeriesDataset,
        original_gap: TimelineGap,
        validation_gap: TimelineGap,
    ) -> list[TimelineRepairMethodEvaluationResult]:
        """
        Evaluate every configured repair strategy on one
        historical validation interval.
        """

        series = dataset.target_series

        if not self._contains_valid_observations(
            series=series,
            gap=validation_gap,
        ):
            return []

        actual = series.loc[
            validation_gap.index
        ].copy()

        masked = self._mask_gap(
            series=series,
            gap=validation_gap,
        )

        if not self._all_strategies_can_repair(
            series=masked,
            gap=validation_gap,
        ):
            return []

        results: list[
            TimelineRepairMethodEvaluationResult
        ] = []

        for strategy in self._strategies:
            results.extend(
                self._evaluate_strategy(
                    actual=actual,
                    masked=masked,
                    original_gap=original_gap,
                    validation_gap=validation_gap,
                    strategy=strategy,
                )
            )

        return results

    def _evaluate_strategy(
        self,
        actual: pd.Series,
        masked: pd.Series,
        original_gap: TimelineGap,
        validation_gap: TimelineGap,
        strategy: TimelineRepairStrategy,
    ) -> list[TimelineRepairMethodEvaluationResult]:
        """
        Evaluate one repair strategy using every configured metric.
        """

        predicted = strategy.repair(
            series=masked,
            gap=validation_gap,
        )

        self._validate_prediction(
            actual=actual,
            predicted=predicted,
        )

        results: list[
            TimelineRepairMethodEvaluationResult
        ] = []

        for metric_name, metric in self._metrics.items():
            score = metric(
                actual,
                predicted,
            )

            results.append(
                TimelineRepairMethodEvaluationResult(
                    gap_start=original_gap.start,
                    gap_end=original_gap.end,
                    validation_start=validation_gap.start,
                    validation_end=validation_gap.end,
                    method=strategy.name,
                    metric=metric_name,
                    score=float(score),
                )
            )

        return results

    def _mask_gap(
        self,
        series: pd.Series,
        gap: TimelineGap,
    ) -> pd.Series:
        """
        Return a copy of the series with the supplied interval
        replaced by missing values.

        The original known observations remain unchanged and are used
        later as the ground truth for evaluation.
        """

        masked = series.copy()

        masked.loc[
            gap.index
        ] = float("nan")

        return masked

    def _contains_valid_observations(
        self,
        series: pd.Series,
        gap: TimelineGap,
    ) -> bool:
        """
        Return True when the complete validation interval exists
        and contains valid target observations.
        """

        if not gap.index.isin(
            series.index
        ).all():
            return False

        values = series.loc[
            gap.index
        ]

        return not values.isna().any()

    def _all_strategies_can_repair(
        self,
        series: pd.Series,
        gap: TimelineGap,
    ) -> bool:
        """
        Return True when every configured repair strategy can repair
        the validation interval.

        Requiring all methods to be applicable keeps the comparison
        fair because every method is evaluated on the same samples.
        """

        return all(
            strategy.can_repair(
                series=series,
                gap=gap,
            )
            for strategy in self._strategies
        )

    def _validate_gap(
        self,
        gap: TimelineGap,
    ) -> None:
        """
        Ensure that the real gap is suitable for repair-method
        evaluation.
        """

        if (
            gap.boundary
            is not TimelineBoundaryType.NONE
        ):
            raise ValueError(
                "Timeline repair method evaluation requires "
                "an internal timeline gap."
            )

    def _validate_prediction(
        self,
        actual: pd.Series,
        predicted: pd.Series,
    ) -> None:
        """
        Ensure a repair strategy produced a complete prediction
        aligned with the validation interval.
        """

        if not predicted.index.equals(
            actual.index
        ):
            raise ValueError(
                "Timeline repair strategy returned values "
                "with an unexpected timeline index."
            )

        if predicted.isna().any():
            raise ValueError(
                "Timeline repair strategy returned missing values."
            )
