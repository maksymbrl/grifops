# src/grifops/timeline/repairer.py

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

import pandas as pd

from grifops.dataset import TimeSeriesDataset
from grifops.timeline.model import (
    TimelineBoundaryType,
    TimelineGap,
)


class TimelineRepairStrategy(ABC):
    """
    Defines how a TimelineGap is reconstructed.

    Concrete strategies implement different reconstruction
    algorithms without modifying the original time series.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of the repair strategy.
        """

        ...

    @abstractmethod
    def required_context(
        self,
        gap: TimelineGap,
    ) -> pd.DatetimeIndex:
        """
        Return the existing timestamps required to repair the gap.
        """

        ...

    def can_repair(
        self,
        series: pd.Series,
        gap: TimelineGap,
    ) -> bool:
        """
        Return True if all observations required by the strategy
        exist and contain valid values.
        """

        required = self.required_context(
            gap
        )

        if not required.isin(
            series.index
        ).all():
            return False

        return not series.loc[
            required
        ].isna().any()

    @abstractmethod
    def repair(
        self,
        series: pd.Series,
        gap: TimelineGap,
    ) -> pd.Series:
        """
        Reconstruct and return the target values within the gap.
        """

        ...


class LinearTimelineRepairStrategy(
    TimelineRepairStrategy
):
    """
    Reconstruct a gap using time-based linear interpolation
    between the valid observations surrounding the gap.
    """

    @property
    def name(self) -> str:
        return "linear"

    def required_context(
        self,
        gap: TimelineGap,
    ) -> pd.DatetimeIndex:
        offset = pd.tseries.frequencies.to_offset(
            gap.frequency
        )

        return pd.DatetimeIndex(
            [
                gap.start - offset,
                gap.end + offset,
            ]
        )

    def repair(
        self,
        series: pd.Series,
        gap: TimelineGap,
    ) -> pd.Series:
        if not self.can_repair(
            series=series,
            gap=gap,
        ):
            raise ValueError(
                "Linear repair requires valid observations "
                "immediately before and after the gap."
            )

        context = self.required_context(
            gap
        )

        repair_index = (
            pd.DatetimeIndex(
                [context[0]]
            )
            .append(
                gap.index
            )
            .append(
                pd.DatetimeIndex(
                    [context[1]]
                )
            )
        )

        repaired = (
            series
            .reindex(repair_index)
            .interpolate(
                method="time",
                limit_area="inside",
            )
        )

        result = repaired.loc[
            gap.index
        ]

        if result.isna().any():
            raise RuntimeError(
                "Linear timeline repair produced missing values."
            )

        return result


class PreviousWeekTimelineRepairStrategy(
    TimelineRepairStrategy
):
    """
    Reconstruct a gap using observations from the same
    timestamps one week earlier.
    """

    @property
    def name(self) -> str:
        return "previous_week"

    def required_context(
        self,
        gap: TimelineGap,
    ) -> pd.DatetimeIndex:
        return (
            gap.index
            - pd.Timedelta(days=7)
        )

    def repair(
        self,
        series: pd.Series,
        gap: TimelineGap,
    ) -> pd.Series:
        if not self.can_repair(
            series=series,
            gap=gap,
        ):
            raise ValueError(
                "Previous-week repair requires valid observations "
                "for the same timestamps one week earlier."
            )

        source_index = self.required_context(
            gap
        )

        values = (
            series
            .loc[source_index]
            .to_numpy(dtype=float)
        )

        return pd.Series(
            values,
            index=gap.index,
            name=series.name,
        )


class NextWeekTimelineRepairStrategy(
    TimelineRepairStrategy
):
    """
    Reconstruct a gap using observations from the same
    timestamps one week later.
    """

    @property
    def name(self) -> str:
        return "next_week"

    def required_context(
        self,
        gap: TimelineGap,
    ) -> pd.DatetimeIndex:
        return (
            gap.index
            + pd.Timedelta(days=7)
        )

    def repair(
        self,
        series: pd.Series,
        gap: TimelineGap,
    ) -> pd.Series:
        if not self.can_repair(
            series=series,
            gap=gap,
        ):
            raise ValueError(
                "Next-week repair requires valid observations "
                "for the same timestamps one week later."
            )

        source_index = self.required_context(
            gap
        )

        values = (
            series
            .loc[source_index]
            .to_numpy(dtype=float)
        )

        return pd.Series(
            values,
            index=gap.index,
            name=series.name,
        )


class WeeklyAverageTimelineRepairStrategy(
    TimelineRepairStrategy
):
    """
    Reconstruct a gap using the average of observations
    one week before and one week after each missing timestamp.
    """

    @property
    def name(self) -> str:
        return "weekly_average"

    def required_context(
        self,
        gap: TimelineGap,
    ) -> pd.DatetimeIndex:
        previous_week = (
            gap.index
            - pd.Timedelta(days=7)
        )

        next_week = (
            gap.index
            + pd.Timedelta(days=7)
        )

        return previous_week.union(
            next_week
        )

    def repair(
        self,
        series: pd.Series,
        gap: TimelineGap,
    ) -> pd.Series:
        if not self.can_repair(
            series=series,
            gap=gap,
        ):
            raise ValueError(
                "Weekly-average repair requires valid observations "
                "one week before and one week after the gap."
            )

        previous_week_index = (
            gap.index
            - pd.Timedelta(days=7)
        )

        next_week_index = (
            gap.index
            + pd.Timedelta(days=7)
        )

        previous_values = (
            series
            .loc[previous_week_index]
            .to_numpy(dtype=float)
        )

        next_values = (
            series
            .loc[next_week_index]
            .to_numpy(dtype=float)
        )

        values = (
            previous_values
            + next_values
        ) / 2.0

        return pd.Series(
            values,
            index=gap.index,
            name=series.name,
        )


class TimelineRepairer:
    """
    Repairs timeline gaps using preselected repair strategies.

    The caller determines which strategy should be used for each
    internal gap. Boundary gaps are removed because the current
    reconstruction policy only repairs gaps bounded by valid
    observations.
    """

    def repair(
        self,
        dataset: TimeSeriesDataset,
        gaps: Iterable[TimelineGap],
        strategies: Mapping[
            TimelineGap,
            TimelineRepairStrategy,
        ],
    ) -> TimeSeriesDataset:
        """
        Return a new TimeSeriesDataset with timeline gaps repaired.

        Start and end boundary gaps are removed. Missing timestamps
        are materialized, and every internal gap is repaired using
        its preselected repair strategy.
        """

        gaps = tuple(
            gaps
        )

        self._validate_gaps(
            dataset=dataset,
            gaps=gaps,
        )

        self._validate_strategies(
            gaps=gaps,
            strategies=strategies,
        )

        start, end = self._repair_boundaries(
            dataset=dataset,
            gaps=gaps,
        )

        reconstructed_data = self._materialize_timeline(
            dataset=dataset,
            start=start,
            end=end,
        )

        for gap in gaps:
            if (
                gap.boundary
                is not TimelineBoundaryType.NONE
            ):
                continue

            strategy = strategies[
                gap
            ]

            repaired_values = strategy.repair(
                series=reconstructed_data[
                    dataset.target
                ],
                gap=gap,
            )

            reconstructed_data.loc[
                repaired_values.index,
                dataset.target,
            ] = repaired_values

        if reconstructed_data[
            dataset.target
        ].isna().any():
            raise RuntimeError(
                "Timeline repair completed with missing "
                "target values remaining."
            )

        return TimeSeriesDataset(
            data=reconstructed_data,
            target=dataset.target,
            frequency=dataset.frequency,
            expected_start=reconstructed_data.index[0],
            expected_end=reconstructed_data.index[-1],
        )

    def _validate_gaps(
        self,
        dataset: TimeSeriesDataset,
        gaps: tuple[TimelineGap, ...],
    ) -> None:
        """
        Ensure all gaps are compatible with the dataset timeline.
        """

        for gap in gaps:
            if gap.frequency != dataset.frequency:
                raise ValueError(
                    "TimelineGap frequency does not match "
                    "the dataset frequency."
                )

    def _validate_strategies(
        self,
        gaps: tuple[TimelineGap, ...],
        strategies: Mapping[
            TimelineGap,
            TimelineRepairStrategy,
        ],
    ) -> None:
        """
        Ensure every internal gap has a selected repair strategy.
        """

        internal_gaps = {
            gap
            for gap in gaps
            if (
                gap.boundary
                is TimelineBoundaryType.NONE
            )
        }

        supplied_gaps = set(
            strategies
        )

        missing_gaps = (
            internal_gaps
            - supplied_gaps
        )

        if missing_gaps:
            raise ValueError(
                "No repair strategy was supplied for "
                f"{len(missing_gaps)} internal timeline gap(s)."
            )

    def _repair_boundaries(
        self,
        dataset: TimeSeriesDataset,
        gaps: tuple[TimelineGap, ...],
    ) -> tuple[
        pd.Timestamp,
        pd.Timestamp,
    ]:
        """
        Determine usable timeline boundaries after removing
        start and end gaps.
        """

        offset = pd.tseries.frequencies.to_offset(
            dataset.frequency
        )

        start = dataset.data.index[0]
        end = dataset.data.index[-1]

        for gap in gaps:
            if (
                gap.boundary
                is TimelineBoundaryType.START
            ):
                start = gap.end + offset

            elif (
                gap.boundary
                is TimelineBoundaryType.END
            ):
                end = gap.start - offset

        if end < start:
            raise ValueError(
                "No usable timeline remains after "
                "removing boundary gaps."
            )

        return start, end

    def _materialize_timeline(
        self,
        dataset: TimeSeriesDataset,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Materialize every expected timestamp between the usable
        timeline boundaries.

        Completely missing timestamps therefore become rows with
        missing values before reconstruction.
        """

        expected_index = pd.date_range(
            start=start,
            end=end,
            freq=dataset.frequency,
            name=dataset.data.index.name,
        )

        return (
            dataset.data
            .reindex(expected_index)
            .copy()
        )
