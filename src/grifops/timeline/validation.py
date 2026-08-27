from abc import ABC, abstractmethod

import pandas as pd

from grifops.timeline.model import (
    TimelineDefectType,
    TimelineGap,
    TimelineGapSegment,
)


class TimelineValidationStrategy(ABC):
    """
    Defines how historical validation intervals are selected
    for evaluating a timeline repair method.

    Validation intervals contain known observations but have the same
    temporal shape as the real gap being evaluated.
    """

    @abstractmethod
    def generate(
        self,
        series: pd.Series,
        gap: TimelineGap,
    ) -> list[TimelineGap]:
        """
        Generate historical validation gaps corresponding to
        the supplied real timeline gap.
        """

        ...


class MatchingWeekdayHourValidationStrategy(
    TimelineValidationStrategy
):
    """
    Generate historical validation intervals that begin on the same
    weekday and hour as the real timeline gap.

    Only intervals occurring completely before the real gap are
    considered. This avoids using future observations when selecting
    validation samples.
    """

    def generate(
        self,
        series: pd.Series,
        gap: TimelineGap,
    ) -> list[TimelineGap]:
        """
        Return historical validation gaps matching the weekday,
        hour, length, and frequency of the real gap.
        """

        candidate_starts = self._find_candidate_starts(
            series=series,
            gap=gap,
        )

        validation_gaps: list[
            TimelineGap
        ] = []

        for start in candidate_starts:
            validation_gap = self._create_validation_gap(
                start=start,
                gap=gap,
            )

            if not self._fits_series(
                series=series,
                gap=validation_gap,
            ):
                continue

            validation_gaps.append(
                validation_gap
            )

        return validation_gaps

    def _find_candidate_starts(
        self,
        series: pd.Series,
        gap: TimelineGap,
    ) -> pd.DatetimeIndex:
        """
        Find timestamps before the real gap that match its
        weekday and hour.
        """

        historical_index = series.index[
            series.index < gap.start
        ]

        return historical_index[
            (
                historical_index.dayofweek
                == gap.start.dayofweek
            )
            & (
                historical_index.hour
                == gap.start.hour
            )
        ]

    def _create_validation_gap(
        self,
        start: pd.Timestamp,
        gap: TimelineGap,
    ) -> TimelineGap:
        """
        Create a synthetic gap with the same length and frequency
        as the real gap.
        """

        validation_index = pd.date_range(
            start=start,
            periods=gap.length,
            freq=gap.frequency,
        )

        segment = TimelineGapSegment(
            start=validation_index[0],
            end=validation_index[-1],
            defect_type=TimelineDefectType.MISSING_VALUE,
        )

        return TimelineGap(
            segments=(segment,),
            frequency=gap.frequency,
        )

    def _fits_series(
        self,
        series: pd.Series,
        gap: TimelineGap,
    ) -> bool:
        """
        Return True when the complete validation interval exists
        inside the supplied series timeline.
        """

        return gap.index.isin(
            series.index
        ).all()
