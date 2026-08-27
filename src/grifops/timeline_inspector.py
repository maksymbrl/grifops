from __future__ import annotations

import pandas as pd

from grifops.dataset import TimeSeriesDataset
from grifops.timeline_repairer import (
    TimelineBoundaryType,
    TimelineDefectType,
    TimelineGap,
    TimelineGapSegment,
)


class TimelineInspector:
    """
    Inspects the temporal integrity of a TimeSeriesDataset.

    Detects missing timestamps and missing target values, groups
    consecutive defects into TimelineGapSegments, and combines
    adjacent segments into TimelineGaps.

    The inspector does not modify or repair the dataset.
    """

    def inspect(
        self,
        dataset: TimeSeriesDataset,
    ) -> list[TimelineGap]:
        """
        Inspect the dataset timeline and return all discovered gaps.
        """

        self._validate_dataset(dataset)

        expected_index = self._build_expected_index(
            dataset
        )

        self._validate_observed_timestamps(
            dataset,
            expected_index,
        )

        defects = self._classify_defects(
            dataset,
            expected_index,
        )

        if not defects:
            return []

        return self._build_gaps(
            defects=defects,
            expected_index=expected_index,
            frequency=dataset.frequency,
        )

    def _validate_dataset(
        self,
        dataset: TimeSeriesDataset,
    ) -> None:
        """
        Validate timeline properties required for reliable inspection.
        """

        if dataset.data.empty:
            raise ValueError(
                "Cannot inspect an empty time-series dataset."
            )

        index = dataset.data.index

        if index.hasnans:
            raise ValueError(
                "Timeline contains observations without valid timestamps."
            )

        if index.has_duplicates:
            raise ValueError(
                "Timeline contains duplicate timestamps."
            )

        if not index.is_monotonic_increasing:
            raise ValueError(
                "Timeline timestamps must be ordered chronologically."
            )

        if not dataset.target_series.notna().any():
            raise ValueError(
                "Timeline contains no valid target observations."
            )

    def _build_expected_index(
        self,
        dataset: TimeSeriesDataset,
    ) -> pd.DatetimeIndex:
        """
        Construct the complete timeline expected for the dataset.

        Explicit expected boundaries are used when supplied.
        Otherwise, the first and last observed timestamps define
        the inspection range.
        """

        observed_index = dataset.data.index

        start = (
            dataset.expected_start
            if dataset.expected_start is not None
            else observed_index[0]
        )

        end = (
            dataset.expected_end
            if dataset.expected_end is not None
            else observed_index[-1]
        )

        try:
            expected_index = pd.date_range(
                start=start,
                end=end,
                freq=dataset.frequency,
            )

        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Unable to construct the expected timeline."
            ) from exc

        if expected_index.empty:
            raise ValueError(
                "Expected timeline contains no observations."
            )

        return expected_index

    def _validate_observed_timestamps(
        self,
        dataset: TimeSeriesDataset,
        expected_index: pd.DatetimeIndex,
    ) -> None:
        """
        Ensure every observed timestamp belongs to the expected timeline.

        Timestamps outside the expected range or not aligned with the
        expected frequency cannot be represented as ordinary gaps.
        """

        unexpected = dataset.data.index.difference(
            expected_index
        )

        if not unexpected.empty:
            raise ValueError(
                "Observed timestamps exist outside or do not align "
                "with the expected timeline: "
                f"{unexpected.tolist()}"
            )

    def _classify_defects(
        self,
        dataset: TimeSeriesDataset,
        expected_index: pd.DatetimeIndex,
    ) -> dict[pd.Timestamp, TimelineDefectType]:
        """
        Classify every defective expected timestamp.

        A missing timestamp means the complete observation row is absent.
        A missing value means the timestamp exists but the target is NaN.
        """

        defects: dict[
            pd.Timestamp,
            TimelineDefectType,
        ] = {}

        observed_index = dataset.data.index

        missing_timestamps = expected_index.difference(
            observed_index
        )

        for timestamp in missing_timestamps:
            defects[timestamp] = (
                TimelineDefectType.MISSING_TIMESTAMP
            )

        missing_values = (
            dataset.target_series[
                dataset.target_series.isna()
            ]
            .index
        )

        for timestamp in missing_values:
            defects[timestamp] = (
                TimelineDefectType.MISSING_VALUE
            )

        return defects

    def _build_gaps(
        self,
        defects: dict[
            pd.Timestamp,
            TimelineDefectType,
        ],
        expected_index: pd.DatetimeIndex,
        frequency: str,
    ) -> list[TimelineGap]:
        """
        Group consecutive defective timestamps into TimelineGaps.
        """

        gaps: list[TimelineGap] = []

        current_gap: list[
            tuple[
                pd.Timestamp,
                TimelineDefectType,
            ]
        ] = []

        for timestamp in expected_index:
            defect_type = defects.get(
                timestamp
            )

            if defect_type is not None:
                current_gap.append(
                    (
                        timestamp,
                        defect_type,
                    )
                )
                continue

            if current_gap:
                gaps.append(
                    self._create_gap(
                        points=current_gap,
                        expected_index=expected_index,
                        frequency=frequency,
                    )
                )

                current_gap = []

        # Handle a gap touching the end of the timeline.
        if current_gap:
            gaps.append(
                self._create_gap(
                    points=current_gap,
                    expected_index=expected_index,
                    frequency=frequency,
                )
            )

        return gaps

    def _create_gap(
        self,
        points: list[
            tuple[
                pd.Timestamp,
                TimelineDefectType,
            ]
        ],
        expected_index: pd.DatetimeIndex,
        frequency: str,
    ) -> TimelineGap:
        """
        Convert one continuous defective interval into a TimelineGap.
        """

        segments = self._build_segments(
            points
        )

        start = points[0][0]
        end = points[-1][0]

        boundary = self._classify_boundary(
            start=start,
            end=end,
            expected_index=expected_index,
        )

        return TimelineGap(
            segments=tuple(segments),
            frequency=frequency,
            boundary=boundary,
        )

    def _build_segments(
        self,
        points: list[
            tuple[
                pd.Timestamp,
                TimelineDefectType,
            ]
        ],
    ) -> list[TimelineGapSegment]:
        """
        Group consecutive defects of the same type into segments.
        """

        first_timestamp, first_type = points[0]

        segment_start = first_timestamp
        segment_end = first_timestamp
        current_type = first_type

        segments: list[
            TimelineGapSegment
        ] = []

        for timestamp, defect_type in points[1:]:
            if defect_type is current_type:
                segment_end = timestamp
                continue

            segments.append(
                TimelineGapSegment(
                    start=segment_start,
                    end=segment_end,
                    defect_type=current_type,
                )
            )

            segment_start = timestamp
            segment_end = timestamp
            current_type = defect_type

        segments.append(
            TimelineGapSegment(
                start=segment_start,
                end=segment_end,
                defect_type=current_type,
            )
        )

        return segments

    def _classify_boundary(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        expected_index: pd.DatetimeIndex,
    ) -> TimelineBoundaryType:
        """
        Determine whether a gap touches a timeline boundary.
        """

        if start == expected_index[0]:
            return TimelineBoundaryType.START

        if end == expected_index[-1]:
            return TimelineBoundaryType.END

        return TimelineBoundaryType.NONE
