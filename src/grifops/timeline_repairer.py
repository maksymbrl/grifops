# src/grifops/timeline_repairer.py

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class TimelineDefectType(Enum):
    """
    Type of defect affecting an expected timeline observation.

    Describes why an observation is considered defective.
    """

    MISSING_TIMESTAMP = "missing_timestamp"
    MISSING_VALUE = "missing_value"


class TimelineBoundaryType(Enum):
    """
    Position of a timeline gap relative to the dataset boundaries.
    """

    NONE = "none"
    START = "start"
    END = "end"


@dataclass(frozen=True)
class TimelineGapSegment:
    """
    A contiguous portion of a timeline gap affected by one
    specific defect type.

    Represents one homogeneous defective interval.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    defect_type: TimelineDefectType

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                "TimelineGapSegment end cannot precede start."
            )

    def index(
        self,
        frequency: str,
    ) -> pd.DatetimeIndex:
        """
        Return all expected timestamps covered by this segment.
        """

        return pd.date_range(
            start=self.start,
            end=self.end,
            freq=frequency,
        )

    def length(
        self,
        frequency: str,
    ) -> int:
        """
        Return the number of expected observations in this segment.
        """

        return len(self.index(frequency))


@dataclass(frozen=True)
class TimelineGap:
    """
    A continuous interval containing no valid observations.

    A gap consists of one or more contiguous TimelineGapSegments.
    Different segments may represent different defect types.

    The gap also records whether it touches the start or end of
    the inspected timeline.
    """

    segments: tuple[TimelineGapSegment, ...]
    frequency: str
    boundary: TimelineBoundaryType = TimelineBoundaryType.NONE

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError(
                "TimelineGap must contain at least one segment."
            )

        offset = pd.tseries.frequencies.to_offset(
            self.frequency
        )

        for segment in self.segments:
            segment_index = segment.index(
                self.frequency
            )

            if segment_index[-1] != segment.end:
                raise ValueError(
                    "TimelineGapSegment boundaries must align "
                    "with the timeline frequency."
                )

        for previous, current in zip(
            self.segments,
            self.segments[1:],
        ):
            expected_start = previous.end + offset

            if current.start != expected_start:
                raise ValueError(
                    "TimelineGap segments must be contiguous."
                )

            if current.defect_type is previous.defect_type:
                raise ValueError(
                    "Adjacent TimelineGap segments must have "
                    "different defect types."
                )

    @property
    def start(self) -> pd.Timestamp:
        """
        Return the first defective timestamp in the gap.
        """

        return self.segments[0].start

    @property
    def end(self) -> pd.Timestamp:
        """
        Return the last defective timestamp in the gap.
        """

        return self.segments[-1].end

    @property
    def index(self) -> pd.DatetimeIndex:
        """
        Return all expected timestamps covered by the gap.
        """

        return pd.date_range(
            start=self.start,
            end=self.end,
            freq=self.frequency,
        )

    @property
    def length(self) -> int:
        """
        Return the number of expected observations in the gap.
        """

        return len(self.index)

    @property
    def is_mixed(self) -> bool:
        """
        Return True if the gap contains multiple defect types.
        """

        return len(
            {
                segment.defect_type
                for segment in self.segments
            }
        ) > 1
