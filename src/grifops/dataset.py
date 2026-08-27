# src/grifops/dataset.py

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSeriesDataset:
    """
    Canonical internal representation of a time-series dataset.

    The dataset preserves the observed timeline as supplied by the
    data source. Timeline defects are detected later by inspectors.
    """

    data: pd.DataFrame
    target: str
    frequency: str
    expected_start: pd.Timestamp | None = None
    expected_end: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError(
                "TimeSeriesDataset requires a pandas DatetimeIndex."
            )

        if self.target not in self.data.columns:
            raise ValueError(
                f"Target column '{self.target}' does not exist."
            )

        try:
            pd.tseries.frequencies.to_offset(
                self.frequency
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid timeline frequency: '{self.frequency}'."
            ) from exc

        if (
            self.expected_start is not None
            and self.expected_end is not None
            and self.expected_end < self.expected_start
        ):
            raise ValueError(
                "Expected timeline end cannot precede start."
            )

    @property
    def target_series(self) -> pd.Series:
        """
        Return the forecasting target series.
        """

        return self.data[self.target]
