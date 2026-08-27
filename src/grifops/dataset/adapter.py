# src/grifops/data_adapter.py

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import pandas as pd

from grifops.dataset import TimeSeriesDataset


DataT = TypeVar("DataT")


class TimeSeriesDataAdapter(
    ABC,
    Generic[DataT],
):
    """
    Converts an external data representation into the canonical
    GRIFOps TimeSeriesDataset representation.
    """

    @abstractmethod
    def adapt(
        self,
        data: DataT,
    ) -> TimeSeriesDataset:
        """
        Convert external data into a TimeSeriesDataset.
        """

        ...


class PandasDataFrameAdapter(
    TimeSeriesDataAdapter[pd.DataFrame]
):
    """
    Converts a pandas DataFrame into a TimeSeriesDataset.

    Parses timestamps and establishes the DatetimeIndex without
    repairing, sorting, or otherwise modifying timeline defects.
    """

    def __init__(
        self,
        timestamp_column: str,
        target_column: str,
        frequency: str,
        *,
        normalize_to_utc: bool = False,
        expected_start: str | pd.Timestamp | None = None,
        expected_end: str | pd.Timestamp | None = None,
    ) -> None:
        self._timestamp_column = timestamp_column
        self._target_column = target_column
        self._frequency = frequency
        self._normalize_to_utc = normalize_to_utc

        self._expected_start = self._parse_boundary(
            expected_start
        )

        self._expected_end = self._parse_boundary(
            expected_end
        )

    def adapt(
        self,
        data: pd.DataFrame,
    ) -> TimeSeriesDataset:
        """
        Convert a DataFrame into GRIFOps' canonical representation.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "PandasDataFrameAdapter requires a pandas DataFrame."
            )

        self._validate_columns(data)

        df = data.copy()

        df[self._timestamp_column] = self._parse_timestamps(
            df[self._timestamp_column]
        )

        df = df.set_index(
            self._timestamp_column
        )

        return TimeSeriesDataset(
            data=df,
            target=self._target_column,
            frequency=self._frequency,
            expected_start=self._expected_start,
            expected_end=self._expected_end,
        )

    def _validate_columns(
        self,
        data: pd.DataFrame,
    ) -> None:
        required_columns = {
            self._timestamp_column,
            self._target_column,
        }

        missing_columns = (
            required_columns
            - set(data.columns)
        )

        if missing_columns:
            raise ValueError(
                "Required columns are missing: "
                f"{sorted(missing_columns)}"
            )

    def _parse_timestamps(
        self,
        timestamps: pd.Series,
    ) -> pd.DatetimeIndex:
        try:
            parsed = pd.to_datetime(
                timestamps,
                errors="raise",
                utc=self._normalize_to_utc,
            )

        except (ValueError, TypeError) as exc:
            raise ValueError(
                "Timestamp column contains values that "
                "cannot be parsed."
            ) from exc

        if parsed.isna().any():
            raise ValueError(
                "Timestamp column contains missing timestamps "
                "whose temporal position cannot be determined."
            )

        return pd.DatetimeIndex(parsed)

    def _parse_boundary(
        self,
        value: str | pd.Timestamp | None,
    ) -> pd.Timestamp | None:
        if value is None:
            return None

        timestamp = pd.Timestamp(value)

        if self._normalize_to_utc:
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(
                    "UTC"
                )
            else:
                timestamp = timestamp.tz_convert(
                    "UTC"
                )

        return timestamp
