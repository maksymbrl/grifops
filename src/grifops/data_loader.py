# src/grifops/data_loader.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import pandas as pd

from grifops.data_adapter import TimeSeriesDataAdapter
from grifops.dataset import TimeSeriesDataset


SourceT = TypeVar("SourceT")


class TimeSeriesDataLoader(
    ABC,
    Generic[SourceT],
):
    """
    Loads time-series data from an external source.
    """

    @abstractmethod
    def load(
        self,
        source: SourceT,
    ) -> TimeSeriesDataset:
        """
        Load an external time-series source.
        """

        ...


class CsvTimeSeriesLoader(
    TimeSeriesDataLoader[str | Path]
):
    """
    Loads a CSV file and converts it into a TimeSeriesDataset
    through a supplied DataFrame adapter.
    """

    def __init__(
        self,
        adapter: TimeSeriesDataAdapter[pd.DataFrame],
        **read_csv_kwargs: Any,
    ) -> None:

        self._adapter = adapter
        self._read_csv_kwargs = read_csv_kwargs

    def load(
        self,
        source: str | Path,
    ) -> TimeSeriesDataset:

        path = Path(source)

        if not path.is_file():
            raise FileNotFoundError(
                f"Time-series data file does not exist: {path}"
            )

        data = pd.read_csv(
            path,
            **self._read_csv_kwargs,
        )

        return self._adapter.adapt(data)
