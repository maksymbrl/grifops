from abc import abstractmethod, ABC 
from dataclasses import dataclass 
from pathlib import Path

import pandas as pd
import numpy as np

from sktime.split import temporal_train_test_split, ExpandingWindowSplitter
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanSquaredError,
)
from sktime.forecasting.base import ForecastingHorizon

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
)






def load_data(
    filepath: Path,
    time_col: str,
    data_cols: list[str],
    target: str
) -> TimeSeriesDataset:

    df = pd.read_csv(
        filepath,
        usecols=[time_col, *data_cols],
    )

    df[time_col] = pd.to_datetime(
        df[time_col],
        utc=True,
    )

    df = (
        df
        .set_index(time_col)
        .sort_index()
    )

    return TimeSeriesDataset(
        data=df,
        frequency="1h",
        target=target
    )


def inspect_time_axis(
    dataset: TimeSeriesDataset,
    time_col: str,
    expected_frequency: str = "1h",
) -> dict:
    """
    Inspect the time axis of a time-series dataset.

    Checks:
      - missing/invalid timestamps;
      - chronological ordering;
      - duplicate timestamps;
      - missing timestamps;
      - irregular time intervals.

    Does not modify the dataset.
    """

    df = dataset.data

    timestamps = df.index

    # 1. Check for invalid/missing timestamps
    missing_timestamp_values = timestamps.isna().sum()

    # Ignore invalid timestamps for the remaining checks.
    valid_timestamps = timestamps.dropna()

    # 2. Check chronological ordering
    is_sorted = valid_timestamps.is_monotonic_increasing

    # 3. Check duplicate timestamps
    duplicate_count = valid_timestamps.duplicated().sum()

    # 4. Construct the expected complete time axis
    expected_timestamps = pd.date_range(
        start=valid_timestamps.min(),
        end=valid_timestamps.max(),
        freq=expected_frequency,
    )

    actual_timestamps = pd.DatetimeIndex(
        valid_timestamps
    )

    missing_timestamps = expected_timestamps.difference(
        actual_timestamps
    )

    # 5. Check differences between consecutive timestamps
    sorted_timestamps = valid_timestamps.sort_values()

    time_differences = sorted_timestamps.diff()

    expected_delta = pd.Timedelta(
        expected_frequency
    )

    irregular_intervals = time_differences[
        time_differences.notna()
        & (time_differences != expected_delta)
    ]

    # 6. Build inspection report
    report = {
        "is_sorted": is_sorted,
        "missing_timestamp_values": int(
            missing_timestamp_values
        ),
        "duplicate_count": int(
            duplicate_count
        ),
        "missing_timestamps": missing_timestamps,
        "missing_timestamp_count": len(
            missing_timestamps
        ),
        "irregular_intervals": irregular_intervals,
        "irregular_interval_count": len(
            irregular_intervals
        ),
    }

    return report



def inspect_missing_values(
    dataset: TimeSeriesDataset,
) -> dict:
    """
    Inspect missing values in all data columns.

    For each column:
      - count missing values;
      - calculate missing percentage;
      - identify consecutive missing-value gaps;
      - identify whether each gap is internal or at a boundary.

    Does not modify the dataset.
    """

    df = dataset.data

    report = {}

    for column in df.columns:

        is_missing = df[column].isna()

        missing_count = int(is_missing.sum())

        missing_percentage = (
            missing_count / len(df) * 100
        )

        # No missing values in this column.
        if missing_count == 0:
            report[column] = {
                "missing_count": 0,
                "missing_percentage": 0.0,
                "gaps": pd.DataFrame(),
            }

            continue

        # Give each consecutive block of True/False values
        # its own group number.
        groups = (
            is_missing
            .ne(is_missing.shift())
            .cumsum()
        )

        # Extract only groups containing missing values.
        gaps = (
            df.loc[is_missing]
            .groupby(groups[is_missing])
            .apply(
                lambda group: pd.Series({
                    "start": group.index[0],
                    "end": group.index[-1],
                    "length": len(group),
                }),
                include_groups=False,
            )
            .reset_index(drop=True)
        )

        # Determine whether gaps occur at dataset boundaries.
        gaps["boundary"] = (
            (gaps["start"] == df.index.min())
            | (gaps["end"] == df.index.max())
        )

        report[column] = {
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "gaps": gaps,
        }

    return report


def evaluate_repair_methods(
    dataset: TimeSeriesDataset,
    missing_report: dict,
) -> pd.DataFrame:
    """
    Evaluate candidate missing-value repair methods.

    For every internal missing gap:
      - find known intervals starting at the same weekday/hour;
      - simulate a gap of the same length;
      - reconstruct it using several methods;
      - compare reconstructed values against the known values.

    Boundary gaps are ignored here.
    """

    results = []

    for column, info in missing_report.items():

        series = dataset.data[column]

        for _, gap in info["gaps"].iterrows():

            # Boundary gaps cannot be evaluated/repaired
            # using observations on both sides.
            if gap["boundary"]:
                continue

            gap_start = gap["start"]
            gap_length = int(gap["length"])

            # Find candidate validation intervals that begin
            # at the same weekday and hour as the real gap.
            candidate_starts = series.index[
                (series.index.dayofweek == gap_start.dayofweek)
                & (series.index.hour == gap_start.hour)
            ]

            for validation_start in candidate_starts:

                target_index = pd.date_range(
                    start=validation_start,
                    periods=gap_length,
                    freq="1h",
                )

                previous_week_index = (
                    target_index - pd.Timedelta(days=7)
                )

                next_week_index = (
                    target_index + pd.Timedelta(days=7)
                )

                before_gap = (
                    validation_start
                    - pd.Timedelta(hours=1)
                )

                after_gap = (
                    validation_start
                    + pd.Timedelta(hours=gap_length)
                )

                # Every timestamp needed by every candidate
                # method must exist.
                required_index = (
                    target_index
                    .union(previous_week_index)
                    .union(next_week_index)
                    .union(
                        pd.DatetimeIndex([
                            before_gap,
                            after_gap,
                        ])
                    )
                )

                if not required_index.isin(series.index).all():
                    continue

                required_values = series.loc[
                    required_index
                ]

                # Do not use validation intervals containing
                # genuine missing observations.
                if required_values.isna().any():
                    continue

                actual = series.loc[
                    target_index
                ].to_numpy(dtype=float)

                previous_week = series.loc[
                    previous_week_index
                ].to_numpy(dtype=float)

                next_week = series.loc[
                    next_week_index
                ].to_numpy(dtype=float)

                weekly_average = (
                    previous_week + next_week
                ) / 2

                linear = np.linspace(
                    series.loc[before_gap],
                    series.loc[after_gap],
                    gap_length + 2,
                )[1:-1]

                predictions = {
                    "linear": linear,
                    "previous_week": previous_week,
                    "next_week": next_week,
                    "weekly_average": weekly_average,
                }

                for method, predicted in predictions.items():

                    errors = actual - predicted

                    mae = np.mean(
                        np.abs(errors)
                    )

                    rmse = np.sqrt(
                        np.mean(errors ** 2)
                    )

                    results.append({
                        "column": column,
                        "gap_start": gap_start,
                        "gap_length": gap_length,
                        "validation_start": validation_start,
                        "method": method,
                        "mae": float(mae),
                        "rmse": float(rmse),
                    })

    return pd.DataFrame(results)


def repair_missing_values(
    dataset: TimeSeriesDataset,
    missing_report: dict,
    repair_results: pd.DataFrame,
    selection_metric: str = "mae",
) -> TimeSeriesDataset:
    """
    Repair missing values.

    - Boundary gaps are dropped.
    - Internal gaps use the repair method with the best
      average validation score.
    """

    df = dataset.data.copy()

    for column, info in missing_report.items():

        for _, gap in info["gaps"].iterrows():

            gap_start = gap["start"]
            gap_length = int(gap["length"])

            target_index = pd.date_range(
                start=gap_start,
                periods=gap_length,
                freq="1h",
            )

            # Boundary gaps: remove them.
            if gap["boundary"]:
                df = df.drop(
                    index=target_index,
                    errors="ignore",
                )
                continue

            # Get validation results for this specific gap.
            gap_results = repair_results[
                (repair_results["column"] == column)
                & (repair_results["gap_start"] == gap_start)
                & (repair_results["gap_length"] == gap_length)
            ]

            # Average score across all simulated validation gaps.
            method_scores = (
                gap_results
                .groupby("method")[selection_metric]
                .mean()
            )

            best_method = method_scores.idxmin()

            print(
                f"Repairing {column}: "
                f"{gap_start}, "
                f"{gap_length} hours -> "
                f"{best_method}"
            )

            previous_week_index = (
                target_index - pd.Timedelta(days=7)
            )

            next_week_index = (
                target_index + pd.Timedelta(days=7)
            )

            if best_method == "previous_week":

                repaired_values = (
                    df.loc[
                        previous_week_index,
                        column,
                    ]
                    .to_numpy()
                )

            elif best_method == "next_week":

                repaired_values = (
                    df.loc[
                        next_week_index,
                        column,
                    ]
                    .to_numpy()
                )

            elif best_method == "weekly_average":

                previous_week = (
                    df.loc[
                        previous_week_index,
                        column,
                    ]
                    .to_numpy()
                )

                next_week = (
                    df.loc[
                        next_week_index,
                        column,
                    ]
                    .to_numpy()
                )

                repaired_values = (
                    previous_week + next_week
                ) / 2

            elif best_method == "linear":

                before_gap = (
                    gap_start - pd.Timedelta(hours=1)
                )

                after_gap = (
                    gap_start
                    + pd.Timedelta(hours=gap_length)
                )

                repaired_values = np.linspace(
                    df.loc[before_gap, column],
                    df.loc[after_gap, column],
                    gap_length + 2,
                )[1:-1]

            else:
                raise ValueError(
                    f"Unknown repair method: {best_method}"
                )

            df.loc[
                target_index,
                column,
            ] = repaired_values

    dataset.data = df

    return dataset


def validate_clean_data(
    dataset: TimeSeriesDataset,
    expected_frequency: str = "1h",
) -> dict:
    """
    Validate the cleaned time-series dataset.

    Checks:
      - chronological ordering;
      - duplicate timestamps;
      - missing timestamps;
      - irregular intervals;
      - missing values in data columns.

    Raises ValueError if validation fails.
    """

    df = dataset.data
    timestamps = df.index

    if not isinstance(timestamps, pd.DatetimeIndex):
        raise TypeError(
            "TimeSeriesDataset must use a DatetimeIndex."
        )

    # 1. Chronological ordering
    is_sorted = timestamps.is_monotonic_increasing

    # 2. Duplicate timestamps
    duplicate_count = int(
        timestamps.duplicated().sum()
    )

    # 3. Missing timestamps
    expected_timestamps = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq=expected_frequency,
        tz=timestamps.tz,
    )

    missing_timestamps = (
        expected_timestamps.difference(
            timestamps
        )
    )

    # 4. Irregular intervals
    time_differences = (
        timestamps.to_series().diff()
    )

    expected_delta = pd.Timedelta(
        expected_frequency
    )

    irregular_intervals = time_differences[
        time_differences.notna()
        & (time_differences != expected_delta)
    ]

    # 5. Missing values in data columns
    missing_values = (
        df.isna()
        .sum()
        .astype(int)
        .to_dict()
    )

    report = {
        "is_sorted": is_sorted,
        "duplicate_count": duplicate_count,
        "missing_timestamp_count": len(
            missing_timestamps
        ),
        "irregular_interval_count": len(
            irregular_intervals
        ),
        "missing_values": missing_values,
    }

    is_valid = (
        is_sorted
        and duplicate_count == 0
        and len(missing_timestamps) == 0
        and len(irregular_intervals) == 0
        and all(
            count == 0
            for count in missing_values.values()
        )
    )

    report["is_valid"] = is_valid

    if not is_valid:
        raise ValueError(
            f"Clean dataset validation failed: {report}"
        )

    return report




def create_candidate_models() -> dict:
    """
    Create the initial set of forecasting models.

    Returns simple baseline models that later models
    must outperform.
    """

    models = {
        "naive": NaiveForecaster(
            strategy="last",
        ),

        "seasonal_naive_24h": NaiveForecaster(
            strategy="last",
            sp=24,
        ),

        "seasonal_naive_168h": NaiveForecaster(
            strategy="last",
            sp=168,
        ),
    }

    return models




def evaluate_models(
    models: dict,
    y_train: pd.Series,
    horizon: int = 24,
    backtest_days: int = 365,
) -> pd.DataFrame:

    fh = np.arange(1, horizon + 1)

    backtest_size = backtest_days * 24
    initial_window = len(y_train) - backtest_size

    # kfold cross validation where k = 365 days
    cv = ExpandingWindowSplitter(
        initial_window=initial_window,
        step_length=24,
        fh=fh,
    )

    metrics = [
        MeanAbsoluteError(),
        MeanSquaredError(
            square_root=True,
        ),
    ]

    results = []

    for name, model in models.items():

        model_results = evaluate(
            forecaster=model,
            y=y_train,
            cv=cv,
            scoring=metrics,
            error_score="raise",
        )

        results.append({
            "model": name,
            "mae": model_results[
                "test_MeanAbsoluteError"
            ].mean(),
            "rmse": model_results[
                "test_MeanSquaredError"
            ].mean(),
            "origins": len(model_results),
        })

    return pd.DataFrame(results)


def select_best_model(
    models: dict,
    results: pd.DataFrame,
    metric: str = "mae",
):
    """
    Select the candidate model with the lowest
    evaluation metric.
    """

    best_row = results.loc[
        results[metric].idxmin()
    ]

    best_model_name = best_row["model"]

    return (
        best_model_name,
        models[best_model_name],
    )



def fit_model(
    model,
    y_train: pd.Series,
):
    """
    Fit the selected forecasting model
    on the complete training dataset.
    """

    model.fit(y_train)

    return model

def predict_test(
    model,
    y_test: pd.Series,
) -> pd.Series:
    """
    Generate predictions for the untouched test period.
    """

    fh = ForecastingHorizon(
        y_test.index,
        is_relative=False,
    )

    return model.predict(fh=fh)



def evaluate_test(
    model,
    y_train: pd.Series,
    y_test: pd.Series,
    horizon: int = 24,
) -> pd.DataFrame:
    """
    Evaluate the selected model on the untouched test period
    using rolling 24-hour forecasts.
    """

    y = pd.concat([
        y_train,
        y_test,
    ])

    fh = np.arange(
        1,
        horizon + 1,
    )

    cv = ExpandingWindowSplitter(
        initial_window=len(y_train),
        step_length=horizon,
        fh=fh,
    )

    metrics = [
        MeanAbsoluteError(),
        MeanSquaredError(
            square_root=True,
        ),
    ]

    results = evaluate(
        forecaster=model,
        y=y,
        cv=cv,
        scoring=metrics,
        error_score="raise",
    )

    return results


def generate_forecast(
    model,
    y: pd.Series,
    horizon: int = 24,
) -> pd.Series:
    """
    Fit the selected model on all available observations
    and forecast future values.
    """

    model.fit(y)

    fh = np.arange(
        1,
        horizon + 1,
    )

    forecast = model.predict(
        fh=fh,
    )

    return forecast


def main(): 
    print("Programs Started")

    # 1. Data Loading
    filepath = Path("./data/raw/time_series_60min_singleindex.csv")
    columns = ["utc_timestamp", "NO_load_actual_entsoe_transparency"]

    cols = { "time_col": "utc_timestamp", "data_cols": ["NO_load_actual_entsoe_transparency"] }

    df = load_data(
        filepath, 
        time_col = cols["time_col"], 
        data_cols = cols["data_cols"],
        target=cols["data_cols"][0],
    )

    # What do the first few observations actually look like?
    print(df.data.head())

    # print(f"Number of timestamps: {df.shape[0]}, and number of variables: {df.shape[1]}")

    # df = check_data(df, time_col = cols["time_col"])

    # 2. Data Inspection / Validation

    time_report = inspect_time_axis(
        df,
        time_col=cols["time_col"],
    )

    print(time_report)

    print(" --------- ")

    missing_report = inspect_missing_values(df)

    for column, info in missing_report.items():

        print(f"\nColumn: {column}")
        print("Missing values:", info["missing_count"])

        print(
            "Missing percentage:",
            f"{info['missing_percentage']:.4f}%",
        )

        print("Missing gaps:")
        print(info["gaps"])


    print(" --------- ")

    repair_results = evaluate_repair_methods(
        df,
        missing_report,
    )

    print(
        repair_results
        .groupby(
            ["gap_start", "gap_length", "method"]
        )[["mae", "rmse"]]
        .mean()
        .round(2)
    )

    df = repair_missing_values(
        df,
        missing_report,
        repair_results,
    )

    # # 3. Data Cleaning / Preprocessing

    clean_report = validate_clean_data(df)

    print(clean_report)

    # 4. Post-cleaning Validation / Visualization

    # 5. Forecasting Strategy / Model Candidate Selection

    target = df.data[df.target].copy()
    
    # sktime internally uses PeriodIndex in several forecasting routines.
    # PeriodIndex does not preserve timezone information.
    #
    # Since our timestamps are already UTC, dropping the timezone here
    # does not change the chronological meaning of the data.
    target.index = (
        target.index
        .tz_convert("UTC")
        .tz_localize(None)
    )
    
    # Explicitly establish the hourly frequency.
    target = target.asfreq(df.frequency)

    
    y_train, y_test = temporal_train_test_split(
        target,
        test_size=24 * 365,
    )

    print("Training set:")
    print(y_train.index.min())
    print(y_train.index.max())
    print("Observations:", len(y_train))

    print()

    print("Test set:")
    print(y_test.index.min())
    print(y_test.index.max())
    print("Observations:", len(y_test))


    print(y_train.index)
    print("Frequency:", y_train.index.freq)
    print("Timezone:", y_train.index.tz)

    models = create_candidate_models()

    for name, model in models.items():
        print(name, model)

    # 6. Model Training + Backtesting

    results = evaluate_models(
        models,
        y_train,
        horizon=24,
        backtest_days=365,
    )
    
    print(
        results
        .sort_values("mae")
        .round(2)
    )


    # 7. Model Evaluation + Final Model Selection

    best_model_name, best_model = select_best_model(
        models,
        results,
    )

    print("Best model:", best_model_name)

    best_model = fit_model(
        best_model,
        y_train,
    )

    print("Model fitted:", best_model.is_fitted)

    # 8. Final Forecast Generation

    y_pred = predict_test(
        best_model,
        y_test,
    )

    print(y_pred.head())

    # 9. Results Reporting

    test_results = evaluate_test(
        best_model,
        y_train,
        y_test,
        horizon=24,
    )

    print(
        "Test MAE:",
        test_results[
            "test_MeanAbsoluteError"
        ].mean(),
    )

    print(
        "Test RMSE:",
        test_results[
            "test_MeanSquaredError"
        ].mean(),
    )

    print(
        "Forecast origins:",
        len(test_results),
    )

    # 10. Forecast the entire thing 

    final_model = best_model.clone()

    forecast = generate_forecast(
        final_model,
        target,
        horizon=24,
    )

    print("Future forecast:")
    print(forecast)

    print("Program Ended")


if __name__ == '__main__':
    main()
