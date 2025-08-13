# autoarima_forecast.py
"""
Independent AutoARIMA forecasting helper.

Provides run_autoarima_forecast(df, ...) that:
 - takes a wide-format dataframe (DatetimeIndex, columns = series)
 - picks the first series (or a named column if provided)
 - runs rolling-window AutoARIMA forecasts (same windowing as your Moirai test setup)
 - returns (forecast_df, fig) where forecast_df has columns ['ds', 'autoarima']
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from pmdarima import auto_arima
except Exception as e:
    raise ImportError("pmdarima is required. Install with `pip install pmdarima`") from e


def run_autoarima_forecast(
    df,
    series_col=None,
    prediction_length=20,
    test_len=100,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
):
    """
    df: pandas DataFrame with DatetimeIndex (wide format). Example: ts_wide.csv
    series_col: column name or index to use. If None, use first column.
    prediction_length: horizon for each window (PDT)
    test_len: total length of test period (TEST)
    Returns: (forecast_df, fig)
      - forecast_df: DataFrame with ['ds', 'autoarima'] for the test period
      - fig: matplotlib Figure with actual vs forecast plot
    """
    # pick series
    if series_col is None:
        series = df.iloc[:, 0].copy()
        colname = df.columns[0]
    else:
        series = df[series_col].copy()
        colname = series_col

    # ensure datetime index
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    n = len(series)
    if not (0 < test_len < n):
        raise ValueError("test_len must be >0 and < len(series)")

    train_len = n - test_len

    preds = []
    # rolling window: iterate over test period by blocks of prediction_length
    for start in range(0, test_len, prediction_length):
        train_end = train_len + start  # exclusive index for training
        train_slice = series.iloc[:train_end]

        # fit AutoARIMA on current training slice
        model = auto_arima(
            train_slice,
            seasonal=seasonal,
            stepwise=stepwise,
            suppress_warnings=suppress_warnings,
            error_action=error_action,
        )

        n_periods = min(prediction_length, test_len - start)
        out = model.predict(n_periods=n_periods)
        preds.extend(out)

    # make a forecast series aligned to the last test_len timestamps
    forecast_vals = np.array(preds[:test_len])
    forecast_index = series.index[-test_len:]
    forecast_series = pd.Series(forecast_vals, index=forecast_index)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series.index, series.values, label="Actual", color="tab:blue")
    ax.plot(forecast_series.index, forecast_series.values, label="AutoARIMA Forecast", color="tab:orange")
    ax.axvline(x=series.index[train_len], color="gray", linestyle="--", label="Train/Test split")
    ax.set_title(f"AutoARIMA forecast - {colname}")
    ax.legend()
    fig.tight_layout()

    forecast_df = pd.DataFrame({"ds": forecast_series.index, "autoarima": forecast_series.values})
    return forecast_df, fig


if __name__ == "__main__":
    # quick CLI test: python autoarima_forecast.py ts_wide.csv
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "ts_wide.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    forecast_df, fig = run_autoarima_forecast(df, prediction_length=20, test_len=100)
    print(forecast_df.tail())
    fig.show()
