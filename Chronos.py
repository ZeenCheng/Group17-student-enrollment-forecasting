"""
Independent Chronos forecasting helper.

Provides run_chronos_forecast(df, ...) that:
 - takes a wide-format dataframe (DatetimeIndex, columns = series)
 - picks the first series (or a named column if provided)
 - runs simple rolling-window Chronos forecasts
 - returns (forecast_df, fig) where forecast_df has columns ['ds', 'chronos']
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from chronos import ChronosPipeline


def run_chronos_forecast(
    df,
    series_col=None,
    prediction_length=20,
    test_len=100,
    model_name="amazon/chronos-t5-tiny",
    device=None,
):
    """
    df: pandas DataFrame with DatetimeIndex (wide format).
    series_col: column name or index to use. If None, use first column.
    prediction_length: horizon for each block forecast
    test_len: total length of test period (TEST)
    model_name: pretrained Chronos model (tiny/small/large)
    device: 'cuda' or 'cpu'. If None, auto-detect.
    Returns: (forecast_df, fig)
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

    # detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # load Chronos
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )

    preds = []
    # rolling forecast by prediction_length
    for start in range(0, test_len, prediction_length):
        train_end = train_len + start
        train_slice = series.iloc[:train_end]

        context = torch.tensor(train_slice.values, dtype=torch.float32)

        n_periods = min(prediction_length, test_len - start)
        forecast = pipeline.predict(context, prediction_length=n_periods)  # [num_series, num_samples, pred_len]

        # take median forecast
        median_forecast = np.median(forecast[0].numpy(), axis=0)
        preds.extend(median_forecast)

    # align forecast to test period
    forecast_vals = np.array(preds[:test_len])
    forecast_index = series.index[-test_len:]
    forecast_series = pd.Series(forecast_vals, index=forecast_index)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series.index, series.values, label="Actual", color="tab:blue")
    ax.plot(forecast_series.index, forecast_series.values, label="Chronos Forecast", color="tab:orange")
    ax.axvline(x=series.index[train_len], color="gray", linestyle="--", label="Train/Test split")
    ax.set_title(f"Chronos forecast - {colname}")
    ax.legend()
    fig.tight_layout()

    forecast_df = pd.DataFrame({"ds": forecast_series.index, "chronos": forecast_series.values})
    return forecast_df, fig


if __name__ == "__main__":
    # quick CLI test: python chronos_forecast.py ts_wide.csv
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "ts_wide.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    forecast_df, fig = run_chronos_forecast(df, prediction_length=20, test_len=100)
    print(forecast_df.tail())
    fig.show()
