# Chronos_expanding.py
"""
Expanding-window Chronos forecasting helper.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from chronos import ChronosPipeline


def run_chronos_expanding_forecast(
    df: pd.DataFrame,
    series_col=None,
    prediction_length: int = 20,
    initial_train: int = 50,
    step: int = 20,
    model_name: str = "amazon/chronos-t5-small",
    device: str = None,
):
    """
    Expanding-window evaluation for Chronos model.

    Parameters
    ----------
    df : DataFrame
        DatetimeIndex, first column as the target series.
    series_col : str|int|None
        The column name/index to forecast. If None, use the first column.
    prediction_length : int
        Forecast horizon for each window.
    initial_train : int
        Training size of the first window.
    step : int
        Window step size (usually equal to prediction_length).
    model_name : str
        Chronos pre-trained model name.
    device : str
        "cuda" or "cpu". If None, auto-detect.

    Returns
    -------
    metrics : dict
        {"MAE": ..., "RMSE": ..., "MAPE (%)": ...}
    forecast_df : DataFrame
        Columns ["ds", "chronos_expanding", "true"], concatenating predictions and truths.
    meta : list of tuples
        [(window_start_index, pred_array, true_array), ...] for plotting/debugging.
    """
    # Select column
    if series_col is None:
        series = df.iloc[:, 0].copy()
        colname = df.columns[0]
    else:
        series = df[series_col].copy()
        colname = series_col

    # Ensure DatetimeIndex
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    n = len(series)
    if initial_train >= n:
        raise ValueError("initial_train must be smaller than the length of series.")

    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Chronos
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )

    preds_all = []
    trues_all = []
    ds_all = []      # Collect datetime indices of true values from each window
    meta = []

    # Expanding-window loop
    for w_start in range(initial_train, n - prediction_length + 1, step):
        train_slice = series.iloc[:w_start]
        true_slice = series.iloc[w_start: w_start + prediction_length]

        context = torch.tensor(train_slice.values, dtype=torch.float32)

        # Forecast
        forecast = pipeline.predict(context, prediction_length=prediction_length)  # [1, num_samples, pred_len]
        forecast_array = forecast[0].numpy()  # shape [num_samples, pred_len]
        pred_mean = np.mean(forecast_array, axis=0)   # Mean forecast curve
        pred_std  = np.std(forecast_array, axis=0)    # Uncertainty band (std dev)

        preds_all.extend(pred_mean.tolist())
        trues_all.extend(true_slice.values.tolist())
        ds_all.extend(true_slice.index.tolist())
        meta.append((w_start, forecast_array, true_slice.values))

    preds_all = np.asarray(preds_all, dtype=float)
    trues_all = np.asarray(trues_all, dtype=float)

    # Error metrics
    mae = float(mean_absolute_error(trues_all, preds_all))
    rmse = float(np.sqrt(mean_squared_error(trues_all, preds_all)))
    mape = float(np.mean(np.abs((trues_all - preds_all) / (trues_all + 1e-8))) * 100.0)

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

    forecast_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(ds_all),
            "chronos_expanding": preds_all,
            "true": trues_all,
        }
    )

    return metrics, forecast_df, meta


if __name__ == "__main__":
    # quick CLI test:
    #   python Chronos_expanding.py ts_wide.csv --model amazon/chronos-t5-tiny --pred_len 12 --init_train 50 --step 12
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="ts_wide.csv")
    parser.add_argument("--model", default="amazon/chronos-t5-small")
    parser.add_argument("--pred_len", type=int, default=20)
    parser.add_argument("--init_train", type=int, default=50)
    parser.add_argument("--step", type=int, default=20)
    args = parser.parse_args()

    df_cli = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    metrics_cli, forecast_df_cli, meta_cli = run_chronos_expanding_forecast(
        df_cli,
        prediction_length=args.pred_len,
        initial_train=args.init_train,
        step=args.step,
        model_name=args.model,
        device=None,
    )
    print("Metrics:", metrics_cli)
    print(forecast_df_cli.head())


