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
        DatetimeIndex, 第一列为目标序列 (或用 series_col 指定列)。
    series_col : str|int|None
        预测的列名/列序号。None 则用第一列。
    prediction_length : int
        每个窗口的预测步长。
    initial_train : int
        第一个窗口的训练长度。
    step : int
        窗口步长（通常等于 prediction_length）。
    model_name : str
        Chronos 预训练模型名称。
    device : str
        "cuda" 或 "cpu"，None 自动检测。

    Returns
    -------
    metrics : dict
        {"MAE": ..., "RMSE": ..., "MAPE (%)": ...}
    forecast_df : DataFrame
        列为 ["ds", "chronos_expanding", "true"]，拼接所有窗口的预测与真实。
    meta : list of tuples
        [(window_start_index, pred_array, true_array), ...] 用于绘图/调试。
    """
    # 选列
    if series_col is None:
        series = df.iloc[:, 0].copy()
        colname = df.columns[0]
    else:
        series = df[series_col].copy()
        colname = series_col

    # 确保是 DatetimeIndex
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    n = len(series)
    if initial_train >= n:
        raise ValueError("initial_train must be smaller than the length of series.")

    # 设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 Chronos
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )

    preds_all = []
    trues_all = []
    ds_all = []      # 收集每个窗口真实值对应的时间索引
    meta = []

    # 扩窗循环
    for w_start in range(initial_train, n - prediction_length + 1, step):
        train_slice = series.iloc[:w_start]
        true_slice = series.iloc[w_start: w_start + prediction_length]

        context = torch.tensor(train_slice.values, dtype=torch.float32)

        # 预测
        forecast = pipeline.predict(context, prediction_length=prediction_length)  # [1, num_samples, pred_len]
        forecast_array = forecast[0].numpy()  # shape [num_samples, pred_len]
        pred_mean = np.mean(forecast_array, axis=0)   # 连续曲线
        pred_std  = np.std(forecast_array, axis=0)    # 可用于阴影表示不确定性


        preds_all.extend(pred_mean.tolist())
        trues_all.extend(true_slice.values.tolist())
        ds_all.extend(true_slice.index.tolist())
        meta.append((w_start, forecast_array, true_slice.values))

    preds_all = np.asarray(preds_all, dtype=float)
    trues_all = np.asarray(trues_all, dtype=float)

    # 误差指标（兼容旧版 sklearn：不用 squared 参数）
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

