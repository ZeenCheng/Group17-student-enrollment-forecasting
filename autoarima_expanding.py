# autoarima_expanding.py
"""
Expanding-window AutoARIMA forecasting helper.

返回：
- metrics: {"MAE": ..., "RMSE": ..., "MAPE (%)": ...}
- forecast_df: 列 ["ds", "autoarima_expanding", "true"]
- meta: [(w_start, pred_mean (np.array[pred_len]), pred_std (np.array[pred_len]), true_vals (np.array[pred_len]))]
  用于在前端绘图（均值曲线 + 阴影）
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from pmdarima import auto_arima
except Exception as e:
    raise ImportError("pmdarima is required. Install with `pip install pmdarima`") from e


def run_autoarima_expanding_forecast(
    df: pd.DataFrame,
    series_col=None,
    prediction_length: int = 20,
    initial_train: int = 50,
    step: int = 20,
    seasonal: bool = False,
    stepwise: bool = True,
    suppress_warnings: bool = True,
    error_action: str = "ignore",
    alpha: float = 0.05,  # 置信区间 (1 - alpha)，默认 95%
):
    # 选择列
    if series_col is None:
        series = df.iloc[:, 0].copy()
        colname = df.columns[0]
    else:
        series = df[series_col].copy()
        colname = series_col

    # 时间索引
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    n = len(series)
    if initial_train >= n:
        raise ValueError("initial_train must be smaller than the length of series.")
    if prediction_length <= 0:
        raise ValueError("prediction_length must be > 0")

    preds_all = []
    trues_all = []
    ds_all = []
    meta = []

    # 正态近似下，95%CI ≈ mean ± 1.96*std
    # 通用 z 值：z = 1.96 对应 alpha=0.05
    z = 1.96 if abs(alpha - 0.05) < 1e-6 else 1.96  # 简化：统一按 1.96 处理

    # 扩窗循环
    for w_start in range(initial_train, n - prediction_length + 1, step):
        train_slice = series.iloc[:w_start]
        true_slice = series.iloc[w_start: w_start + prediction_length]

        # 拟合 AutoARIMA
        model = auto_arima(
            train_slice,
            seasonal=seasonal,
            stepwise=stepwise,
            suppress_warnings=suppress_warnings,
            error_action=error_action,
        )

        # 预测 + 置信区间
        out = model.predict(n_periods=prediction_length, return_conf_int=True, alpha=alpha)
        if isinstance(out, tuple):
            # 新版 pmdarima：返回 (y_pred, conf_int)
            y_pred, conf_int = out
        else:
            # 保险处理（几乎不会走到）
            y_pred = out
            conf_int = None

        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        if conf_int is not None:
            # conf_int: shape [pred_len, 2] -> [lower, upper]
            lower = conf_int[:, 0].astype(float)
            upper = conf_int[:, 1].astype(float)
            # std 近似： (upper - lower) / (2*z)
            pred_std = (upper - lower) / (2.0 * z)
        else:
            # 没有 CI 时，给一个很小的 std，保证可以画阴影
            pred_std = np.full_like(y_pred, 1e-8, dtype=float)

        true_vals = true_slice.values.astype(float)

        preds_all.extend(y_pred.tolist())
        trues_all.extend(true_vals.tolist())
        ds_all.extend(true_slice.index.tolist())

        meta.append((w_start, y_pred, pred_std, true_vals))

    preds_all = np.asarray(preds_all, dtype=float)
    trues_all = np.asarray(trues_all, dtype=float)

    # 指标
    mae = float(mean_absolute_error(trues_all, preds_all))
    rmse = float(np.sqrt(mean_squared_error(trues_all, preds_all)))
    mape = float(np.mean(np.abs((trues_all - preds_all) / (trues_all + 1e-8))) * 100.0)
    metrics = {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

    forecast_df = pd.DataFrame(
        {
            "ds": pd.to_datetime(ds_all),
            "autoarima_expanding": preds_all,
            "true": trues_all,
        }
    )

    return metrics, forecast_df, meta


if __name__ == "__main__":
    # quick CLI test:
    #   python autoarima_expanding.py ts_wide.csv --pred_len 12 --init_train 50 --step 12
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="ts_wide.csv")
    parser.add_argument("--pred_len", type=int, default=20)
    parser.add_argument("--init_train", type=int, default=50)
    parser.add_argument("--step", type=int, default=20)
    args = parser.parse_args()

    df_cli = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    metrics_cli, forecast_df_cli, meta_cli = run_autoarima_expanding_forecast(
        df_cli,
        prediction_length=args.pred_len,
        initial_train=args.init_train,
        step=args.step,
    )
    print("Metrics:", metrics_cli)
    print(forecast_df_cli.head())

