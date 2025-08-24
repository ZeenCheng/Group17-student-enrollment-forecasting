# sample_moirai.py
"""
Expanding-window Moirai forecasting helper.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule


def run_moirai_expanding_forecast(
    df: pd.DataFrame,
    model_choice: str = "moirai",
    size_choice: str = "small",
    prediction_length: int = 20,
    initial_train: int = 100,
    max_context_length: int = 200,
    batch_size: int = 32,
):
    """
    Expanding-window evaluation for Moirai / Moirai-MoE model.

    Parameters
    ----------
    df : pd.DataFrame
        DatetimeIndex, 第一列为目标序列。
    model_choice : str
        "moirai" 或 "moirai-moe"
    size_choice : str
        模型大小：small/base/large
    prediction_length : int
        每个窗口的预测步长
    initial_train : int
        第一个窗口的训练长度
    max_context_length : int
        最大上下文长度
    batch_size : int
        Predictor batch size

    Returns
    -------
    metrics : dict
        {"MAE": ..., "RMSE": ..., "MAPE (%)": ...}
    forecast_df : pd.DataFrame
        ["ds", "moirai_expanding", "true"]，拼接所有窗口预测
    meta : list of tuples
        [(window_start_index, forecast_samples, true_array), ...] 用于绘图
    """
    # 转换成 PandasDataset
    ds = PandasDataset(dict(df))

    # 加载模型
    if model_choice == "moirai":
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size_choice}"),
            prediction_length=prediction_length,
            context_length=max_context_length,
            patch_size="auto",
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
    else:
        model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{size_choice}"),
            prediction_length=prediction_length,
            context_length=max_context_length,
            patch_size=16,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )

    predictor = model.create_predictor(batch_size=batch_size)

    # Expanding-window evaluation
    n = len(df)
    step = prediction_length
    preds_all = []
    trues_all = []
    ds_all = []
    meta = []
    metrics_data = []

    for w_start in range(initial_train, n - prediction_length + 1, step):
        train_w, test_w = split(ds, offset=-(n - w_start))
        instances = list(test_w.generate_instances(prediction_length=prediction_length, windows=1))
        if len(instances) == 0:
            continue
        input_data, label_data = instances[0]

        current_ctx = min(initial_train + (w_start - initial_train), max_context_length)
        if "past_target" in input_data:
            input_data["past_target"] = input_data["past_target"][-current_ctx:]
        if "past_feat_dynamic_real" in input_data and input_data["past_feat_dynamic_real"] is not None:
            input_data["past_feat_dynamic_real"] = input_data["past_feat_dynamic_real"][:, -current_ctx:]

        forecast = next(predictor.predict([input_data]))
        pred_mean = forecast.samples.mean(axis=0).squeeze()
        true_vals = np.array(label_data["target"], dtype=float)

        preds_all.extend(pred_mean.tolist())
        trues_all.extend(true_vals.tolist())
        ds_all.extend(df.index[w_start: w_start + prediction_length].tolist())
        meta.append((w_start, forecast.samples, true_vals))

        mae = mean_absolute_error(true_vals, pred_mean)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_mean))
        mape = np.mean(np.abs((true_vals - pred_mean) / (true_vals + 1e-8))) * 100
        metrics_data.append({"Window Start": w_start, "Context Length": current_ctx, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape})

    preds_all = np.array(preds_all)
    trues_all = np.array(trues_all)

    metrics = {k: np.mean([m[k] for m in metrics_data]) for k in ["MAE", "RMSE", "MAPE (%)"]}

    forecast_df = pd.DataFrame({
        "ds": pd.to_datetime(ds_all),
        "moirai_expanding": preds_all,
        "true": trues_all
    })

    return metrics, forecast_df, meta


if __name__ == "__main__":
    # quick CLI test
    import pandas as pd
    df_cli = pd.read_csv("ts_wide.csv", parse_dates=["date"]).set_index("date")
    metrics_cli, forecast_df_cli, meta_cli = run_moirai_expanding_forecast(
        df_cli,
        model_choice="moirai",
        size_choice="small",
        prediction_length=20,
        initial_train=100,
        max_context_length=200,
        batch_size=32,
    )
    print("Metrics:", metrics_cli)
    print(forecast_df_cli.head())


