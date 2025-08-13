# streamlit_app.py
import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
import matplotlib.dates as mdates

# Moirai pipeline imports
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

# AutoARIMA helper
from autoarima_forecast import run_autoarima_forecast

# Chronos helper
import chronos_model

st.set_page_config(page_title="Moirai Time Series Forecasting", layout="wide")
st.title("Moirai / AutoARIMA / Chronos Forecast App")

# ----------------------------
# Sidebar: Model & Data Options
# ----------------------------
st.sidebar.title("Configuration")

model_choice = st.sidebar.selectbox("Select Model", ["moirai", "moirai-moe", "AutoARIMA", "Chronos"])
size_choice = st.sidebar.selectbox("Model Size", ["small", "base", "large"])
prediction_length = st.sidebar.slider("Prediction Length", 4, 52, 20)
initial_context_length = st.sidebar.slider("Initial Context Length", 20, 500, 100)
max_context_length = st.sidebar.slider("Max Context Length", 20, 500, 200)
batch_size = st.sidebar.slider("Batch Size", 1, 64, 32)
test_len = st.sidebar.slider("Test Set Length", 20, 200, 100)

chronos_model_name = st.sidebar.selectbox("Chronos model", ["amazon/chronos-t5-small", "amazon/chronos-bolt-small"])

data_choice = st.sidebar.selectbox("Select Data Source", ["Google Trends", "Baidu Trends", "Local CSV"])
uploaded_file = None
if data_choice == "Local CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV (date,value)", type=["csv"])

st.write("### Loading time series data...")

def load_data(choice):
    if choice == "Google Trends":
        file_path = "ts_wide.csv"
    elif choice == "Baidu Trends":
        file_path = "baidu_ts.csv"
    else:
        return None
    try:
        return pd.read_csv(file_path, parse_dates=["date"]).set_index("date")
    except Exception as e:
        st.error(f"Failed to read {file_path}: {e}")
        return None

def load_uploaded(filebuf):
    try:
        df = pd.read_csv(filebuf, parse_dates=["date"]).set_index("date")
        return df
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        return None

if data_choice == "Local CSV":
    if uploaded_file is None:
        st.warning("Please upload a local CSV file with columns ['date','value']")
        st.stop()
    df = load_uploaded(uploaded_file)
else:
    df = load_data(data_choice)

if df is None or df.empty:
    st.error("Failed to load data. Please check your file.")
    st.stop()

st.dataframe(df.head())

# ----------------------------
# AutoARIMA branch
# ----------------------------
if model_choice == "AutoARIMA":
    st.write("### Running AutoARIMA...")
    with st.spinner("Training and forecasting with AutoARIMA..."):
        try:
            forecast_df, fig_arima = run_autoarima_forecast(
                df,
                series_col=None,
                prediction_length=prediction_length,
                test_len=test_len,
            )
        except Exception as e:
            st.error(f"AutoARIMA failed: {e}")
            st.stop()

    st.subheader("AutoARIMA Forecast Plot")
    st.pyplot(fig_arima)
    st.subheader("AutoARIMA Forecast Values (test period)")
    st.dataframe(forecast_df.set_index("ds"))

# ----------------------------
# Chronos branch
# ----------------------------
elif model_choice == "Chronos":
    st.write("### Running Chronos (Expanding-window evaluation)")
    with st.spinner("Running Chronos expanding-window evaluation..."):
        try:
            metrics, out_df, meta = chronos_model.expanding_window_forecast(
                df,
                prediction_length=prediction_length,
                initial_train=initial_context_length,
                step=prediction_length,
                model_name=chronos_model_name,
                device=None,
            )
        except Exception as e:
            st.error(f"Chronos expanding-window evaluation failed: {e}")
            st.stop()

    st.subheader("Chronos metrics")
    st.json(metrics)
    st.subheader("Predictions vs True (head)")
    st.dataframe(out_df.head(50))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df.iloc[:, 0].astype(float).values, label="Actual")
    last_wend, last_pred, last_true = meta[-1]
    pred_idx = df.index[last_wend:last_wend + prediction_length]
    ax.plot(pred_idx, last_pred, label="Chronos_pred (last window)", marker="o")
    ax.legend()
    st.pyplot(fig)

    towrite = io.BytesIO()
    out_df.to_csv(towrite, index=False, encoding="utf-8-sig")
    towrite.seek(0)
    st.download_button("Download Chronos predictions CSV", towrite, file_name="chronos_expanding_preds.csv", mime="text/csv")

# ----------------------------
# Moirai / Moirai-MoE expanding window branch
# ----------------------------
else:
    st.write("### Moirai Dynamic Expanding Window Forecast")

    ds = PandasDataset(dict(df))

    # load model
    st.write("### Loading model...")
    with st.spinner("Downloading and initializing model..."):
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

    windows = test_len // prediction_length
    all_forecasts, all_labels, all_inputs = [], [], []
    metrics_data = []

    for w in range(windows):
        current_ctx = min(initial_context_length + w * prediction_length, max_context_length)
        offset = -(test_len - w * prediction_length)

        # split dataset for current window
        train_w, test_w = split(ds, offset=offset)

        # generate one instance for current window (no context_length param here!)
        instances = test_w.generate_instances(
            prediction_length=prediction_length,
            windows=1,
            distance=1,
        )
        instance_list = list(instances)
        if len(instance_list) == 0:
            st.warning(f"No instance generated for window {w+1}")
            continue

        input_data, label_data = instance_list[0]

        # 截断上下文长度
        if "past_target" in input_data:
            input_data["past_target"] = input_data["past_target"][-current_ctx:]
        if "past_feat_dynamic_real" in input_data and input_data["past_feat_dynamic_real"] is not None:
            input_data["past_feat_dynamic_real"] = input_data["past_feat_dynamic_real"][:, -current_ctx:]

        # 预测
        forecast = next(predictor.predict([input_data]))

        # 取出真实值数组
        true_vals = np.array(label_data["target"], dtype=float)
        pred_vals = forecast.samples.mean(axis=0).squeeze()

        all_forecasts.append(forecast)
        all_labels.append(true_vals)
        all_inputs.append(input_data)

        # 计算误差指标
        mae = mean_absolute_error(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        mape = np.mean(np.abs((true_vals - pred_vals) / (true_vals + 1e-8))) * 100

        metrics_data.append({
            "Window": w + 1,
            "Context Length": current_ctx,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape
        })

    import matplotlib.dates as mdates
    fig_all, axes = plt.subplots(windows, 1, figsize=(12, 3 * windows))
    if windows == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        # 当前窗口预测起始在整个df里的索引位置
        window_start_idx = len(df) - test_len + idx * prediction_length

        # 历史数据索引范围到预测结束
        history_end_idx = window_start_idx + prediction_length
        history_end_idx = min(history_end_idx, len(df))  # 防止越界

        # 取完整历史真实数据（从头到预测结束）
        time_all = df.index[:history_end_idx]
        values_all = df.iloc[:history_end_idx, 0].values

        # 预测期时间索引
        pred_time = df.index[window_start_idx : window_start_idx + prediction_length]

        # 预测均值
        pred_mean = all_forecasts[idx].samples.mean(axis=0).squeeze()

        # 画完整历史真实数据线（黑色实线）
        ax.plot(time_all, values_all, label="Historical Data", color="black", linewidth=1.5)

        # 画预测期预测均值线（蓝色实线）
        ax.plot(pred_time, pred_mean, label="Forecast (mean)", color="blue", linewidth=2)

        ax.set_title(f"Window {idx+1} Forecast with Full History")
        ax.legend()
        ax.grid(True)

        # 设置x轴为日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        # 关键修改：单独旋转每个子图的x轴刻度标签
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')

    # 不要在这里调用 fig_all.autofmt_xdate()
    # 调整子图间垂直间距，防止重叠
    fig_all.subplots_adjust(hspace=0.7)  # 这里0.5可以根据需求调整

    st.subheader("All Expanding Windows Forecasts (Full History + Forecast)")
    st.pyplot(fig_all)


    # 侧边栏选择窗口，默认第1个窗口
    selected_window = st.sidebar.selectbox(
        "Select Window to View Forecast Data",
        options=list(range(1, windows + 1)),
        index=0
    )

    idx = selected_window - 1
    df_window = pd.DataFrame({
        "TimeStep": range(len(all_labels[idx])),
        "True Value": all_labels[idx],
        "Predicted Mean": all_forecasts[idx].samples.mean(axis=0).squeeze()
    })

    st.subheader(f"Forecast and True Values for Window {selected_window}")
    st.dataframe(df_window)


    # 显示误差指标表
    st.subheader("Forecast Error Metrics by Window")
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.style.format({
        "MAE": "{:.4f}",
        "RMSE": "{:.4f}",
        "MAPE (%)": "{:.2f}"
    }))





