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

model_choice = st.sidebar.selectbox("Select Model", ["moirai", "moirai-moe", "AutoARIMA", "AutoARIMA-expanding", "Chronos", "Chronos-expanding", "Multi-model Forecast Comparison"])
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
# AutoARIMA-expanding branch
# ----------------------------
elif model_choice == "AutoARIMA-expanding":
    st.write("### Running AutoARIMA (Expanding-window evaluation)")
    from autoarima_expanding import run_autoarima_expanding_forecast

    with st.spinner("Running AutoARIMA expanding-window evaluation..."):
        try:
            metrics, forecast_df, meta = run_autoarima_expanding_forecast(
                df,
                series_col=None,
                prediction_length=prediction_length,
                initial_train=initial_context_length,
                step=prediction_length,
                seasonal=False,          # 需要季节性可改 True
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                alpha=0.05,              # 95% 置信区间
            )
        except Exception as e:
            st.error(f"AutoARIMA expanding-window evaluation failed: {e}")
            st.stop()

    # 指标
    st.subheader("AutoARIMA Expanding Metrics")
    st.json(metrics)

    # 结果表
    st.subheader("AutoARIMA Expanding Forecast Values (head)")
    st.dataframe(forecast_df.head(50))

    # -------------------------
    # 绘图：Actual + 连续预测 + 阴影
    # -------------------------
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 5))

    # 真实值
    ax.plot(df.index, df.iloc[:, 0].astype(float).values, label="Actual", color="black")

    # 拼接所有窗口的预测均值与 std
    preds_all = []
    std_all = []
    for w_start, pred_mean, pred_std, true_vals in meta:
        preds_all.extend(pred_mean.tolist())
        std_all.extend(pred_std.tolist())

    preds_all = np.asarray(preds_all, dtype=float)
    std_all = np.asarray(std_all, dtype=float)

    # 与 forecast_df 对齐（稳妥起见）
    L = len(forecast_df)
    if len(preds_all) > L:
        preds_all = preds_all[:L]
        std_all = std_all[:L]
    elif len(preds_all) < L:
        pad = L - len(preds_all)
        preds_all = np.pad(preds_all, (0, pad), mode="edge")
        std_all = np.pad(std_all, (0, pad), mode="edge")

    # 连续预测曲线
    ax.plot(
        forecast_df["ds"],
        preds_all,
        label="AutoARIMA_expanding (all windows)",
        color="orange",
    )

    # 阴影：mean ± std （std 由CI换算得到）
    ax.fill_between(
        forecast_df["ds"],
        preds_all - std_all,
        preds_all + std_all,
        color="orange",
        alpha=0.2,
    )

    ax.set_title("AutoARIMA Expanding Forecast (All Windows)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)

    st.pyplot(fig)

    # -------------------------
    # 下载 CSV
    # -------------------------
    import io
    towrite = io.BytesIO()
    forecast_df.to_csv(towrite, index=False, encoding="utf-8-sig")
    towrite.seek(0)
    st.download_button(
        "Download AutoARIMA Expanding predictions CSV",
        towrite,
        file_name="autoarima_expanding_preds.csv",
        mime="text/csv"
    )



# ----------------------------
# Chronos branch
# ----------------------------
elif model_choice == "Chronos":
    st.write("### Running Chronos (Simple rolling forecast)")
    from Chronos import run_chronos_forecast

    with st.spinner("Forecasting with Chronos..."):
        try:
            forecast_df, fig_chronos = run_chronos_forecast(
                df,
                series_col=None,
                prediction_length=prediction_length,
                test_len=test_len,
                model_name=chronos_model_name,
                device=None,
            )
        except Exception as e:
            st.error(f"Chronos forecast failed: {e}")
            st.stop()

    st.subheader("Chronos Forecast Plot")
    st.pyplot(fig_chronos)

    st.subheader("Chronos Forecast Values (test period)")
    st.dataframe(forecast_df.set_index("ds"))

    # 下载按钮
    towrite = io.BytesIO()
    forecast_df.to_csv(towrite, index=False, encoding="utf-8-sig")
    towrite.seek(0)
    st.download_button(
        "Download Chronos predictions CSV",
        towrite,
        file_name="chronos_preds.csv",
        mime="text/csv"
    )

# ----------------------------
# Chronos-expanding branch
# ----------------------------
elif model_choice == "Chronos-expanding":
    st.write("### Running Chronos (Expanding-window evaluation)")
    from Chronos_expanding import run_chronos_expanding_forecast

    with st.spinner("Running Chronos expanding-window evaluation..."):
        try:
            metrics, forecast_df, meta = run_chronos_expanding_forecast(
                df,
                series_col=None,
                prediction_length=prediction_length,
                initial_train=initial_context_length,
                step=prediction_length,
                model_name=chronos_model_name,
                device=None,
            )
        except Exception as e:
            st.error(f"Chronos expanding-window evaluation failed: {e}")
            st.stop()

    # 显示指标
    st.subheader("Chronos Expanding Metrics")
    st.json(metrics)

    # 显示前50行预测值
    st.subheader("Chronos Expanding Forecast Values (head)")
    st.dataframe(forecast_df.head(50))

    # -------------------------
    # 绘图：所有窗口的连续预测曲线
    # -------------------------
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 5))

    # 真实值
    ax.plot(df.index, df.iloc[:, 0].astype(float).values, label="Actual", color="black")

    # 所有窗口预测
    preds_all = []
    for w_start, pred_samples, true_vals in meta:
        pred_mean = np.mean(pred_samples, axis=0)  # 平均值作为预测曲线
        preds_all.extend(pred_mean)

    # 绘制预测曲线
    ax.plot(forecast_df['ds'], preds_all, label="Chronos_expanding (all windows)", color="orange")

    # 可选：绘制预测不确定性阴影
    all_preds = np.concatenate([p for _, p, _ in meta], axis=0)
    pred_std = np.std(all_preds, axis=0)
    # 如果长度对不上 forecast_df，则截断或填充
    if len(pred_std) > len(forecast_df):
        pred_std = pred_std[:len(forecast_df)]
    elif len(pred_std) < len(forecast_df):
        pred_std = np.pad(pred_std, (0, len(forecast_df)-len(pred_std)), mode='edge')

    ax.fill_between(
        forecast_df['ds'],
        np.array(preds_all) - pred_std,
        np.array(preds_all) + pred_std,
        color="orange", alpha=0.2
    )

    ax.set_title("Chronos Expanding Forecast (All Windows)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)

    st.pyplot(fig)

    # -------------------------
    # 下载 CSV
    # -------------------------
    import io
    towrite = io.BytesIO()
    forecast_df.to_csv(towrite, index=False, encoding="utf-8-sig")
    towrite.seek(0)
    st.download_button(
        "Download Chronos Expanding predictions CSV",
        towrite,
        file_name="chronos_expanding_preds.csv",
        mime="text/csv"
    )

# ----------------------------
# Multi-model Forecast Comparison branch
# ----------------------------
elif model_choice == "Multi-model Forecast Comparison":
    st.write("### Multi-model Forecast Comparison")

    from Chronos import run_chronos_forecast
    from Chronos_expanding import run_chronos_expanding_forecast
    from autoarima_forecast import run_autoarima_forecast
    from autoarima_expanding import run_autoarima_expanding_forecast
    from sample_moirai import run_moirai_expanding_forecast  # Moirai expanding

    metrics_list = []
    forecast_dict = {}

    # ============ Chronos ============
    try:
        forecast_df, _ = run_chronos_forecast(
            df,
            series_col=None,
            prediction_length=prediction_length,
            test_len=test_len,
            model_name=chronos_model_name,
            device=None,
        )
        y_true = forecast_df["true"].values if "true" in forecast_df else df.iloc[-len(forecast_df):,0].values
        y_pred = forecast_df["chronos"].values if "chronos" in forecast_df else forecast_df.iloc[:,1].values
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))
        metrics_list.append({"Model": "Chronos", "MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape})
        forecast_dict["Chronos"] = (forecast_df["ds"], y_pred)
    except Exception as e:
        st.error(f"Chronos failed: {e}")

    # ============ Chronos-expanding ============
    try:
        metrics, forecast_df, meta = run_chronos_expanding_forecast(
            df,
            series_col=None,
            prediction_length=prediction_length,
            initial_train=initial_context_length,
            step=prediction_length,
            model_name=chronos_model_name,
            device=None,
        )
        mae, rmse, mape = metrics["MAE"], metrics["RMSE"], metrics["MAPE (%)"]
        smape = 100 * np.mean(2 * np.abs(forecast_df["chronos_expanding"].values - forecast_df["true"].values) /
                             (np.abs(forecast_df["chronos_expanding"].values) + np.abs(forecast_df["true"].values) + 1e-8))
        metrics_list.append({"Model": "Chronos-expanding", "MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape})
        forecast_dict["Chronos-expanding"] = (forecast_df["ds"], forecast_df["chronos_expanding"].values)
    except Exception as e:
        st.error(f"Chronos-expanding failed: {e}")

    # ============ AutoARIMA ============
    try:
        forecast_df, _ = run_autoarima_forecast(
            df,
            series_col=None,
            prediction_length=prediction_length,
            test_len=test_len,
        )
        y_true = forecast_df["true"].values if "true" in forecast_df else df.iloc[-len(forecast_df):,0].values
        y_pred = forecast_df["autoarima"].values if "autoarima" in forecast_df else forecast_df.iloc[:,1].values
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))
        metrics_list.append({"Model": "AutoARIMA", "MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape})
        forecast_dict["AutoARIMA"] = (forecast_df["ds"], y_pred)
    except Exception as e:
        st.error(f"AutoARIMA failed: {e}")

    # ============ AutoARIMA-expanding ============
    try:
        metrics, forecast_df, meta = run_autoarima_expanding_forecast(
            df,
            series_col=None,
            prediction_length=prediction_length,
            initial_train=initial_context_length,
            step=prediction_length,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            alpha=0.05,
        )
        mae, rmse, mape = metrics["MAE"], metrics["RMSE"], metrics["MAPE (%)"]
        smape = 100 * np.mean(2 * np.abs(forecast_df["autoarima_expanding"].values - forecast_df["true"].values) /
                             (np.abs(forecast_df["autoarima_expanding"].values) + np.abs(forecast_df["true"].values) + 1e-8))
        metrics_list.append({"Model": "AutoARIMA-expanding", "MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape})
        forecast_dict["AutoARIMA-expanding"] = (forecast_df["ds"], forecast_df["autoarima_expanding"].values)
    except Exception as e:
        st.error(f"AutoARIMA-expanding failed: {e}")

    # ============ Moirai-expanding ============
    try:
        metrics, forecast_df, meta = run_moirai_expanding_forecast(
            df,
            model_choice="moirai",   # 或 "moirai-moe"
            size_choice=size_choice,
            prediction_length=prediction_length,
            initial_train=initial_context_length,
            max_context_length=max_context_length,
            batch_size=batch_size,
        )

        # 计算指标
        y_true = forecast_df["true"].values if "true" in forecast_df else df.iloc[-len(forecast_df):,0].values
        preds_all = [np.mean(p, axis=0) for _, p, _ in meta]
        y_pred = np.concatenate(preds_all)
        # 对齐长度
        if len(y_pred) < len(y_true):
            y_pred = np.pad(y_pred, (len(y_true)-len(y_pred),0), mode='edge')
        elif len(y_pred) > len(y_true):
            y_pred = y_pred[-len(y_true):]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))

        metrics_list.append({"Model": "Moirai-expanding", "MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape})
        forecast_dict["Moirai-expanding"] = (forecast_df["ds"], y_pred)
    except Exception as e:
        st.error(f"Moirai-expanding failed: {e}")

    # ============ 显示对比表 ============
    if metrics_list:
        st.subheader("Comparison of Forecasting Models")
        results_df = pd.DataFrame(metrics_list).set_index("Model")
        st.dataframe(results_df)

    # ============ 绘制对比图 ============
    if forecast_dict:
        st.subheader("Forecast Comparison Plot")
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.index, df.iloc[:,0].astype(float), label="Actual", color="black")

        for name, (ds, y_pred) in forecast_dict.items():
            ax.plot(ds, y_pred, label=name)

        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)



# ----------------------------
# Moirai / Moirai-MoE expanding window branch (Expanding-window style)
# ----------------------------
else:
    st.write("### Moirai Expanding-Window Forecast")

    from sample_moirai import run_moirai_expanding_forecast

    with st.spinner("Running Moirai expanding-window evaluation..."):
        metrics, forecast_df, meta = run_moirai_expanding_forecast(
            df,
            model_choice=model_choice,
            size_choice=size_choice,
            prediction_length=prediction_length,
            initial_train=initial_context_length,
            max_context_length=max_context_length,
            batch_size=batch_size,
        )

    # 显示指标
    st.subheader("Moirai Expanding Metrics")
    st.json(metrics)

    # 显示前50行预测值
    st.subheader("Moirai Expanding Forecast Values (head)")
    st.dataframe(forecast_df.head(50))

    # 绘图
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df.index, df.iloc[:,0].astype(float), label="Actual", color="black")
    preds_plot = [np.mean(p, axis=0) for _, p, _ in meta]
    preds_plot = np.concatenate(preds_plot)
    ax.plot(forecast_df['ds'], preds_plot, label="Moirai Expanding", color="blue")
    # 阴影
    all_preds = np.concatenate([p for _, p, _ in meta], axis=0)
    pred_std = np.std(all_preds, axis=0)
    if len(pred_std) < len(forecast_df):
        pred_std = np.pad(pred_std, (0, len(forecast_df)-len(pred_std)), mode='edge')
    elif len(pred_std) > len(forecast_df):
        pred_std = pred_std[:len(forecast_df)]
    ax.fill_between(forecast_df['ds'], preds_plot - pred_std, preds_plot + pred_std, color="blue", alpha=0.2)
    ax.set_title("Moirai Expanding Forecast (All Windows)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 下载 CSV
    towrite = io.BytesIO()
    forecast_df.to_csv(towrite, index=False, encoding="utf-8-sig")
    towrite.seek(0)
    st.download_button(
        "Download Moirai Expanding predictions CSV",
        towrite,
        file_name="moirai_expanding_preds.csv",
        mime="text/csv"
    )








