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
import requests  # added to upload file to backend

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

# read the DataFrame for showing in the UI
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
# If user uploaded a local file, upload to FastAPI /data and save a local temp copy.
# ----------------------------
backend_file_id = None
csv_path = None  # fallback path used in requests if file_id isn't available

if uploaded_file is not None:
    # Save local temporary copy (backwards compatibility with endpoints expecting csv_path)
    try:
        csv_path = "temp_uploaded.csv"
        with open(csv_path, "wb") as f:
            # UploadedFile provides getbuffer(); use that to preserve bytes
            f.write(uploaded_file.getbuffer())
        st.info(f"Saved uploaded file locally to {csv_path}")
    except Exception as e:
        st.warning(f"Failed to save local temp copy: {e}")
        csv_path = None

    # Try to upload to backend /data (store in SQLite), get file_id
    try:
        st.info("Uploading file to backend DB (/data)...")
        # Use the bytes content for multipart/form-data
        # streamlit UploadedFile: .getvalue() or .getbuffer() both available; use getvalue() for requests
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        resp = requests.post("http://127.0.0.1:8000/data", files=files, timeout=60)
        if resp.status_code == 200:
            jr = resp.json()
            if jr.get("status") == "success" and "file_id" in jr:
                backend_file_id = int(jr["file_id"])
                st.success(f"Uploaded to backend DB (file_id={backend_file_id}). Subsequent requests will include this file_id.")
            else:
                # backend may return {"status":"success","file_id":...} OR {"error": "..."}
                if "error" in jr:
                    st.warning(f"Backend /data returned an error: {jr['error']}")
                else:
                    st.warning(f"Backend /data returned unexpected response: {jr}")
        else:
            st.warning(f"Backend /data upload failed: {resp.status_code} {resp.text}")
    except Exception as e:
        st.warning(f"Could not upload file to backend /data: {e}")

else:
    # No local upload -> set csv_path based on selected standard sources
    if data_choice == "Google Trends":
        csv_path = "ts_wide.csv"
    elif data_choice == "Baidu Trends":
        csv_path = "baidu_ts.csv"
    else:
        csv_path = None

# Helper: build request payload that prefers file_id (DB) then csv_path (local file on disk)
def make_req_payload(base: dict) -> dict:
    """
    base: dict containing commonly used keys (prediction_length, test_len, model_name, etc.)
    returns a copy that includes 'file_id' if backend_file_id is available, otherwise 'csv_path'.
    """
    req = base.copy()
    if backend_file_id is not None:
        req["file_id"] = backend_file_id
        # Some endpoints still expect csv_path; keep csv_path too for backward compatibility if available
        if csv_path is not None:
            req["csv_path"] = csv_path
    else:
        if csv_path is not None:
            req["csv_path"] = csv_path
    return req

# ----------------------------
# AutoARIMA branch via FastAPI
# ----------------------------
if model_choice == "AutoARIMA":
    st.write("### Running AutoARIMA (via FastAPI backend)")

    import io
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Build payload (prefer file_id)
    base = {
        "prediction_length": prediction_length,
        "test_len": test_len,
        "model_name": "autoarima"
    }
    req = make_req_payload(base)

    try:
        res = requests.post("http://127.0.0.1:8000/forecast_autoarima", json=req, timeout=10000)
    except Exception as e:
        st.error(f"Request to FastAPI backend failed: {e}")
        st.stop()

    if res.status_code == 200:
        result = res.json()
        if "error" in result:
            st.error(result["error"])
            st.stop()

        metrics = result.get("metrics", {})
        forecast_df = pd.DataFrame(result["predictions"])

        # ensure ds is datetime
        if "ds" in forecast_df.columns:
            forecast_df["ds"] = pd.to_datetime(forecast_df["ds"], errors="coerce")
        else:
            forecast_df = forecast_df.reset_index().rename(columns={forecast_df.index.name or 0: "ds"})
            forecast_df["ds"] = pd.to_datetime(forecast_df["ds"], errors="coerce")

        # numeric cols
        for col in ["true", "autoarima", "lower", "upper"]:
            if col in forecast_df.columns:
                forecast_df[col] = pd.to_numeric(forecast_df[col], errors="coerce")

        # sort df (safety)
        forecast_df = forecast_df.sort_values("ds")

        # ensure original df (full history) is sorted by date
        df = df.sort_index()

        # Display metrics
        st.subheader("AutoARIMA Forecast Metrics (from FastAPI)")
        st.json(metrics)

        # Displaying the result table
        st.subheader("AutoARIMA Forecast Values (head)")
        st.dataframe(forecast_df.head(50))

        # Plot: whole history + prediction segment overlay
        fig, ax = plt.subplots(figsize=(12, 5))

        # plot full actual history (first column)
        ax.plot(df.index, df.iloc[:, 0].astype(float).values, label="Actual (history)", color="black")

        # overlay forecast (test period)
        x = forecast_df["ds"]
        y_pred = forecast_df["autoarima"].astype(float).values
        lower = forecast_df["lower"].astype(float).values if "lower" in forecast_df.columns else y_pred * 0.9
        upper = forecast_df["upper"].astype(float).values if "upper" in forecast_df.columns else y_pred * 1.1

        ax.plot(x, y_pred, label="AutoARIMA Forecast", color="green")
        ax.fill_between(x, lower, upper, color="green", alpha=0.2)

        # mark train/test split if available
        try:
            if "train_len" in metrics and metrics["train_len"] is not None:
                split_idx = int(metrics["train_len"])
                if 0 <= split_idx < len(df):
                    ax.axvline(x=df.index[split_idx], color="gray", linestyle="--", label="Train/Test split")
        except Exception:
            pass

        ax.set_title("AutoARIMA Forecast (history + test overlay)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # debug quick stats to help diagnose flatness
        st.write("Prediction summary:")
        st.write({
            "pred_mean": float(np.nanmean(y_pred)),
            "pred_std": float(np.nanstd(y_pred)),
            "pred_min": float(np.nanmin(y_pred)),
            "pred_max": float(np.nanmax(y_pred))
        })

        # Download CSV
        towrite = io.BytesIO()
        forecast_df.to_csv(towrite, index=False, encoding="utf-8-sig")
        towrite.seek(0)
        st.download_button("Download AutoARIMA predictions CSV", towrite, file_name="autoarima_preds.csv", mime="text/csv")
    else:
        st.error(f"FastAPI request failed: {res.status_code} {res.text}")

# ----------------------------
# AutoARIMA-expanding branch
# ----------------------------
elif model_choice == "AutoARIMA-expanding":
    st.write("### Running AutoARIMA (Expanding-window evaluation via FastAPI)")

    import io
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    base = {
        "prediction_length": prediction_length,
        "test_len": test_len,
        "initial_train": initial_context_length,
        "model_name": "autoarima_expanding"
    }
    req = make_req_payload(base)

    try:
        res = requests.post("http://127.0.0.1:8000/forecast_autoarima_expanding", json=req, timeout=600)
    except Exception as e:
        st.error(f"Request to FastAPI backend failed: {e}")
        st.stop()

    if res.status_code == 200:
        result = res.json()
        if "error" in result:
            st.error(result["error"])
            st.stop()

        metrics = result.get("metrics", {})
        forecast_df = pd.DataFrame(result["predictions"])

        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"], errors="coerce")
        forecast_df = forecast_df.sort_values("ds")

        # Show the results
        st.subheader("AutoARIMA Expanding Forecast Metrics (from FastAPI)")
        st.json(metrics)

        st.subheader("AutoARIMA Expanding Forecast Values (head)")
        st.dataframe(forecast_df.head(50))

        # Drawing
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df.iloc[:, 0].astype(float).values, label="Actual", color="black")

        preds = forecast_df["autoarima_expanding"].astype(float).values
        std = forecast_df["std"].astype(float).values if "std" in forecast_df.columns else np.zeros_like(preds)

        ax.plot(forecast_df["ds"], preds, label="AutoARIMA-expanding Forecast", color="orange")
        ax.fill_between(forecast_df["ds"], preds - std, preds + std, color="orange", alpha=0.2)

        ax.set_title("AutoARIMA Expanding Forecast (All Windows)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Download CSV
        towrite = io.BytesIO()
        forecast_df.to_csv(towrite, index=False, encoding="utf-8-sig")
        towrite.seek(0)
        st.download_button(
            "Download AutoARIMA Expanding predictions CSV",
            towrite,
            file_name="autoarima_expanding_preds.csv",
            mime="text/csv"
        )
    else:
        st.error(f"FastAPI request failed: {res.status_code} {res.text}")

# ----------------------------
# Chronos branch via FastAPI backend
# ----------------------------
elif model_choice == "Chronos":
    st.write("### Running Chronos (via FastAPI backend)")

    import matplotlib
    matplotlib.use("Agg")  
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import io

    base = {
        "prediction_length": prediction_length,
        "test_len": test_len,
        "model_name": chronos_model_name,
    }
    req = make_req_payload(base)

    try:
        res = requests.post("http://127.0.0.1:8000/forecast", json=req)
    except Exception as e:
        st.error(f"Request to FastAPI backend failed: {e}")
        st.stop()

    if res.status_code == 200:
        result = res.json()

        # Parsing backend returns
        metrics = result.get("metrics", {})
        forecast_df = pd.DataFrame(result["predictions"])
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

        # -------------------
        # Display metrics
        # -------------------
        st.subheader("Chronos Forecast Metrics (from FastAPI)")
        st.json(metrics)

        # -------------------
        # Displaying the result table
        # -------------------
        st.subheader("Chronos Forecast Values (head)")
        st.dataframe(forecast_df.head(50))

        # -------------------
        # Plot: actual + predicted values
        # -------------------
        fig, ax = plt.subplots(figsize=(12,5))

        # True value
        ax.plot(df.index, df.iloc[:,0].astype(float).values, label="Actual", color="black")

        # Predicted value
        preds = forecast_df["chronos"].values if "chronos" in forecast_df.columns else forecast_df.iloc[:,1].values
        ax.plot(forecast_df["ds"], preds, label="Chronos Forecast", color="orange")

        # Add a simple uncertainty shadow
        if "std" in forecast_df.columns:
            ax.fill_between(forecast_df["ds"], preds - forecast_df["std"], preds + forecast_df["std"],
                            color="orange", alpha=0.2)

        ax.set_title("Chronos Forecast (All Windows Style)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)

        st.pyplot(fig)

        # -------------------
        # Download CSV
        # -------------------
        towrite = io.BytesIO()
        forecast_df.to_csv(towrite, index=False, encoding="utf-8-sig")
        towrite.seek(0)
        st.download_button(
            "Download Chronos predictions CSV",
            towrite,
            file_name="chronos_preds.csv",
            mime="text/csv"
        )
    else:
        st.error(f"FastAPI request failed: {res.text}")

# ----------------------------
# Chronos-expanding branch
# ----------------------------
elif model_choice == "Chronos-expanding":
    st.write("### Running Chronos (Expanding-window evaluation via FastAPI)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import io

    base = {
        "prediction_length": prediction_length,
        "test_len": test_len,
        "model_name": chronos_model_name,
        "initial_train": initial_context_length,
    }
    req = make_req_payload(base)

    try:
        res = requests.post("http://127.0.0.1:8000/forecast_expanding", json=req)
    except Exception as e:
        st.error(f"Request to FastAPI backend failed: {e}")
        st.stop()

    if res.status_code == 200:
        result = res.json()

        # Parse return
        metrics = result.get("metrics", {})
        forecast_df = pd.DataFrame(result["predictions"])
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

        # Metrics
        st.subheader("Chronos Expanding Forecast Metrics (from FastAPI)")
        st.json(metrics)

        # first 50 lines
        st.subheader("Chronos Expanding Forecast Values (head)")
        st.dataframe(forecast_df.head(50))

        # plot
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(df.index, df.iloc[:,0].astype(float).values, label="Actual", color="black")
        preds = forecast_df["chronos_expanding"].values
        ax.plot(forecast_df["ds"], preds, label="Chronos-expanding Forecast", color="orange")

        # uncertainty shadow
        if "std" in forecast_df.columns:
            ax.fill_between(forecast_df["ds"], preds - forecast_df["std"], preds + forecast_df["std"], color="orange", alpha=0.2)

        ax.set_title("Chronos Expanding Forecast (All Windows)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)

        st.pyplot(fig)

        # download CSV
        towrite = io.BytesIO()
        forecast_df.to_csv(towrite, index=False, encoding="utf-8-sig")
        towrite.seek(0)
        st.download_button(
            "Download Chronos Expanding predictions CSV",
            towrite,
            file_name="chronos_expanding_preds.csv",
            mime="text/csv"
        )
    else:
        st.error(f"FastAPI request failed: {res.text}")

# ----------------------------
# Multi-model Forecast Comparison branch via FastAPI
# ----------------------------
elif model_choice == "Multi-model Forecast Comparison":
    st.write("### Multi-model Forecast Comparison via FastAPI")

    import io
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import time

    # csv_path already prepared above (either temp_uploaded.csv or standard files)
    if uploaded_file is not None:
        csv_path = "temp_uploaded.csv"
    else:
        if data_choice == "Google Trends":
            csv_path = "ts_wide.csv"
        elif data_choice == "Baidu Trends":
            csv_path = "baidu_ts.csv"
        else:
            st.error("Unknown data source")
            st.stop()

    models_api = {
        "Chronos": "forecast",
        "Chronos-expanding": "forecast_expanding",
        "AutoARIMA": "forecast_autoarima",
        "AutoARIMA-expanding": "forecast_autoarima_expanding",
        "Moirai-expanding": "forecast_moirai_expanding"
    }

    metrics_list = []
    forecast_dict = {}

    total = len(models_api)
    prog = st.progress(0)
    idx = 0

    # Choose a per-model timeout (seconds).
    PER_MODEL_TIMEOUT = 120

    for model_name, endpoint in models_api.items():
        idx += 1
        prog.progress((idx-1) / total)
        st.write(f"Running **{model_name}** (endpoint: `{endpoint}`)...")

        base = {
            "prediction_length": prediction_length,
            "test_len": test_len,
            "initial_train": initial_context_length,
            "model_name": chronos_model_name if "Chronos" in model_name else "autoarima" if "AutoARIMA" in model_name else "moirai",
            "size_choice": size_choice if "Moirai" in model_name else None,
            "max_context_length": max_context_length if "Moirai" in model_name else None,
            "batch_size": batch_size if "Moirai" in model_name else None
        }
        # remove None values to keep payload clean
        base = {k: v for k, v in base.items() if v is not None}
        req = make_req_payload(base)

        url = f"http://127.0.0.1:8000/{endpoint}"
        start_time = time.time()
        try:
            res = requests.post(url, json=req, timeout=PER_MODEL_TIMEOUT)
            elapsed = time.time() - start_time

            if res.status_code != 200:
                st.warning(f"{model_name} FastAPI request failed: {res.status_code} {res.text}")
                continue

            result = res.json()
            if "error" in result:
                st.warning(f"{model_name} failed: {result['error']}")
                continue

            # Set up the DataFrame and do a strict check
            preds = result.get("predictions", [])
            if not preds:
                st.warning(f"{model_name} returned empty predictions.")
                continue

            forecast_df = pd.DataFrame(preds)

            # ensure ds exists and is datetime
            if "ds" not in forecast_df.columns:
                st.warning(f"{model_name} returned predictions without 'ds' column, skipping.")
                continue
            forecast_df["ds"] = pd.to_datetime(forecast_df["ds"], errors="coerce")
            forecast_df = forecast_df.dropna(subset=["ds"])  # remove invalid dates
            if forecast_df.empty:
                st.warning(f"{model_name} produced no valid date entries after parsing 'ds'.")
                continue

            # convert known numeric cols to numeric (safe)
            for col in ["true", "chronos", "chronos_expanding", "autoarima", "autoarima_expanding", "moirai_expanding", "lower", "upper", "std"]:
                if col in forecast_df.columns:
                    forecast_df[col] = pd.to_numeric(forecast_df[col], errors="coerce")

            # Align the true and predicted values: take the overlapping tail length of the two
            n_pred = len(forecast_df)
            n_true = len(df)
            n = min(n_pred, n_true)
            if n == 0:
                st.warning(f"{model_name} no overlapping rows with actual data (n_pred={n_pred}, n_true={n_true}).")
                continue

            # take last n rows for alignment
            forecast_df_al = forecast_df.tail(n).reset_index(drop=True)
            y_true = df.iloc[-n:, 0].astype(float).values

            # pick prediction column based on model
            if model_name == "Chronos":
                if "chronos" in forecast_df_al.columns:
                    y_pred = forecast_df_al["chronos"].astype(float).values
                else:
                    # fallback to second column if named differently
                    y_pred = forecast_df_al.iloc[:, 1].astype(float).values
            elif model_name == "Chronos-expanding":
                y_pred = forecast_df_al["chronos_expanding"].astype(float).values
            elif model_name == "AutoARIMA":
                y_pred = forecast_df_al["autoarima"].astype(float).values
            elif model_name == "AutoARIMA-expanding":
                y_pred = forecast_df_al["autoarima_expanding"].astype(float).values
            elif model_name == "Moirai-expanding":
                y_pred = forecast_df_al["moirai_expanding"].astype(float).values
            else:
                # generic fallback
                numeric_cols = forecast_df_al.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 1:
                    y_pred = forecast_df_al[numeric_cols[0]].astype(float).values
                else:
                    st.warning(f"{model_name}: no numeric prediction column found, skipping.")
                    continue

            # ensure shape matches n (trim/pad conservatively)
            if len(y_pred) > n:
                y_pred = y_pred[-n:]
            elif len(y_pred) < n:
                if len(y_pred) == 0:
                    st.warning(f"{model_name} returned empty numeric predictions after selection.")
                    continue
                # pad with last value (conservative)
                pad = n - len(y_pred)
                y_pred = np.pad(y_pred, (pad, 0), mode="edge")

            # Calculate metrics (defensive: catch exceptions)
            try:
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))
            except Exception as e:
                st.warning(f"{model_name} metric computation failed: {e}")
                mae = rmse = mape = smape = None

            metrics_list.append({"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape})
            # Stores ds/y_pred (the tail aligned with y_true) for comparison plots.
            forecast_dict[model_name] = (forecast_df_al["ds"].values, y_pred)

            st.write(f"✔ {model_name} finished in {elapsed:.1f}s (aligned n={n}).")

        except requests.exceptions.Timeout:
            st.warning(f"{model_name} request timed out after {PER_MODEL_TIMEOUT}s. Skipping.")
        except Exception as e:
            st.warning(f"{model_name} request failed: {e}")

    prog.progress(1.0)

    # Display comparison table
    if metrics_list:
        st.subheader("Comparison of Forecasting Models")
        results_df = pd.DataFrame(metrics_list).set_index("Model")
        st.dataframe(results_df)

    # ============ Draw a comparison chart ============
    if forecast_dict:
        st.subheader("Forecast Comparison Plot")
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df.index, df.iloc[:,0].astype(float), label="Actual", color="black")

        for name, (ds, y_pred) in forecast_dict.items():
            # ensure ds is datetime-like for plotting
            ds = pd.to_datetime(ds)
            ax.plot(ds, y_pred, label=name)

        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# ----------------------------
# Moirai / Moirai-MoE expanding window branch (via FastAPI backend)
# ----------------------------
else:
    st.write("### Moirai Expanding-Window Forecast (via FastAPI)")

    import matplotlib
    matplotlib.use("Agg")  # Backend Security
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import io

    base = {
        "prediction_length": prediction_length,
        "test_len": test_len,
        "model_name": model_choice,
        "initial_train": initial_context_length,
        "size_choice": size_choice,
        "max_context_length": max_context_length,
        "batch_size": batch_size,
    }
    # cleanup None values
    base = {k: v for k, v in base.items() if v is not None}
    req = make_req_payload(base)

    try:
        res = requests.post("http://127.0.0.1:8000/forecast_moirai_expanding", json=req)
    except Exception as e:
        st.error(f"Request to FastAPI backend failed: {e}")
        st.stop()

    if res.status_code == 200:
        result = res.json()
        metrics = result.get("metrics", {})
        forecast_df = pd.DataFrame(result["predictions"])
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

        # -------------------
        # Display metrics
        # -------------------
        st.subheader("Moirai Expanding Metrics (from FastAPI)")
        st.json(metrics)

        # -------------------
        # Displays the predictions for the first 50 rows
        # -------------------
        st.subheader("Moirai Expanding Forecast Values (head)")
        st.dataframe(forecast_df.head(50))

        # -------------------
        # plot
        # -------------------
        fig, ax = plt.subplots(figsize=(12,5))

        # True value
        ax.plot(df.index, df.iloc[:,0].astype(float).values, label="Actual", color="black")

        # Predicted value
        preds_plot = forecast_df["moirai_expanding"].values
        ax.plot(forecast_df['ds'], preds_plot, label="Moirai Expanding", color="blue")

        # Shadow: Estimate std
        if "std" in forecast_df.columns:
            ax.fill_between(forecast_df['ds'], preds_plot - forecast_df["std"], preds_plot + forecast_df["std"], color="blue", alpha=0.2)
        else:
            # If there is no std, do simple ±10% assumed shading
            ax.fill_between(forecast_df['ds'], preds_plot*0.9, preds_plot*1.1, color="blue", alpha=0.2)

        ax.set_title("Moirai Expanding Forecast (All Windows)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # -------------------
        # download CSV
        # -------------------
        towrite = io.BytesIO()
        forecast_df.to_csv(towrite, index=False, encoding="utf-8-sig")
        towrite.seek(0)
        st.download_button(
            "Download Moirai Expanding predictions CSV",
            towrite,
            file_name="moirai_expanding_preds.csv",
            mime="text/csv"
        )
    else:
        st.error(f"FastAPI request failed: {res.text}")






