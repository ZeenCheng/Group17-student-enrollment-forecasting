# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import sqlite3
import os
import numpy as np
from fastapi import UploadFile, File
import io
import datetime
from typing import Optional


from Chronos import run_chronos_forecast
from Chronos_expanding import run_chronos_expanding_forecast
from sample_moirai import run_moirai_expanding_forecast

# Import autoarima helper for API
from autoarima_forecast import run_autoarima_forecast_api
from autoarima_expanding import run_autoarima_expanding_forecast

app = FastAPI(title="Forecast API")

DB_FILE = "database.db"

# ----------- Utility function: load data from SQLite ----------- #
def load_data_from_db(file_id: int) -> pd.DataFrame:
    """
    Load time series data with a given file_id from SQLite `data` table.
    Returns: DataFrame (index=date, column named value).
    """
    conn = sqlite3.connect(DB_FILE)
    query = """
        SELECT date, value
        FROM data
        WHERE file_id = ?
        ORDER BY date
    """
    df = pd.read_sql_query(query, conn, params=(file_id,))
    conn.close()

    if df.empty:
        raise ValueError(f"No data found for file_id={file_id}")

    # Convert date and set as index
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])  # ensure valid dates
    df = df.set_index("date").sort_index()

    # Ensure value column is numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])  # drop rows without values

    # Return normalized wide format
    df_wide = df[["value"]].copy()
    return df_wide


def get_dataframe_for_request(req) -> pd.DataFrame:
    """
    Get input DataFrame from a ForecastRequest-like object.
    Priority: file_id from database → fallback to csv_path.
    Returns DataFrame with datetime index and at least one column.
    """
    # prefer database file_id
    if getattr(req, "file_id", None) is not None:
        try:
            df = load_data_from_db(int(req.file_id))
            return df
        except Exception as e:
            raise

    # fallback to csv_path
    if getattr(req, "csv_path", None):
        csv_path = req.csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
        return df

    raise ValueError("Either file_id or csv_path must be provided in the request")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            ds TEXT,
            value REAL
        )
    """)

    # files table (manage uploaded CSV files)
    c.execute("""
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # data table (store each row of CSV)
    c.execute("""
    CREATE TABLE IF NOT EXISTS data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER,
        date TEXT,
        value REAL,
        FOREIGN KEY (file_id) REFERENCES files (id) ON DELETE CASCADE
    )
    """)

    # Indexes for faster lookup
    c.execute("CREATE INDEX IF NOT EXISTS idx_data_file_id ON data (file_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_data_date ON data (date)")
    conn.commit()
    conn.close()

init_db()

class ForecastRequest(BaseModel):
    # Either option is acceptable:
    csv_path: str | None = None
    file_id: int | None = None

    prediction_length: int
    test_len: int
    model_name: str
    initial_train: int | None = None
    size_choice: str | None = None
    max_context_length: int | None = None
    batch_size: int | None = None


# ---------- POST /data (upload CSV and write into SQLite) ----------
@app.post("/data")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload a CSV file (must contain `date` and `value` columns, or at least two columns: date,value).
    Returns: {"status":"success","file_id":..., "rows": n} or {"error": "..."}
    """
    try:
        contents = await file.read()
        text = contents.decode("utf-8")  # Assume UTF-8; adjust for BOM/other encodings if needed
        buf = io.StringIO(text)

        # Expect at least date column; value column can be 'value' or the second column
        df = pd.read_csv(buf, parse_dates=["date"], dayfirst=False)  # set dayfirst=True if date format requires

        # If no 'date' column, try first column as date
        if "date" not in df.columns:
            df = pd.read_csv(io.StringIO(text), header=0, parse_dates=[0])
            df.columns = ["date"] + [f"col{i}" for i in range(1, len(df.columns))]
        
        # Try to detect value column (prefer 'value')
        if "value" in df.columns:
            val_col = "value"
        else:
            # fallback: use second column
            val_col = df.columns[1]

        # Convert and validate
        df = df[["date", val_col]].copy()
        df.rename(columns={val_col: "value"}, inplace=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Drop rows with invalid date
        df = df.dropna(subset=["date"])

        # Insert file metadata and batch insert data rows
        conn = sqlite3.connect(DB_FILE)
        conn.execute("PRAGMA foreign_keys = ON")
        cur = conn.cursor()

        cur.execute("INSERT INTO files (filename) VALUES (?)", (file.filename,))
        file_id = cur.lastrowid

        # Prepare rows (convert date to ISO string)
        rows = []
        for _, r in df.iterrows():
            d = r["date"]
            if isinstance(d, pd.Timestamp):
                d_str = d.isoformat()  # e.g. '2023-01-01T00:00:00'
            else:
                d_str = str(d)
            v = None if pd.isna(r["value"]) else float(r["value"])
            rows.append((file_id, d_str, v))

        # Bulk insert
        cur.executemany("INSERT INTO data (file_id, date, value) VALUES (?, ?, ?)", rows)
        conn.commit()
        conn.close()

        return {"status": "success", "file_id": file_id, "rows": len(rows)}

    except Exception as e:
        return {"error": f"Failed to upload data: {e}"}

# ---------- GET /data (list uploaded files) ----------
@app.get("/data")
def list_uploaded_files():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, filename, uploaded_at FROM files ORDER BY uploaded_at DESC")
    rows = c.fetchall()
    conn.close()
    files = [{"id": r[0], "filename": r[1], "uploaded_at": r[2]} for r in rows]
    return {"files": files}

# ---------- GET /data/{file_id} (read rows of a file) ----------
@app.get("/data/{file_id}")
def get_file_data(file_id: int, limit: int | None = None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    q = "SELECT date, value FROM data WHERE file_id = ? ORDER BY date"
    if limit is not None:
        q += " LIMIT ?"
        c.execute(q, (file_id, limit))
    else:
        c.execute(q, (file_id,))
    rows = c.fetchall()
    conn.close()
    data = [{"date": r[0], "value": r[1]} for r in rows]
    return {"file_id": file_id, "rows": len(data), "data": data}


# ----------- Chronos standard-----------
@app.post("/forecast")
def forecast(req: ForecastRequest):
    try:
        # Get input data (priority: file_id → csv_path)
        try:
            df = get_dataframe_for_request(req)
        except FileNotFoundError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Failed to load input data: {e}"}

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                return {"error": "Input index could not be parsed as dates"}

        # Ensure at least one value column
        if df.shape[1] >= 1:
            # keep as-is; Chronos helper expects a DataFrame with DatetimeIndex and values in first column
            pass
        else:
            return {"error": "Input data must contain at least one value column"}

        # Run Chronos
        forecast_df, metrics = run_chronos_forecast(
            df,
            prediction_length=req.prediction_length,
            test_len=req.test_len,
            model_name=req.model_name,
        )

        # Compatibility handling
        if isinstance(metrics, dict) and "predictions" in forecast_df:
            # unlikely path, but keep compatibility
            pass

        # Ensure forecast_df has ds column
        if "ds" not in forecast_df.columns:
            forecast_df = forecast_df.reset_index().rename(columns={forecast_df.index.name or 0: "ds"})

        # Save into SQLite
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        model_label = "Chronos"
        # choose forecast column to store: prefer column named 'chronos', else second column or first numeric
        store_col = None
        if "chronos" in forecast_df.columns:
            store_col = "chronos"
        else:
            # pick the first numeric column other than 'ds' if possible
            for col in forecast_df.columns:
                if col == "ds":
                    continue
                if pd.api.types.is_numeric_dtype(forecast_df[col]):
                    store_col = col
                    break
        if store_col is None:
            # fallback: try second column
            if len(forecast_df.columns) >= 2:
                store_col = forecast_df.columns[1]

        for _, row in forecast_df.iterrows():
            ds_str = str(row["ds"])
            try:
                val = float(row[store_col]) if store_col is not None and not pd.isna(row[store_col]) else None
            except Exception:
                val = None
            if val is not None:
                c.execute("INSERT INTO forecasts (model, ds, value) VALUES (?, ?, ?)", (model_label, ds_str, val))
        conn.commit()
        conn.close()

        forecast_out = forecast_df.copy()
        forecast_out["ds"] = pd.to_datetime(forecast_out["ds"]).astype(str)

        return {
            "model": model_label,
            "metrics": metrics if isinstance(metrics, dict) else {},
            "predictions": forecast_out.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": f"Chronos forecast failed: {e}"}

# Query forecast history
@app.get("/history")
def history():
    conn = sqlite3.connect("database.db")
    df = pd.read_sql_query("SELECT * FROM forecasts", conn)
    conn.close()
    return df.to_dict(orient="records")

# ----------- Chronos expanding----------- 
@app.post("/forecast_expanding")
def forecast_expanding(req: ForecastRequest):
    try:
        df = get_dataframe_for_request(req)
        init_train = req.initial_train if req.initial_train is not None else 100

        metrics, forecast_df, _ = run_chronos_expanding_forecast(
            df,
            series_col=None,
            prediction_length=req.prediction_length,
            initial_train=init_train,
            step=req.prediction_length,
            model_name=req.model_name,
            device=None
        )

        if "chronos_expanding" not in forecast_df.columns:
            forecast_df.rename(columns={forecast_df.columns[1]: "chronos_expanding"}, inplace=True)

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        for _, row in forecast_df.iterrows():
            c.execute(
                "INSERT INTO forecasts (model, ds, value) VALUES (?, ?, ?)",
                ("Chronos-expanding", str(row["ds"]), float(row["chronos_expanding"]))
            )
        conn.commit()
        conn.close()

        forecast_df_out = forecast_df.copy()
        forecast_df_out["ds"] = forecast_df_out["ds"].astype(str)

        return {
            "model": "Chronos-expanding",
            "metrics": metrics,
            "predictions": forecast_df_out.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": f"Chronos expanding forecast failed: {e}"}


# ----------- Moirai expanding ----------- 
@app.post("/forecast_moirai_expanding")
def forecast_moirai_expanding(req: ForecastRequest):
    try:
        df = get_dataframe_for_request(req)
        init_train = req.initial_train if req.initial_train is not None else 100

        metrics, forecast_df, meta = run_moirai_expanding_forecast(
            df,
            model_choice=req.model_name,
            size_choice=req.size_choice or "base",
            prediction_length=req.prediction_length,
            initial_train=init_train,
            max_context_length=req.max_context_length or 200,
            batch_size=req.batch_size or 32
        )

        preds_plot = [p.mean(axis=0) for _, p, _ in meta]
        preds_plot = np.concatenate(preds_plot)
        if len(preds_plot) > len(forecast_df):
            preds_plot = preds_plot[:len(forecast_df)]
        elif len(preds_plot) < len(forecast_df):
            preds_plot = np.pad(preds_plot, (0, len(forecast_df)-len(preds_plot)), mode='edge')

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        for ds, val in zip(forecast_df["ds"], preds_plot):
            c.execute(
                "INSERT INTO forecasts (model, ds, value) VALUES (?, ?, ?)",
                ("Moirai-expanding", str(ds), float(val))
            )
        conn.commit()
        conn.close()

        forecast_df["moirai_expanding"] = preds_plot
        forecast_df_out = forecast_df.copy()
        forecast_df_out["ds"] = forecast_df_out["ds"].astype(str)

        return {
            "model": "Moirai-expanding",
            "metrics": metrics,
            "predictions": forecast_df_out.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": f"Moirai expanding forecast failed: {e}"}


# ----------- AutoARIMA----------- 
@app.post("/forecast_autoarima")
def forecast_autoarima(req: ForecastRequest):
    try:
        df = get_dataframe_for_request(req)
        df = df.sort_index()

        series = df.iloc[:, 0].copy()
        series = pd.to_numeric(series, errors="coerce").dropna()
        if len(series) < 12:
            return {"error": f"Series too short for AutoARIMA (need >=12 non-NaN points). Got {len(series)}."}

        try:
            inferred = pd.infer_freq(series.index)
            if inferred is None:
                series = series.resample("W").mean().dropna()
        except Exception:
            pass

        df_pre = series.to_frame(name=df.columns[0])

        try:
            metrics, forecast_df, _ = run_autoarima_forecast_api(
                df_pre,
                series_col=None,
                prediction_length=req.prediction_length,
                test_len=req.test_len,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
            )
        except TypeError:
            metrics, forecast_df, _ = run_autoarima_forecast_api(
                df_pre,
                series_col=None,
                prediction_length=req.prediction_length,
                test_len=req.test_len,
            )

        expected_cols = ["ds", "autoarima", "lower", "upper", "true"]
        for c in expected_cols:
            if c not in forecast_df.columns:
                if c == "ds":
                    forecast_df = forecast_df.reset_index().rename(columns={forecast_df.index.name or 0: "ds"})
                elif c == "lower":
                    forecast_df["lower"] = forecast_df.get("autoarima", np.nan) * 0.9
                elif c == "upper":
                    forecast_df["upper"] = forecast_df.get("autoarima", np.nan) * 1.1
                elif c == "true":
                    forecast_df["true"] = np.nan

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        for _, row in forecast_df.iterrows():
            ds_str = str(row["ds"])
            val = float(row["autoarima"]) if "autoarima" in row and not pd.isna(row["autoarima"]) else None
            if val is not None:
                c.execute("INSERT INTO forecasts (model, ds, value) VALUES (?, ?, ?)",
                          ("AutoARIMA", ds_str, val))
        conn.commit()
        conn.close()

        forecast_df_out = forecast_df.copy()
        forecast_df_out["ds"] = pd.to_datetime(forecast_df_out["ds"]).astype(str)

        return {
            "model": "AutoARIMA",
            "metrics": metrics,
            "predictions": forecast_df_out.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": f"AutoARIMA forecast failed: {e}"}


# ----------- AutoARIMA expanding---------- 
@app.post("/forecast_autoarima_expanding")
def forecast_autoarima_expanding(req: ForecastRequest):
    try:
        df = get_dataframe_for_request(req)
        init_train = req.initial_train if req.initial_train is not None else 100

        metrics, forecast_df, meta = run_autoarima_expanding_forecast(
            df,
            series_col=None,
            prediction_length=req.prediction_length,
            initial_train=init_train,
            step=req.prediction_length,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True
        )

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        for _, row in forecast_df.iterrows():
            c.execute(
                "INSERT INTO forecasts (model, ds, value) VALUES (?, ?, ?)",
                ("AutoARIMA-expanding", str(row["ds"]), float(row["autoarima_expanding"]))
            )
        conn.commit()
        conn.close()

        forecast_df_out = forecast_df.copy()
        forecast_df_out["ds"] = forecast_df_out["ds"].astype(str)

        return {
            "model": "AutoARIMA-expanding",
            "metrics": metrics,
            "predictions": forecast_df_out.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": f"AutoARIMA expanding forecast failed: {e}"}

