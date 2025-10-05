# autoarima_forecast.py
"""
Robust AutoARIMA helper for single-fit FastAPI usage.

Behavior:
 - run_autoarima_forecast_api: attempt single-fit. If predictions are
   suspiciously flat, retry with different fit settings. If still flat,
   optionally fall back to expanding-window helper (if available).
 - run_autoarima_forecast: kept for backward compatibility (rolling).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from pmdarima import auto_arima
except Exception as e:
    raise ImportError("pmdarima is required. Install with `pip install pmdarima`") from e

from sklearn.metrics import mean_absolute_error, mean_squared_error

# small helper to infer seasonal period m from a DatetimeIndex
def _infer_seasonal_m(index: pd.DatetimeIndex):
    try:
        freq = pd.infer_freq(index)
    except Exception:
        freq = None
    if freq:
        freq = freq.upper()
        if freq.startswith("W"):
            return 52
        if freq.startswith("D"):
            return 7
        if freq.startswith("M"):
            return 12
        if freq.startswith("Q"):
            return 4
        if freq.startswith("A") or freq.startswith("Y"):
            return 1
    diffs = index.to_series().diff().dropna()
    if len(diffs) == 0:
        return None
    median_delta = diffs.median()
    days = int(median_delta / np.timedelta64(1, "D"))
    if 6 <= days <= 9:
        return 52
    if 27 <= days <= 32:
        return 12
    if days == 1:
        return 7
    return None


def _fit_predict_autoarima(train, test, seasonal, m, stepwise=True, max_p=5, max_q=5, alpha=0.05):
    """Fit auto_arima with given options and return y_pred, conf_int, model_info (or raise)"""
    model = auto_arima(
        train,
        seasonal=seasonal,
        m=(m if seasonal and (m is not None) else 1),
        stepwise=stepwise,
        suppress_warnings=True,
        error_action="ignore",
        max_p=max_p,
        max_q=max_q,
        start_p=0,
        start_q=0,
    )
    out = model.predict(n_periods=len(test), return_conf_int=True, alpha=alpha)
    if isinstance(out, tuple):
        y_pred, conf_int = out
    else:
        y_pred = np.asarray(out, dtype=float)
        conf_int = np.column_stack((y_pred * 0.9, y_pred * 1.1))
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    conf_int = np.asarray(conf_int, dtype=float)
    return y_pred, conf_int, model


def run_autoarima_forecast_api(
    df,
    series_col=None,
    prediction_length: int = 20,
    test_len: int = 50,
    alpha: float = 0.05,
    retry_stepwise_false: bool = True,
    expanding_fallback: bool = True,
):
    """
    Single-fit AutoARIMA for FastAPI.
    Returns: metrics (dict), forecast_df (ds, autoarima, lower, upper, true), meta (None)
    """

    # choose series
    if series_col is None:
        series = df.iloc[:, 0].copy().astype(float)
    else:
        series = df[series_col].copy().astype(float)

    # ensure datetime index & sort
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    series = series.sort_index()

    n = len(series)
    if not (0 < test_len < n):
        raise ValueError("test_len must be > 0 and < len(series)")
    train_len = n - test_len
    train = series.iloc[:train_len]
    test = series.iloc[train_len:]

    # infer seasonal
    m = _infer_seasonal_m(series.index)
    use_seasonal = (m is not None) and (n >= 2 * (m if m > 1 else 2))

    attempts = []
    fallback_used = None
    used_expanding = False
    final_y_pred = None
    final_conf = None
    final_model_info = None

    # first attempt: default (stepwise, seasonal as inferred)
    try:
        y_pred, conf_int, model = _fit_predict_autoarima(train, test, seasonal=use_seasonal, m=m, stepwise=True, alpha=alpha)
        final_y_pred = y_pred
        final_conf = conf_int
        final_model_info = model
        attempts.append({"attempt": 1, "seasonal": bool(use_seasonal), "stepwise": True, "note": "initial"})
    except Exception as e:
        attempts.append({"attempt": 1, "error": str(e)})
        # don't raise yet; we'll fallback below

    # check flatness
    def _is_flat(preds, train):
        preds = np.asarray(preds, dtype=float)
        train_std = float(np.nanstd(train.values.astype(float))) if len(train) > 1 else 0.0
        pred_std = float(np.nanstd(preds))
        # relative threshold: if pred_std < 2% of train std (or an absolute tiny threshold) -> flat
        rel_threshold = 0.02
        abs_threshold = 1e-6
        is_flat = (pred_std < max(abs_threshold, rel_threshold * max(train_std, 1.0)))
        return is_flat, pred_std, train_std

    flat_flag = True
    if final_y_pred is not None:
        flat_flag, pred_std, train_std = _is_flat(final_y_pred, train)
    else:
        pred_std = None
        train_std = None

    # If flat -> try a more aggressive re-fit: stepwise=False and limited max_p/q
    if (final_y_pred is None) or flat_flag:
        try:
            attempts.append({"attempt": 2, "seasonal": False, "stepwise": False, "note": "try stepwise=False, seasonal=False"})
            y_pred2, conf_int2, model2 = _fit_predict_autoarima(train, test, seasonal=False, m=1, stepwise=False, max_p=3, max_q=3, alpha=alpha)
            is_flat2, pred_std2, train_std2 = _is_flat(y_pred2, train)
            if (not is_flat2) and (final_y_pred is None or pred_std2 > (pred_std or 0.0)):
                final_y_pred = y_pred2
                final_conf = conf_int2
                final_model_info = model2
                flat_flag = False
                pred_std = pred_std2
            else:
                # keep the best so far (if any)
                if final_y_pred is None:
                    final_y_pred = y_pred2
                    final_conf = conf_int2
                    final_model_info = model2
                flat_flag = True
        except Exception as e:
            attempts.append({"attempt": 2, "error": str(e)})

    # If still flat & user allows expanding fallback, try expanding-window helper
    if flat_flag and expanding_fallback:
        try:
            # local import to avoid mandatory dependency if the file doesn't exist
            from autoarima_expanding import run_autoarima_expanding_forecast

            # choose an initial_train reasonably
            initial_train = max(20, int(train_len / 2))
            m_exp = max(1, initial_train)  # dummy so signature happy

            metrics_exp, forecast_df_exp, meta = run_autoarima_expanding_forecast(
                pd.DataFrame(series),
                series_col=series.name if hasattr(series, "name") else None,
                prediction_length=prediction_length,
                initial_train=initial_train,
                step=prediction_length,
            )

            # forecast_df_exp has columns ["ds", "autoarima_expanding", "true"]
            # convert to single forecast values aligned to test set
            # We'll aggregate the expanding predictions by taking the sequence in forecast_df_exp
            y_pred_exp = forecast_df_exp.iloc[:, 1].values.astype(float)
            ds_exp = pd.to_datetime(forecast_df_exp["ds"])
            conf_lower = y_pred_exp * 0.9
            conf_upper = y_pred_exp * 1.1

            final_y_pred = np.asarray(y_pred_exp, dtype=float)
            final_conf = np.column_stack((conf_lower, conf_upper))
            final_model_info = None
            used_expanding = True
            fallback_used = "expanding_fallback"
            attempts.append({"attempt": "expanding_fallback", "note": f"used expanding with init_train={initial_train}"})
        except Exception as e:
            attempts.append({"attempt": "expanding_fallback", "error": str(e)})
            # if expanding not available, we'll fall back to last-value fallback below

    # final fallback: if still no final_y_pred, or somehow None -> use last-value
    if final_y_pred is None:
        last = float(train.iloc[-1]) if len(train) > 0 else 0.0
        final_y_pred = np.full(len(test), last, dtype=float)
        final_conf = np.column_stack((final_y_pred * 0.9, final_y_pred * 1.1))
        fallback_used = "last_value_fallback"
        attempts.append({"attempt": "last_value_fallback", "note": "train empty or fits failed"})

    # build forecast_df
    forecast_df = pd.DataFrame({
        "ds": test.index,
        "autoarima": np.asarray(final_y_pred, dtype=float),
        "lower": final_conf[:, 0].astype(float),
        "upper": final_conf[:, 1].astype(float),
        "true": test.values.astype(float)
    })

    # compute metrics (robust)
    y_true = np.asarray(forecast_df["true"], dtype=float)
    y_pred_arr = np.asarray(forecast_df["autoarima"], dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred_arr))
    mse = float(mean_squared_error(y_true, y_pred_arr))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred_arr) / (y_true + 1e-8))) * 100.0)
    smape = float(100.0 * np.mean(2.0 * np.abs(y_pred_arr - y_true) / (np.abs(y_pred_arr) + np.abs(y_true) + 1e-8)))

    pred_std_final = float(np.nanstd(y_pred_arr))

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "sMAPE": smape,
        "model_order": str(getattr(final_model_info, "order", None)),
        "seasonal_order": str(getattr(final_model_info, "seasonal_order", None)),
        "train_len": int(train_len),
        "test_len": int(test_len),
        "pred_std": pred_std_final,
        "pred_min": float(np.nanmin(y_pred_arr)),
        "pred_max": float(np.nanmax(y_pred_arr)),
        "attempts": attempts,
        "fallback_used": fallback_used,
        "used_expanding": bool(used_expanding),
    }

    return metrics, forecast_df, None


# ---- keep original rolling helper----
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
    if series_col is None:
        series = df.iloc[:, 0].copy()
        colname = df.columns[0]
    else:
        series = df[series_col].copy()
        colname = series_col

    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)
    series = series.sort_index()

    n = len(series)
    if not (0 < test_len < n):
        raise ValueError("test_len must be >0 and < len(series)")

    train_len = n - test_len

    preds = []
    for start in range(0, test_len, prediction_length):
        train_end = train_len + start
        train_slice = series.iloc[:train_end]

        model = auto_arima(
            train_slice,
            seasonal=seasonal,
            stepwise=stepwise,
            suppress_warnings=suppress_warnings,
            error_action=error_action,
        )

        n_periods = min(prediction_length, test_len - start)
        out = model.predict(n_periods=n_periods)
        preds.extend(np.asarray(out, dtype=float).reshape(-1).tolist())

    forecast_vals = np.array(preds[:test_len], dtype=float)
    forecast_index = series.index[-test_len:]
    forecast_series = pd.Series(forecast_vals, index=forecast_index)

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
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "ts_wide.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    metrics, forecast_df, _ = run_autoarima_forecast_api(df, prediction_length=20, test_len=10)
    print("metrics:", metrics)
    print(forecast_df.tail())


