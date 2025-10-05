# chronos_model.py
"""
Robust Chronos helper module.

Functions:
- load_chronos_model(model_name, device=None)
- predict_one_window(tokenizer, model, device, history, pred_len)
- expanding_window_forecast(df, prediction_length, initial_train, step, model_name, device=None)

Input df: pandas.DataFrame with index=date and first column the target (like 'date,value' csv).
"""
import re
import math
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import mean_absolute_error, mean_squared_error

# regex to extract floats from model text output
FLOAT_RE = re.compile(r'[-+]?\d*\.\d+|[-+]?\d+')


def _ensure_sentencepiece_installed():
    try:
        import sentencepiece  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "You must install 'sentencepiece' to load certain Chronos tokenizers.\n"
            "Please run: pip install sentencepiece\n"
            "It is also recommended to install: pip install protobuf\n"
        ) from e


def _coerce_model_name(model_name) -> str:
    """
    Ensure model_name is a usable string (repo id or local path).
    If it's not a str, try to coerce via str(...) and warn.
    """
    if isinstance(model_name, str):
        return model_name
    try:
        m = str(model_name)
        # basic sanity: must contain at least one slash or be non-empty
        if not m:
            raise ValueError("Model name cannot be an empty string.")
        return m
    except Exception as e:
        raise TypeError(f"model_name must be a string or convertible to string, got type: {type(model_name)}") from e


def load_chronos_model(model_name: str, device: Optional[str] = None):
    """
    Load tokenizer & model from Hugging Face with robust fallbacks.
    Returns: tokenizer, model, device
    """
    # coerce model_name to string (handle Path objects etc.)
    model_name = _coerce_model_name(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[chronos_model] Loading model '{model_name}' on device '{device}'")

    # ensure sentencepiece is present (some chronos tokenizers need it)
    try:
        _ensure_sentencepiece_installed()
    except RuntimeError as e:
        # Let user know but continue to try (auto errors will surface)
        print("[chronos_model] Warning:", e)

    # Try loading tokenizer with a few fallbacks
    tokenizer = None
    tokenizer_error = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    except Exception as e:
        tokenizer_error = e
        print(f"[chronos_model] AutoTokenizer failed: {repr(e)}; trying fallback T5Tokenizer...")

    if tokenizer is None:
        try:
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False)
        except Exception as e2:
            print(f"[chronos_model] T5Tokenizer fallback failed: {repr(e2)}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except Exception as e3:
                raise RuntimeError(
                    f"Failed to load tokenizer for '{model_name}'.\n"
                    f"Primary error: {tokenizer_error}\n"
                    f"Fallback1 (T5Tokenizer) error: {e2}\n"
                    f"Fallback2 error: {e3}\n\n"
                    "Common issues and fixes:\n"
                    "- Ensure model_name is a string, e.g., 'amazon/chronos-t5-small' or a valid local directory path.\n"
                    "- If the model uses SentencePiece, run: pip install sentencepiece protobuf\n"
                    "- If you have network or permission issues, download the model locally and set model_name to that path.\n"
                ) from e3

    # Load model (seq2seq) - use trust_remote_code True for custom architectures if needed
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e

    return tokenizer, model, device


def predict_one_window(tokenizer, model, device, history: np.ndarray, pred_len: int, max_input_tokens: Optional[int] = None) -> np.ndarray:
    """
    Produce pred_len forecasts from numeric history using the seq2seq Chronos model.
    Returns numpy array of length pred_len.
    """
    hist = np.asarray(history, dtype=float)
    pred_len = int(pred_len)

    # keep only most recent N numbers to avoid tokenization truncation
    MAX_NUMS = 512
    if len(hist) > MAX_NUMS:
        hist = hist[-MAX_NUMS:]

    # stringify history
    input_str = " ".join([str(float(x)) for x in hist])

    # determine max tokens
    if max_input_tokens is None:
        try:
            max_input_tokens = tokenizer.model_max_length
        except Exception:
            max_input_tokens = 1024

    # tokenize; do NOT call .to(device) on the whole dict
    inputs = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # attempt generation (use max_new_tokens if supported)
    try:
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 max_new_tokens=max(32, pred_len * 2), do_sample=False)
    except TypeError:
        try:
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                     max_length=input_ids.shape[1] + pred_len * 2, do_sample=False)
        except Exception:
            outputs = model.generate(input_ids=input_ids, do_sample=False)

    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # parse floats from text
    parts = FLOAT_RE.findall(pred_text.replace(",", " "))
    preds = []
    for p in parts[:pred_len]:
        try:
            preds.append(float(p))
        except Exception:
            preds.append(0.0)

    # pad to pred_len
    if len(preds) < pred_len:
        pad_val = preds[-1] if preds else (float(hist[-1]) if hist.size else 0.0)
        preds += [pad_val] * (pred_len - len(preds))

    return np.array(preds, dtype=float)


def expanding_window_forecast(
    df: pd.DataFrame,
    prediction_length: int,
    initial_train: int,
    step: int,
    model_name,
    device: Optional[str] = None
) -> Tuple[dict, pd.DataFrame, List[Tuple[int, List[float], List[float]]]]:
    """
    Run expanding-window evaluation.

    Returns:
      - metrics: dict (MAE, MSE, RMSE, MAPE(%))
      - out_df: DataFrame with columns ['y_true','y_pred']
      - meta: list of (window_end_index, pred_list, true_list)
    """
    # basic checks
    if df.shape[1] < 1:
        raise ValueError("Input DataFrame must contain at least one target column.")

    series = df.iloc[:, 0].astype(float).values
    n = len(series)
    if not isinstance(prediction_length, int) or prediction_length <= 0:
        raise ValueError("prediction_length must be a positive integer.")
    if not isinstance(initial_train, int) or initial_train <= 0:
        raise ValueError("initial_train must be a positive integer.")
    if not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer.")

    if initial_train >= n - prediction_length:
        raise ValueError("initial_train too large — cannot produce at least one forecast window.")

    # coerce model_name to str (help avoid 'not a string' errors)
    model_name = _coerce_model_name(model_name)

    # load tokenizer & model
    tokenizer, model, device = load_chronos_model(model_name, device=device)

    windows = list(range(initial_train, n - prediction_length + 1, step))
    preds_all = []
    trues_all = []
    meta = []

    for i, wend in enumerate(windows):
        history = series[:wend]
        pred = predict_one_window(tokenizer, model, device, history, prediction_length)
        true_slice = series[wend: wend + prediction_length]

        preds_all.extend(pred.tolist())
        trues_all.extend(true_slice.tolist())
        meta.append((wend, pred.tolist(), true_slice.tolist()))
        print(f"[chronos_model] window {i+1}/{len(windows)} done (history_len={wend})")

    preds_arr = np.array(preds_all, dtype=float)
    trues_arr = np.array(trues_all, dtype=float)

    mae = float(mean_absolute_error(trues_arr, preds_arr))
    mse = float(mean_squared_error(trues_arr, preds_arr))
    rmse = float(math.sqrt(mse))
    mape = float(np.mean(np.abs((trues_arr - preds_arr) / (np.where(trues_arr == 0, 1e-8, trues_arr)))) * 100.0)

    metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE(%)": mape}
    out_df = pd.DataFrame({"y_true": trues_arr, "y_pred": preds_arr})

    return metrics, out_df, meta


# CLI quick test
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="baidu_ts.csv", help="input CSV (date,value)")
    parser.add_argument("--model", default="amazon/chronos-t5-small", help="huggingface model name or local path")
    parser.add_argument("--pred_len", type=int, default=20)
    parser.add_argument("--init_train", type=int, default=200)
    parser.add_argument("--step", type=int, default=20)
    args = parser.parse_args()

    print("DEBUG: model arg type:", type(args.model), "value:", args.model)
    df = pd.read_csv(args.csv, parse_dates=["date"]).set_index("date")
    metrics, out_df, meta = expanding_window_forecast(df, args.pred_len, args.init_train, args.step, args.model)
    print("Metrics:", metrics)
    print(out_df.head())





