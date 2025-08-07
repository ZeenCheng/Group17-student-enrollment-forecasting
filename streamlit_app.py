import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

# Set Streamlit page configuration
st.set_page_config(page_title="Moirai Forecast App", layout="wide")

st.title("Moirai Time Series Forecasting")

# Sidebar: Model & Data Options

st.sidebar.title("Configuration")

# Model selection
model_choice = st.sidebar.selectbox("Select Moirai Model", ["moirai", "moirai-moe"])
size_choice = st.sidebar.selectbox("Model Size", ["small", "base", "large"])
prediction_length = st.sidebar.slider("Prediction Length", 10, 50, 20)
context_length = st.sidebar.slider("Context Length", 50, 500, 200)
batch_size = st.sidebar.slider("Batch Size", 1, 64, 32)
test_len = st.sidebar.slider("Test Set Length", 50, 200, 100)

# Data source selection
data_choice = st.sidebar.selectbox("Select Data Source", ["Google Trends", "Baidu Trends"])

# Load data based on selection

st.write("### Loading time series data...")

def load_data(data_choice):
    if data_choice == "Google Trends":
        file_path = "ts_wide.csv"  # Replace with actual path to Google Trends data
    elif data_choice == "Baidu Trends":
        file_path = "baidu_ts.csv"  
    else:
        st.error("Unknown data source selected.")
        return None
    return pd.read_csv(file_path, index_col=0, parse_dates=True)

df = load_data(data_choice)
if df is None or df.empty:
    st.error("Failed to load data. Please check your file.")
    st.stop()

st.dataframe(df.head())

# GluonTS Dataset preparation

ds = PandasDataset(dict(df))
train, test_template = split(ds, offset=-test_len)

test_data = test_template.generate_instances(
    prediction_length=prediction_length,
    windows=test_len // prediction_length,
    distance=prediction_length,
)

# Load selected model

st.write("### Loading model...")
with st.spinner("Downloading and initializing model..."):
    if model_choice == "moirai":
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size_choice}"),
            prediction_length=prediction_length,
            context_length=context_length,
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
            context_length=context_length,
            patch_size=16,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )

# Create predictor and generate forecasts
predictor = model.create_predictor(batch_size=batch_size)
forecasts = predictor.predict(test_data.input)

# Plot one forecast result

input_it = iter(test_data.input)
label_it = iter(test_data.label)
forecast_it = iter(forecasts)

inp = next(input_it)
label = next(label_it)
forecast = next(forecast_it)

fig, ax = plt.subplots(figsize=(10, 5))
plot_single(
    inp,
    label,
    forecast,
    context_length=context_length,
    name="forecast",
    show_label=True,
    ax=ax,
)

st.subheader("Forecast Plot")
st.pyplot(fig)
