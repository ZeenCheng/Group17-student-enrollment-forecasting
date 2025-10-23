# Student Enrollment Forecasting (Moirai, Chronos, Auto Models & Streamlit)

This project forecasts **student enrollment trends** using **state-of-the-art time series forecasting models** integrated into an interactive web application.  
It combines **Moirai**, **Chronos**, and **Auto ARIMA** (with expanding-window evaluations) to model complex multi-source time series data such as **Google Trends** and **Baidu Trends**.  
The system includes both a **FastAPI backend with SQLite database** and a **Streamlit-based frontend dashboard** for data upload, visualization, and prediction.

---

## 🌟 Features

- **Moirai Forecasting** – Large-scale foundation model for time series prediction.  
- **Chronos & Chronos Expanding** – Transformer-based time series forecasting with expanding-window evaluation.  
- **Auto ARIMA & Auto ARIMA Expanding** – Classical statistical baselines for comparison.  
- **Streamlit Web Interface** – Upload datasets (Google Trends, Baidu Trends, or others) and visualize model outputs interactively.  
- **FastAPI Backend (main.py)** – Connects the database and model layer.  
- **Multi-source Support** – Integrates both `ts_wide.csv` (Google Trends) and `baidu_ts.csv` (Baidu Trends).

---

## 📁 Project Structure

```
├── main.py                  # FastAPI backend + database integration
├── streamlit_app.py         # Streamlit dashboard for user interaction
├── sample_moirai.py         # Moirai forecasting (base + expanding window)
├── autoarima_forecast.py    # Auto ARIMA forecasting
├── autoarima_expanding.py   # Auto ARIMA expanding-window forecasting
├── Chronos.py               # Chronos forecasting (standard)
├── chronos_model.py         # Chronos forecasting (standard)
├── Chronos_expanding.py     # Chronos expanding-window forecasting
├── ts_wide.csv              # Google Trends dataset
├── baidu_ts.csv             # Baidu Trends dataset
├── images/                  # Saved forecast plots
├── requirements.txt         # Dependency list
└── README.md                # Project documentation
```

---

## 🧠 Model Overview

| Model | Description | File |
|-------|--------------|------|
| **Moirai** | Foundation model for time series (from Microsoft Uni2TS) | `sample_moirai.py` |
| **chronos** | Transformer-based forecasting model | `chronos_model.py` |
| **Chronos Expanding** | Expanding-window evaluation of Chronos | `Chronos_expanding.py` |
| **Auto ARIMA** | Classical statistical forecasting model | `autoarima_forecast.py` |
| **Auto ARIMA Expanding** | Expanding-window evaluation version | `autoarima_expanding.py` |
| **Backend + Database** | FastAPI + SQLite integration | `main.py` |
| **Frontend UI** | Streamlit dashboard | `streamlit_app.py` |

---

## 📊 Example Forecasting Output

![Forecast Output 1](images/forecast_plot_3.png)  
![Forecast Output 2](images/forecast_plot_4.png)  
![Forecast Output 3](images/forecast_plot_5.png)

---

## ⚙️ Requirements

- Python 3.10–3.13  
- Conda (recommended)  
- Moirai (via Uni2TS)  
- Chronos  
- Pandas, NumPy, Scikit-learn  
- Streamlit  
- Matplotlib, Plotly  
- FastAPI, Torch

---

## 🧩 Environment Setup

### 1️⃣ Create Conda Environment
```bash
conda create -n enrollment_forecast python=3.10 -y
conda activate enrollment_forecast
```
> Python 3.10 is the most stable for Moirai (Uni2TS) and Chronos compatibility.

---

### 2️⃣ Install Core Dependencies
```bash
pip install -r requirements.txt
pip install fastapi[all]
pip install pmdarima
pip install datasets einops gluonts matplotlib scikit-learn huggingface-hub
pip install "gluonts[torch]~=0.14"
```
This installs all standard packages including `FastAPI`, `Streamlit`, `Pandas`, `Torch`, and others.

---

### 3️⃣ Install **Torch**
```bash
pip install torch --upgrade
pip install streamlit pandas numpy matplotlib scikit-learn transformers torch
```

---

### 4️⃣ Install **Chronos**
Used for Chronos and Chronos Expanding models:
```bash
pip install chronos-forecasting pandas numpy scikit-learn
```

---


---

### 6️⃣ Run the Project
#### Run FastAPI backend
```bash
pip install uvicorn
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
#### then run Streamlit frontend(using another anaconda_prompt)
```bash
streamlit run streamlit_app.py
```
[http://localhost:8501/](http://localhost:8501/).  
You can upload `ts_wide.csv` , `baidu_ts.csv` or your local data(It must be formatted, you can check the ts_wide.csv given to make sure your data is ready to run) to start forecasting.

---



---

## 📂 Example Datasets

| File | Source | Description |
|------|---------|-------------|
| `ts_wide.csv` | Google Trends | Time series of enrollment-related keywords |
| `baidu_ts.csv` | Baidu Index | Parallel dataset for China-based trend analysis |

Both are in **wide-format** (each column = topic/keyword, each row = time point).

---

##  Author

**Zeen Cheng**  
Massey University  
Email: 2638164080@qq.com  
GitHub: [@ZeenCheng](https://github.com/ZeenCheng)

---

 *This README has been updated to reflect the final integrated version of the Student Enrollment Forecasting project — combining Moirai, Chronos, AutoARIMA, database backend, and Streamlit interface.*
