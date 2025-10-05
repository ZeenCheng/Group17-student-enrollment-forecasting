# 🎓 Student Enrollment Forecasting (Moirai, Chronos, Auto Models & Streamlit)

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
| **Chronos** | Transformer-based forecasting model | `Chronos.py` |
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
```
This installs all standard packages including `FastAPI`, `Streamlit`, `Pandas`, `Torch`, and others.

---

### 3️⃣ Install **Uni2TS (Moirai)**
Used for Moirai and Moirai-MoE models (`sample_moirai.py`, `streamlit_app.py`):
```bash
pip install git+https://github.com/microsoft/uni2ts.git
```
If PyTorch compatibility issues occur, upgrade:
```bash
pip install torch --upgrade
```

---

### 4️⃣ Install **Chronos**
Used for Chronos and Chronos Expanding models:
```bash
pip install git+https://github.com/microsoft/Chronos.git
```
If installation is slow (e.g., in China):
```bash
git clone https://ghproxy.com/https://github.com/microsoft/Chronos.git
pip install ./Chronos
```

---

### 5️⃣ Verify Installation
```bash
python -c "import torch, pandas, streamlit, gluonts; print('✅ All core modules loaded successfully!')"
```

---

### 6️⃣ Run the Project
#### Run FastAPI backend
```bash
python main.py
```
#### Or run Streamlit frontend
```bash
streamlit run streamlit_app.py
```
Then open [http://localhost:8501/](http://localhost:8501/) in your browser.  
You can upload `ts_wide.csv` or `baidu_ts.csv` to start forecasting.

---

### 💡 Optional: GPU Acceleration
If your system supports CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 📂 Example Datasets

| File | Source | Description |
|------|---------|-------------|
| `ts_wide.csv` | Google Trends | Time series of enrollment-related keywords |
| `baidu_ts.csv` | Baidu Index | Parallel dataset for China-based trend analysis |

Both are in **wide-format** (each column = topic/keyword, each row = time point).

---

## 👨‍💻 Author

**Zeen Cheng**  
Massey University  
Email: 2638164080@qq.com  
GitHub: [@ZeenCheng](https://github.com/ZeenCheng)

---

✅ *This README has been updated to reflect the final integrated version of the Student Enrollment Forecasting project — combining Moirai, Chronos, AutoARIMA, database backend, and Streamlit interface.*
