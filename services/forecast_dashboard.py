import os
import json
import io
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

import streamlit as st
from sqlalchemy import create_engine

# ----------------------------
# Config
# ----------------------------
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "111204")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "retail_db")

ARIMA_PATH = os.getenv("ARIMA_PATH", "models/arima_model.pkl")
SARIMA_PATH = os.getenv("SARIMA_PATH", "models/sarima_model.pkl")
METRICS_PATH = os.getenv("METRICS_PATH", "models/metrics.json")

SCALE = 10000.0  # để đưa về VND

# ----------------------------
# DB engine
# ----------------------------
def get_engine():
    url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, echo=False)

# ----------------------------
# Cached load data
# ----------------------------
@st.cache_data(ttl=600)
def load_sales_table():
    eng = get_engine()
    for tbl in ("daily_revenue_scaled", "daily_revenue"):
        try:
            df = pd.read_sql(f"SELECT date, sales FROM {tbl} ORDER BY date", eng, parse_dates=["date"])
            df.set_index("date", inplace=True)
            st.session_state["sales_table_used"] = tbl
            return df
        except Exception:
            continue
    raise RuntimeError("Không thể tìm bảng sales trong DB. Hãy chạy load_data.py trước.")

# ----------------------------
# Cached load models & metrics
# ----------------------------
@st.cache_resource
def load_model_safe(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Không load được model từ {path}: {e}")
        return None

@st.cache_data
def load_metrics(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# ----------------------------
# Forecast util
# ----------------------------
def get_forecast_df(model, last_date, steps, alpha=0.05):
    try:
        res = model.get_forecast(steps=steps)
        mean = res.predicted_mean
        ci = res.conf_int(alpha=alpha)
        dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "forecast": mean.values,
            "ci_lower": ci.iloc[:, 0].values,
            "ci_upper": ci.iloc[:, 1].values
        })
    except Exception:
        mean = model.forecast(steps=steps)
        dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "forecast": mean,
            "ci_lower": np.nan,
            "ci_upper": np.nan
        })
    df.set_index("date", inplace=True)
    return df

# ----------------------------
# UI layout
# ----------------------------
st.set_page_config(page_title="Sales Forecasting (ARIMA & SARIMA)", layout="wide")
st.title("📊 Sales Forecasting Dashboard — ARIMA & SARIMA")

with st.sidebar:
    st.header("Controls")
    days = st.slider("Days to forecast", min_value=1, max_value=120, value=30)
    show_ci = st.checkbox("Show confidence interval", value=True)
    model_choice = st.selectbox("Model", options=["ARIMA", "SARIMA", "Compare both"], index=2)
    show_raw = st.checkbox("Show raw values (VND)", value=False)
    backtest_days = st.number_input("Backtest horizon (days)", min_value=7, max_value=90, value=30)

# Load everything
try:
    df = load_sales_table()
except Exception as exc:
    st.error(f"Cannot load sales table: {exc}")
    st.stop()

arima_model = load_model_safe(ARIMA_PATH)
sarima_model = load_model_safe(SARIMA_PATH)
metrics = load_metrics(METRICS_PATH)

tab1, tab2, tab3, tab4 = st.tabs(["Data", "Forecast", "Backtest", "Model Card"])

# ---- Tab 1: Data ----
with tab1:
    st.subheader("Historical Sales")
    display_series = df["sales"] * SCALE
    st.line_chart(display_series)

# ---- Tab 2: Forecast ----
with tab2:
    last_date = df.index.max()
    forecasts = {}

    if model_choice in ("ARIMA", "Compare both") and arima_model:
        fc_arima = get_forecast_df(arima_model, last_date, days)
        fc_arima *= SCALE
        forecasts["ARIMA"] = fc_arima

    if model_choice in ("SARIMA", "Compare both") and sarima_model:
        fc_sarima = get_forecast_df(sarima_model, last_date, days)
        fc_sarima *= SCALE
        forecasts["SARIMA"] = fc_sarima

    if forecasts:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, display_series, label="Historical", color="#1f77b4")
        colors = {"ARIMA": "#d62728", "SARIMA": "#ff7f0e"}
        for name, fc in forecasts.items():
            ax.plot(fc.index, fc["forecast"], label=f"{name} Forecast", color=colors.get(name))
            if show_ci and "ci_lower" in fc and "ci_upper" in fc:
                ax.fill_between(fc.index, fc["ci_lower"], fc["ci_upper"], color=colors.get(name), alpha=0.2)
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("No forecast available. Train models first.")

# ---- Tab 3: Backtest ----
with tab3:
    N = int(backtest_days)
    if arima_model or sarima_model:
        train = df.iloc[:-N]
        test = df.iloc[-N:]
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(train.index, train["sales"] * SCALE, label="Train")
        ax2.plot(test.index, test["sales"] * SCALE, label="Test")

        if arima_model:
            fc_test = get_forecast_df(arima_model, train.index.max(), N)
            y_pred = fc_test["forecast"].values * SCALE
            ax2.plot(test.index, y_pred, "--", label="ARIMA Pred")

        if sarima_model:
            fc_test = get_forecast_df(sarima_model, train.index.max(), N)
            y_pred = fc_test["forecast"].values * SCALE
            ax2.plot(test.index, y_pred, "--", label="SARIMA Pred")

        ax2.legend()
        st.pyplot(fig2)
    else:
        st.info("No models for backtest.")

# ---- Tab 4: Model Card ----
with tab4:
    st.json(metrics if metrics else {"info": "No metrics.json found"})
