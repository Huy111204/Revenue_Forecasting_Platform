import os
import joblib
import pandas as pd
import numpy as np

ARIMA_PATH = os.getenv("ARIMA_PATH", "models/arima_model.pkl")
SARIMA_PATH = os.getenv("SARIMA_PATH", "models/sarima_model.pkl")
SCALE = 10000.0

def load_models():
    """Load ARIMA and SARIMA models once"""
    models = {"arima": None, "sarima": None}
    try:
        if os.path.exists(ARIMA_PATH):
            models["arima"] = joblib.load(ARIMA_PATH)
    except Exception as e:
        print(f"❌ Error loading ARIMA: {e}")

    try:
        if os.path.exists(SARIMA_PATH):
            models["sarima"] = joblib.load(SARIMA_PATH)
    except Exception as e:
        print(f"❌ Error loading SARIMA: {e}")

    return models

def get_forecast_df(model, steps: int = 30, alpha=0.05):
    """Return forecast DataFrame with confidence intervals"""
    try:
        res = model.get_forecast(steps=steps)
        mean = res.predicted_mean * SCALE
        ci = res.conf_int(alpha=alpha) * SCALE
        dates = pd.date_range(pd.Timestamp.today().normalize() + pd.Timedelta(days=1), periods=steps, freq="D")

        df = pd.DataFrame({
            "forecast": mean.values,
            "ci_lower": ci.iloc[:, 0].values,
            "ci_upper": ci.iloc[:, 1].values
        }, index=dates)
        return df
    except Exception:
        # fallback if model does not support get_forecast
        mean = model.forecast(steps=steps) * SCALE
        dates = pd.date_range(pd.Timestamp.today().normalize() + pd.Timedelta(days=1), periods=steps, freq="D")
        return pd.DataFrame({"forecast": mean}, index=dates)
