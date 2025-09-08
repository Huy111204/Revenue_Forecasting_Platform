# -*- coding: utf-8 -*-
import pandas as pd
import joblib
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sqlalchemy import create_engine
import os
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# =====================
# 1. Config PostgreSQL
# =====================
DB_USER = "postgres"
DB_PASS = "111204"
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "retail_db"

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# =====================
# 2. Load dữ liệu
# =====================
query = "SELECT date, sales FROM daily_revenue ORDER BY date;"
df = pd.read_sql(query, engine, parse_dates=["date"], index_col="date")
y = df["sales"]

# =====================
# 3. Train & Rolling Backtest ARIMA(5,1,5)
# =====================
r2_averages, rmse_averages, mae_averages = [], [], []

for i in range(1, 12):  # chia 11 block
    model = ARIMA(y[:30*i], order=(5,1,5))
    model_fit = model.fit()
    
    n_forecast = 30
    forecast_res = model_fit.get_forecast(steps=n_forecast)
    mean_f = forecast_res.predicted_mean
    
    y_true = np.array(y[30*i:30*(i+1)])
    y_pred = np.array(mean_f)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"Section {i}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    if i > 7:  # lấy trung bình từ block thứ 8 trở đi
        r2_averages.append(r2)
        rmse_averages.append(rmse)
        mae_averages.append(mae)

# =====================
# 4. Tính trung bình metrics
# =====================
metrics = {
    "ARIMA(5,1,5)": {
        "R2": float(np.mean(r2_averages)),
        "RMSE": float(np.mean(rmse_averages)),
        "MAE": float(np.mean(mae_averages))
    }
}

print("📊 Final Average Metrics:", metrics)

# =====================
# 5. Train lại full model và lưu
# =====================
final_model = ARIMA(y, order=(5,1,5)).fit()

os.makedirs("../models", exist_ok=True)

joblib.dump(final_model, "../models/arima_515_full.pkl")

with open("../models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Model + metrics đã được lưu.")
