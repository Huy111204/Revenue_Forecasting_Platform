from fastapi import APIRouter, Query
from pydantic import BaseModel
import pandas as pd
from datetime import timedelta
from api.utils import get_forecast_df, load_models

router = APIRouter()

# Load models once at startup
models = load_models()

class ForecastResponse(BaseModel):
    model: str
    steps: int
    forecast: list
    ci_lower: list | None = None
    ci_upper: list | None = None
    dates: list

@router.get("/arima", response_model=ForecastResponse)
def forecast_arima(steps: int = Query(30, ge=1, le=120)):
    if models["arima"] is None:
        return {"error": "ARIMA model not available"}

    df = get_forecast_df(models["arima"], steps=steps)
    return ForecastResponse(
        model="ARIMA",
        steps=steps,
        forecast=df["forecast"].tolist(),
        ci_lower=df["ci_lower"].tolist() if "ci_lower" in df else None,
        ci_upper=df["ci_upper"].tolist() if "ci_upper" in df else None,
        dates=df.index.strftime("%Y-%m-%d").tolist()
    )

@router.get("/sarima", response_model=ForecastResponse)
def forecast_sarima(steps: int = Query(30, ge=1, le=120)):
    if models["sarima"] is None:
        return {"error": "SARIMA model not available"}

    df = get_forecast_df(models["sarima"], steps=steps)
    return ForecastResponse(
        model="SARIMA",
        steps=steps,
        forecast=df["forecast"].tolist(),
        ci_lower=df["ci_lower"].tolist() if "ci_lower" in df else None,
        ci_upper=df["ci_upper"].tolist() if "ci_upper" in df else None,
        dates=df.index.strftime("%Y-%m-%d").tolist()
    )
