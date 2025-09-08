from fastapi import FastAPI
from api.routers import forecast

app = FastAPI(
    title="Sales Forecasting API",
    version="1.0.0",
    description="Forecast daily revenue using ARIMA & SARIMA"
)

# Register routers
app.include_router(forecast.router, prefix="/api/v1/forecast", tags=["Forecast"])

@app.get("/")
def root():
    return {"message": "Welcome to Sales Forecasting API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
