# Revenue Forecasting Platform

Containerized revenue forecasting platform built with FastAPI, Streamlit, PostgreSQL, and Kubernetes .  
The platform forecasts daily revenue using ARIMA & SARIMA models, provides a REST API for programmatic access, and a Streamlit dashboard for visualization.

---

## Features

- Forecast daily revenue using ARIMA & SARIMA models.
- Interactive Streamlit dashboard for exploring forecasts.
- FastAPI REST API with OpenAPI/Swagger documentation.
- ETL job to process and load historical data into the database.
- Fully containerized and deployable on Kubernetes.
- Persistent database storage via PVC.

---
<img width="1905" height="973" alt="image" src="https://github.com/user-attachments/assets/b0c245ee-33c4-4656-9f95-e0a6602d255b" />
<img width="760" height="407" alt="image" src="https://github.com/user-attachments/assets/0d5baf15-f99e-4199-ae70-670c17854d15" />
<img width="740" height="273" alt="image" src="https://github.com/user-attachments/assets/7a62853c-879e-4388-ae7b-cb9e24d5adeb" />
<img width="731" height="354" alt="image" src="https://github.com/user-attachments/assets/59558817-ea39-4a17-8dd3-ff05d1a20ec0" />
<img width="737" height="371" alt="image" src="https://github.com/user-attachments/assets/94caa49b-857e-4306-ba45-dc2ea0261f4b" />
## To use
```bash
# set up
kubectl create namespace revenue-forecast && \
kubectl create configmap retail-config --from-env-file=.env -n revenue-forecast && \
kubectl create secret generic retail-secret --from-env-file=.env -n revenue-forecast && \
minikube image build -t fastapi-app:latest -f Dockerfile.fastapi . && \
minikube image build -t streamlit-app:latest -f Dockerfile.streamlit . && \
kubectl apply -f k8s/postgres-pvc.yaml && \
kubectl apply -f k8s/postgres-deployment.yaml -n revenue-forecast && \
kubectl apply -f k8s/fastapi-deployment.yaml -n revenue-forecast && \
kubectl apply -f k8s/streamlit-deployment.yaml -n revenue-forecast && \
kubectl apply -f k8s/etl-job.yaml -n revenue-forecast && \
kubectl get pods -n revenue-forecast
# Clean up
kubectl delete namespace revenue-forecast
minikube stop
