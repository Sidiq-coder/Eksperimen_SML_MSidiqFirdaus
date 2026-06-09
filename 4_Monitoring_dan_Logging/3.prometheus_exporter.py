import json
import os
import time
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "2_Membangun_Model", "artifacts", "titanic_model.pkl")
FEATURE_NAMES_PATH = os.path.join(PROJECT_ROOT, "2_Membangun_Model", "artifacts", "feature_names.json")

app = FastAPI(title="Titanic ML Monitoring API")
start_time = time.time()

# 10+ metrics untuk target Advance
INFERENCE_REQUEST_TOTAL = Counter("inference_request_total", "Total request inference")
INFERENCE_ERROR_TOTAL = Counter("inference_error_total", "Total error inference")
INFERENCE_LATENCY_SECONDS = Histogram("inference_latency_seconds", "Latency inference dalam detik")
INFERENCE_PREDICTION_TOTAL = Counter("inference_prediction_total", "Total prediksi berhasil")
INFERENCE_PREDICTION_BY_CLASS_TOTAL = Counter(
    "inference_prediction_by_class_total", "Total prediksi berdasarkan kelas", ["class_name"]
)
MODEL_UPTIME_SECONDS = Gauge("model_uptime_seconds", "Waktu hidup model serving dalam detik")
MODEL_LOADED_STATUS = Gauge("model_loaded_status", "Status model berhasil dimuat: 1 berhasil, 0 gagal")
CPU_USAGE_PERCENT = Gauge("cpu_usage_percent", "Persentase penggunaan CPU")
MEMORY_USAGE_PERCENT = Gauge("memory_usage_percent", "Persentase penggunaan memory")
INPUT_FEATURE_COUNT = Gauge("input_feature_count", "Jumlah fitur input yang diterima")
INVALID_INPUT_TOTAL = Counter("invalid_input_total", "Jumlah input tidak valid")
PREDICTION_CONFIDENCE_AVERAGE = Gauge("prediction_confidence_average", "Confidence prediksi terakhir")


def load_model_and_features():
    if not os.path.exists(MODEL_PATH):
        MODEL_LOADED_STATUS.set(0)
        raise FileNotFoundError(
            f"Model tidak ditemukan di {MODEL_PATH}. Jalankan modelling_tuning.py terlebih dahulu."
        )
    if not os.path.exists(FEATURE_NAMES_PATH):
        MODEL_LOADED_STATUS.set(0)
        raise FileNotFoundError(
            f"Feature names tidak ditemukan di {FEATURE_NAMES_PATH}. Jalankan modelling_tuning.py terlebih dahulu."
        )

    model = joblib.load(MODEL_PATH)
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    MODEL_LOADED_STATUS.set(1)
    return model, feature_names


try:
    MODEL, FEATURE_NAMES = load_model_and_features()
except Exception as exc:
    print(f"Warning: {exc}")
    MODEL, FEATURE_NAMES = None, []


@app.get("/")
def root():
    return {
        "message": "Titanic ML Monitoring API aktif",
        "metrics_endpoint": "/metrics",
        "predict_endpoint": "/predict",
        "feature_count": len(FEATURE_NAMES),
    }


@app.get("/features")
def features():
    return {"features": FEATURE_NAMES}


@app.post("/predict")
def predict(payload: Dict[str, float]):
    INFERENCE_REQUEST_TOTAL.inc()
    start = time.time()

    try:
        if MODEL is None:
            INFERENCE_ERROR_TOTAL.inc()
            raise HTTPException(status_code=500, detail="Model belum berhasil dimuat.")

        missing_features = [feature for feature in FEATURE_NAMES if feature not in payload]
        if missing_features:
            INVALID_INPUT_TOTAL.inc()
            INFERENCE_ERROR_TOTAL.inc()
            raise HTTPException(
                status_code=400,
                detail=f"Input tidak lengkap. Fitur yang kurang: {missing_features}",
            )

        row = pd.DataFrame([[payload[feature] for feature in FEATURE_NAMES]], columns=FEATURE_NAMES)
        prediction = int(MODEL.predict(row)[0])

        confidence = None
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(row)[0]
            confidence = float(np.max(proba))
            PREDICTION_CONFIDENCE_AVERAGE.set(confidence)

        class_name = "survived" if prediction == 1 else "not_survived"
        INFERENCE_PREDICTION_TOTAL.inc()
        INFERENCE_PREDICTION_BY_CLASS_TOTAL.labels(class_name=class_name).inc()
        INPUT_FEATURE_COUNT.set(len(payload))

        return {
            "prediction": prediction,
            "class_name": class_name,
            "confidence": confidence,
        }

    except HTTPException:
        raise
    except Exception as exc:
        INFERENCE_ERROR_TOTAL.inc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        latency = time.time() - start
        INFERENCE_LATENCY_SECONDS.observe(latency)
        MODEL_UPTIME_SECONDS.set(time.time() - start_time)
        CPU_USAGE_PERCENT.set(psutil.cpu_percent(interval=None))
        MEMORY_USAGE_PERCENT.set(psutil.virtual_memory().percent)


metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
