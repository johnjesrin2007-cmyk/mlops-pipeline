import os
import mlflow
import logging
from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:/app/mlruns"))

        model_name = os.getenv("MODEL_NAME", "HousePriceModel")
        model_alias = os.getenv("MODEL_ALIAS", "production")

        model_uri = f"models:/{model_name}@{model_alias}"
        logger.info(f"Loading model from {model_uri}")

        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Model loaded successfully")

    except Exception as e:
        logger.error(f"⚠️ Model not loaded yet: {e}")
        model = None   # IMPORTANT: don't crash app
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }
