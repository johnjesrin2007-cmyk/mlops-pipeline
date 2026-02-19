import os
import logging
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -------------------------------
# 1. Setup Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="House Price Prediction API")

# -------------------------------
# 2. Enable CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

# -------------------------------
# 3. Data Schema
# -------------------------------
class HouseData(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: int
    guestroom: int


# -------------------------------
# 4. Load Model at Startup
# -------------------------------
@app.on_event("startup")
def load_model():
    global model
    try:
        model_path = os.path.join(os.getcwd(), "model", "model.pkl")

        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Looking for model at: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = joblib.load(model_path)
        logger.info("✅ Model loaded successfully!")

    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        model = None


# -------------------------------
# 5. Routes
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "API is online. Go to /docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: HouseData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check logs.")

    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)

        return {
            "prediction": float(prediction[0]),
            "currency": "USD"
        }

    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
