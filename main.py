import os
import logging
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="House Price Prediction API")

# Global variable for the model
model = None

# 2. Define the input data structure
# Ensure these field names match your training columns exactly!
class HouseData(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: int  # 1 for Yes, 0 for No
    guestroom: int

# 3. Recursive Model Loader
@app.on_event("startup")
def load_model():
    global model
    try:
        # We start searching from the root 'mlruns' folder in the Docker container
        search_root = "/app/mlruns"
        found_model_path = None

        logger.info(f"🔎 Searching for model artifacts in {search_root}...")

        # Walk through all directories to find where the MLmodel file lives
        for root, dirs, files in os.walk(search_root):
            if "MLmodel" in files:
                # We found the folder containing the model metadata
                found_model_path = root
                break

        if found_model_path:
            logger.info(f"🚀 Found model at: {found_model_path}")
            model = mlflow.pyfunc.load_model(found_model_path)
            logger.info("✅ SUCCESS: Model loaded and ready for predictions!")
        else:
            logger.error("❌ CRITICAL: Could not find 'MLmodel' file anywhere in /app/mlruns")
            # Log what we DID find to help debug
            for root, dirs, files in os.walk(search_root):
                 logger.info(f"Directory: {root} | Contains: {dirs} {files}")

    except Exception as e:
        logger.error(f"❌ Error during model startup: {str(e)}")

# 4. API Endpoints
@app.get("/")
def read_root():
    return {"message": "API is running. Go to /docs for Swagger UI."}

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "search_path": "/app/mlruns"
    }

@app.post("/predict")
def predict(data: HouseData):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    
    try:
        # Convert input to DataFrame (required by MLflow)
        input_df = pd.DataFrame([data.dict()])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        return {
            "prediction": float(prediction[0]),
            "currency": "USD"
        }
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
