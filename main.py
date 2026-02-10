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
    # Check if we are on Render (which uses the PORT env var)
    if os.getenv("RENDER"):
        logger.info("Running on Render - checking MLflow connectivity...")
    
    try:
        model_name = "HousePriceModel"
        stage = "Production"
        model_uri = f"models:/{model_name}/{stage}"
        
        # This is where it usually freezes if the URI is wrong
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ Success!")
    except Exception as e:
        logger.error(f"❌ Startup Error: {e}")
        # We don't raise the error, so the server stays alive
        # allowing you to at least see the health page/logs.
        model = None

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
