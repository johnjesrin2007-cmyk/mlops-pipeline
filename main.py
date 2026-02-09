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
        # 1. Let's see EXACTLY where we are and what is around us
        cwd = os.getcwd()
        logger.info(f"📍 Current Working Directory: {cwd}")
        logger.info(f"📂 Folders in {cwd}: {os.listdir(cwd)}")

        # 2. Check if mlruns exists at all
        search_root = os.path.join(cwd, "mlruns")
        if not os.path.exists(search_root):
            logger.error(f"❌ CRITICAL: The folder 'mlruns' does not exist in {cwd}")
            return

        # 3. Walk and find MLmodel
        found_model_path = None
        for root, dirs, files in os.walk(search_root):
            if "MLmodel" in files:
                found_model_path = root
                break

        if found_model_path:
            logger.info(f"🚀 Found model at: {found_model_path}")
            model = mlflow.pyfunc.load_model(found_model_path)
            logger.info("✅ SUCCESS: Model loaded!")
        else:
            logger.error(f"❌ Could not find 'MLmodel' file inside {search_root}")
            # This will show us the structure in the logs so we can fix the path
            for root, dirs, files in os.walk(cwd):
                 logger.info(f"Found Path: {root}")

    except Exception as e:
        logger.error(f"❌ Load Error: {str(e)}")

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
