from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd
import os
import logging

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="House Price Prediction API")

# Global variable to store the model
model = None

# 2. Define the input data structure
# IMPORTANT: Change these names to match your CSV/Training columns!
class HouseData(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: int  # 1 for yes, 0 for no
    guestroom: int

# 3. Startup Logic: Auto-detect and Load Model
@app.on_event("startup")
def load_model():
    global model
    try:
        base_path = "/app/mlruns/0"
        
        if not os.path.exists(base_path):
            logger.error(f"❌ Path not found: {base_path}")
            return

        # Find the folder containing 'artifacts'
        subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        run_folder = None
        for folder in subfolders:
            if os.path.exists(os.path.join(base_path, folder, "artifacts", "model")):
                run_folder = folder
                break
        
        if not run_folder:
            logger.error("❌ No artifact folder found in mlruns/0")
            return

        model_uri = f"{base_path}/{run_folder}/artifacts/model"
        logger.info(f"🚀 Loading model from: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ SUCCESS: Model loaded and ready for predictions!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")

# 4. Endpoints
@app.get("/")
def read_root():
    return {"message": "House Price Prediction API is live!"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: HouseData):
    global model
    if model is None:
        return {"error": "Model not loaded. Check server logs."}
    
    try:
        # Convert Pydantic model to DataFrame for MLflow
        input_df = pd.DataFrame([data.dict()])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Return result (ensure it's a standard float for JSON)
        return {
            "prediction": float(prediction[0]),
            "currency": "USD"
        }
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        return {"error": str(e)}