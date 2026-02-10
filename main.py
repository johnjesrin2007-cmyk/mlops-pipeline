import os
import logging
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="House Price Prediction API")

# 2. Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

# 3. Data Schema
class HouseData(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: int 
    guestroom: int

@app.on_event("startup")
def load_model():
    global model
    try:
        # Use absolute path to ensure we are in /app (Docker root)
        current_dir = os.path.abspath(os.getcwd())
        mlruns_path = os.path.join(current_dir, "mlruns")
        
        # Point MLflow to the local directory
        mlflow.set_tracking_uri(f"file://{mlruns_path}")
        logger.info(f"Looking for experiments in: {mlruns_path}")

        # Search for the latest run in experiment 0
        # This fixes the 'No versions of model found' error
        runs = mlflow.search_runs(experiment_ids=["0"], order_by=["attributes.start_time DESC"])
        
        if not runs.empty:
            latest_run_id = runs.iloc[0].run_id
            # Direct path to the model artifacts
            model_uri = f"runs:/{latest_run_id}/model"
            logger.info(f"Loading latest Run ID: {latest_run_id}")
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("✅ SUCCESS: Model loaded from local run!")
        else:
            logger.error("❌ ERROR: No runs found in mlruns/0. Make sure your mlruns folder is uploaded!")
            
    except Exception as e:
        logger.error(f"❌ CRITICAL: {str(e)}")
        model = None

@app.get("/")
def read_root():
    return {"message": "API is online. Go to /docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: HouseData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check logs.")
    
    try:
        # Pydantic v1: .dict() | Pydantic v2: .model_dump()
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)
        return {"prediction": float(prediction[0]), "currency": "USD"}
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))