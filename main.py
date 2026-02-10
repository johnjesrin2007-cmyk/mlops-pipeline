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

# 2. Enable CORS (Crucial for web deployments)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for the model
model = None

# 3. Input Data Schema
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
        # Get absolute path to the root (/app on Render)
        current_dir = os.path.abspath(os.getcwd())
        mlruns_path = os.path.join(current_dir, "mlruns")
        
        # Explicitly set tracking to local folder
        mlflow.set_tracking_uri(f"file://{mlruns_path}")
        logger.info(f"MLflow tracking set to: {mlruns_path}")

        # Search for the latest run in experiment '0'
        experiment_id = "0"
        runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["attributes.start_time DESC"])
        
        if not runs.empty:
            latest_run_id = runs.iloc[0].run_id
            model_uri = f"runs:/{latest_run_id}/model"
            logger.info(f"Loading latest model from Run ID: {latest_run_id}")
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("✅ SUCCESS: Model loaded and ready for predictions.")
        else:
            logger.warning("⚠️ No runs found in mlruns/0. Checking Model Registry...")
            model = mlflow.pyfunc.load_model("models:/HousePriceModel/Production")
            logger.info("✅ SUCCESS: Model loaded from Registry.")

    except Exception as e:
        logger.error(f"❌ CRITICAL: Model failed to load. Error: {str(e)}")
        model = None

@app.get("/")
def read_root():
    return {"message": "API is online. Use /predict for pricing or /docs for testing."}

@app.get("/health")
def health():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "mlruns_dir_found": os.path.exists(os.path.join(os.getcwd(), "mlruns"))
    }

# 4. The Prediction Endpoint with Safety Check
@app.post("/predict")
def predict(data: HouseData):
    global model
    
    # SAFETY CHECK: If model loading failed at startup, stop here
    if model is None:
        logger.error("Prediction requested but model is not loaded.")
        raise HTTPException(
            status_code=503, 
            detail="Model is currently unavailable. Please check server logs for mlruns errors."
        )
    
    try:
        # Convert Pydantic object to Dictionary, then to DataFrame
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        return {
            "prediction": float(prediction[0]),
            "currency": "USD",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")