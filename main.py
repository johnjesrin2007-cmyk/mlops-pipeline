from fastapi import FastAPI
import mlflow
import os
import logging

# Setup logging to see what's happening in Render logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI()

# Global variable for the model
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        # 1. Define the base path inside the Render Docker container
        # Your project is copied to /app, and your mlruns is in the root
        base_path = "/app/mlruns/0"
        
        if not os.path.exists(base_path):
            logger.error(f"❌ Path not found: {base_path}. Check if mlruns was pushed to Git.")
            return

        # 2. Auto-detect the Run ID folder
        # We look for any folder inside '0' that contains the 'artifacts' directory
        subfolders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        
        run_folder = None
        for folder in subfolders:
            # Check if this subfolder contains the model artifacts
            artifact_path = os.path.join(base_path, folder, "artifacts", "model")
            if os.path.exists(artifact_path):
                run_folder = folder
                break
        
        if not run_folder:
            logger.error("❌ Could not find a folder containing 'artifacts/model' inside mlruns/0")
            return

        # 3. Construct the final URI and load the model
        model_uri = f"{base_path}/{run_folder}/artifacts/model"
        logger.info(f"🚀 Auto-detected model path: {model_uri}")
        
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ SUCCESS: Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Deployment Error during startup: {str(e)}")

@app.get("/health")
def health_check():
    """Check if the API is alive and the model is loaded."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "provider": "Render"
    }

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API is running!"}

# You can add your /predict endpoint below this line