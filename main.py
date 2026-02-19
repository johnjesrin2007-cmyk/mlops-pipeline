from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="ML Model API")

# Load model safely
MODEL_PATH = "model/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Make sure training ran successfully.")

model = joblib.load(MODEL_PATH)


# Input schema matching your dataset
class InputData(BaseModel):
    feature1: float
    feature2: float


@app.get("/")
def home():
    return {"message": "ML Model API is running successfully 🚀"}


@app.post("/predict")
def predict(data: InputData):
    try:
        input_array = np.array([[data.feature1, data.feature2]])
        prediction = model.predict(input_array)

        return {
            "feature1": data.feature1,
            "feature2": data.feature2,
            "prediction": float(prediction[0])
        }

    except Exception as e:
        return {"error": str(e)}
