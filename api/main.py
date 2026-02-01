import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

MODEL_URI = "models:/HousePriceModel@production"
model = mlflow.pyfunc.load_model(MODEL_URI)


class HouseInput(BaseModel):
    area: float
    bedrooms: int

@app.post("/predict")
def predict(data: HouseInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    return {"prediction": float(prediction[0])}
