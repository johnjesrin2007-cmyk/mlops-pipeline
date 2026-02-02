import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prefect import flow
from src.preprocess import preprocess_data
from src.train import train_model

@flow(name="ML Training Pipeline")
def training_pipeline():
    X, y = preprocess_data("data/raw.csv")
    model = train_model(X, y)

if __name__ == "__main__":
    training_pipeline()
