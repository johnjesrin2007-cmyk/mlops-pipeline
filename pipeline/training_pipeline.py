import os
import sys
import mlflow

# Always use local MLflow file store
mlflow.set_tracking_uri("file:./mlruns")

# Make sure src/ is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prefect import flow
from src.preprocess import preprocess_data
from src.train import train_model


@flow(name="ML Training Pipeline")
def training_pipeline():
    # Path to dataset
    data_path = "data/raw.csv"

    # Run preprocessing
    X, y = preprocess_data(data_path)

    # Train model and log artifacts
    train_model(X, y)


if __name__ == "__main__":
    training_pipeline()
