import os
import sys
import mlflow
from prefect import flow

# Always use local MLflow file store (for development only)
mlflow.set_tracking_uri("file:./mlruns")

# Make sure src/ is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import preprocess_data
from src.train import train_model


@flow(name="ML Training Pipeline")
def training_pipeline():
    data_path = "data/raw.csv"

    # Preprocess
    X, y = preprocess_data(
    path=data_path,
    training=True,
    target_col="target"   # 🔥 change this to your actual target column name
)

    # Train + save model
    train_model(X, y)


if __name__ == "__main__":
    training_pipeline()

