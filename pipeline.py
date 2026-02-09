from src.preprocess import preprocess_data
from src.train import train_model


def run_training_pipeline():
    """
    One-command training pipeline
    """
    X, y = preprocess_data(
        path="data/raw.csv",
        training=True
    )

    run_id = train_model(X, y)
    print(f"Training completed. MLflow run_id: {run_id}")


if __name__ == "__main__":
    run_training_pipeline()
