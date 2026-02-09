import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from prefect import task

@task
def train_model(X, y):
    # Always point MLflow to local file store
    mlflow.set_tracking_uri("file:./mlruns")

    # Optional but clean: set experiment
    mlflow.set_experiment("house_price_experiment")

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X, y)

        # Log parameters & metrics
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("r2_score", model.score(X, y))

        # 🔴 IMPORTANT: log model as an artifact (NO registry)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

    return model
