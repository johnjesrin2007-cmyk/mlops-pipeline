import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from prefect import task

@task
def train_model(X, y):
    mlflow.set_experiment("house_price_experiment")

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X, y)

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("r2_score", model.score(X, y))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="HousePriceModel"
        )

    return model
