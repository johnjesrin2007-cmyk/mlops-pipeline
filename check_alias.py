import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

print("Tracking URI:", mlflow.get_tracking_uri())

model = "HousePriceModel"

aliases = client.get_registered_model(model).aliases
print("Aliases:", aliases)

versions = client.search_model_versions(f"name='{model}'")
for v in versions:
    print("Version:", v.version, "Aliases:", v.aliases)
