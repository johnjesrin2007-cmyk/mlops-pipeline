from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="file:./mlruns")

# Assign alias "production" to version 2
client.set_registered_model_alias(
    name="HousePriceModel",
    alias="production",
    version="2"
)

print("Alias 'production' set for HousePriceModel v2")
