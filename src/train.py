import os
import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def train_model(X, y):

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Start MLflow run
    with mlflow.start_run():

        # Initialize model
        model = LinearRegression()

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        r2 = r2_score(y_test, y_pred)

        # Log metric
        mlflow.log_metric("r2_score", r2)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        print(f"✅ R2 Score: {r2}")

    # ------------------------------
    # 🔥 SAVE MODEL FOR PRODUCTION
    # ------------------------------
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")

    print("✅ Model saved to model/model.pkl")
