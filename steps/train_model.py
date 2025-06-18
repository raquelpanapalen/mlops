import random
from pydantic import BaseModel
import mlflow.entities
from zenml import step
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.models import infer_signature


class TrainConfig(BaseModel):
    max_depth: int = 5
    n_estimators: int = 100
    experiment_name: str = "airline_satisfaction_experiment"


@step(enable_cache=False)
def train_model_step(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: TrainConfig,
) -> str:
    """ "
    Training Step with Fault Injection, Error Handling, and MLflow Tracking.

    Simulates failure scenarios for robustness:
    - Aborts if dataset is too small (<1000 records)
    - Simulated random crash (20% chance)

    Includes clean-up logic on failure to:
    - Remove partially written model files
    - Ensure consistent pipeline state
    - Log actionable error messages

    Trained model is logged to MLflow and saved locally.
    """

    # Simulate different training sizes
    # Randomly select a size for the training set
    # This is just for demonstration; in practice, you would use the actual dataset size
    train_size = random.randrange(0, len(X_train))
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]

    # Set tracking URI and experiment
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(config.experiment_name)

    try:

        # Start MLflow run
        with mlflow.start_run(run_name="train_airline_model") as run:

            if len(X_train) < 1000:
                raise RuntimeError(
                    "Training aborted: dataset size is too small (< 1000 records)"
                )

            # Simulate random crash
            if random.random() < 0.05:
                raise RuntimeError("Simulated random crash during training")

            # Train model
            model = RandomForestClassifier(
                max_depth=config.max_depth,
                n_estimators=config.n_estimators,
                random_state=42,
            )
            model.fit(X_train, y_train)

            # Evaluate model
            predictions = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, predictions)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            # Log model and metrics
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_params(model.get_params())
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", accuracy)

            # Log model with metadata
            logged_model = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=f"{model.__class__.__name__}_{config.experiment_name}",
                signature=infer_signature(X_train, y_train),
            )

        mlflow.end_run(status=mlflow.entities.RunStatus.FINISHED)

        # Return the URI of the logged model
        return logged_model.model_uri

    except Exception as e:
        mlflow.end_run(status=mlflow.entities.RunStatus.FAILED)

        raise RuntimeError(f"Training failed: {str(e)}")
