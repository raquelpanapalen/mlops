import mlflow
import mlflow.entities
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def create_baseline(X_train, X_test, y_train, y_test):

    # Set tracking URI and experiment
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("airline_satisfaction_experiment")

    with mlflow.start_run(run_name="train_airline_baseline_model") as run:
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        predictions = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, predictions)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log model and metrics
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_params(model.get_params())
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", accuracy)

        # Log model with metadata
        logged_model = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=f"{model.__class__.__name__}_airline_model",
            signature=infer_signature(X_train, y_train),
        )

    mlflow.end_run(status=mlflow.entities.RunStatus.FINISHED)

    return logged_model.model_uri
