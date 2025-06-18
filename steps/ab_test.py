import os
import json
import mlflow
import tempfile
import numpy as np
import pandas as pd
from zenml import step
from sklearn.metrics import accuracy_score, f1_score


@step
def ab_test_step(
    preds_a: np.ndarray | list,
    preds_b: np.ndarray | list,
    labels_a: np.ndarray | list | pd.Series,
    labels_b: np.ndarray | list | pd.Series,
    model_uri_a: str,
    model_uri_b: str,
    test_id: str,
):
    """Compares predictions from two model versions and logs the results."""

    mlflow.set_experiment("airline_satisfaction")

    acc_a = accuracy_score(labels_a, preds_a)
    acc_b = accuracy_score(labels_b, preds_b)
    f1_a = f1_score(labels_a, preds_a)
    f1_b = f1_score(labels_b, preds_b)

    # === Log into MLflow ===
    with mlflow.start_run(run_name=f"A/B Test: {test_id}") as parent_run:
        mlflow.set_tag("test_identifier", test_id)
        mlflow.set_tag("ab_test", "true")

        # Log both child runs
        with mlflow.start_run(nested=True, run_name="model_A"):
            mlflow.log_param("model_uri", model_uri_a)
            mlflow.log_metric("accuracy", acc_a)
            mlflow.log_metric("f1_score", f1_a)

        with mlflow.start_run(nested=True, run_name="model_B"):
            mlflow.log_param("model_uri", model_uri_b)
            mlflow.log_metric("accuracy", acc_b)
            mlflow.log_metric("f1_score", f1_b)

        # Save and log summary
        summary = {
            "test_id": test_id,
            "split_method": "user_id % 2",
            "model_uri": {"A": model_uri_a, "B": model_uri_b},
            "metrics": {
                "accuracy": {"A": acc_a, "B": acc_b},
                "f1_score": {"A": f1_a, "B": f1_b},
            },
        }

        tmp_path = os.path.join(
            tempfile.gettempdir(), f"ab_test_summary_{test_id}.json"
        )
        with open(tmp_path, "w") as f:
            json.dump(summary, f, indent=2)

        mlflow.log_artifact(tmp_path)
