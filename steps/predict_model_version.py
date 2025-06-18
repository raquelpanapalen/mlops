import mlflow
import numpy as np
import pandas as pd
from zenml import step


@step
def predict_with_model_versionid(
    test_df: pd.DataFrame, flow_version_id: str
) -> tuple[str, np.ndarray]:
    """Returns model artifact URI for the latest run of a specific flow version."""

    runs = mlflow.search_runs(
        experiment_names=["airline_satisfaction"],
        filter_string=f"tags.flow_version = '{flow_version_id}'",
        order_by=["start_time desc"],
    )

    if runs.empty:
        raise ValueError(f"No runs found for flow version: {flow_version_id}")

    # Get the latest run for the specified flow version
    latest_run = runs.iloc[0]
    model_uri = f"runs:/{latest_run.run_id}/model"

    model = mlflow.pyfunc.load_model(model_uri)
    return model_uri, model.predict(test_df)
