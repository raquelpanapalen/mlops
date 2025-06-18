from zenml import step
import pandas as pd
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
)


@step
def drift_test_step(train_data: pd.DataFrame, unseen_data: pd.DataFrame):
    """Run data drift tests on the unseen data against the training data."""

    # Drop ID and Satisfaction columns if they are included in the drift tests
    train_data = train_data.drop(columns=["ID", "Satisfaction"], errors="ignore")
    unseen_data = unseen_data.drop(columns=["ID", "Satisfaction"], errors="ignore")

    data_drift_dataset_tests = TestSuite(
        tests=[
            TestNumberOfDriftedColumns(),
            TestShareOfDriftedColumns(),
        ]
    )

    data_drift_dataset_tests.run(reference_data=train_data, current_data=unseen_data)
    for result in data_drift_dataset_tests.as_dict()["tests"]:
        detected_drifts = {
            k: v for k, v in result["parameters"]["features"].items() if v["detected"]
        }
        print(f"Detected drifts in {result['name']}:\n {detected_drifts}")
        if result["status"] != "SUCCESS":
            raise ValueError(f"Data drift test failed: {result['description']}")
