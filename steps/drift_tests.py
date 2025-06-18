from zenml import step
import pandas as pd
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
)


@step
def drift_test_step(train_data: pd.DataFrame, unseen_data: pd.DataFrame) -> float:
    data_drift_dataset_tests = TestSuite(
        tests=[
            TestNumberOfDriftedColumns(),
            TestShareOfDriftedColumns(),
        ]
    )

    data_drift_dataset_tests.run(reference_data=train_data, current_data=unseen_data)
    for result in data_drift_dataset_tests.as_dict()["tests"]:
        if result["status"] != "SUCCESS":
            raise ValueError(f"Data drift test failed: {result['description']}")
