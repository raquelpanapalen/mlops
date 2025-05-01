from zenml import step
import pandas as pd
import logging
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfColumns,
    TestNumberOfMissingValues,
    TestShareOfMissingValues,
    TestNumberOfColumnsWithMissingValues,
    TestNumberOfRowsWithMissingValues,
    TestShareOfColumnsWithMissingValues,
    TestShareOfRowsWithMissingValues,
    TestNumberOfDifferentMissingValues,
    TestNumberOfConstantColumns,
    TestNumberOfEmptyRows,
    TestNumberOfEmptyColumns,
    TestNumberOfDuplicatedRows,
    TestNumberOfDuplicatedColumns,
    TestColumnsType,
    TestValueRange,
)
from scripts.utils import get_golden_set


@step
def data_validation_step(df: pd.DataFrame) -> pd.DataFrame:
    """Validate data integrity and distribution. Returns the current dataset if all checks pass."""

    data_ref, data_cur = get_golden_set(df)

    # --- SUBTASK 1: Integrity Tests ---
    integrity_tests = [
        TestNumberOfColumns(),
        TestNumberOfMissingValues(),
        TestShareOfMissingValues(),
        TestNumberOfColumnsWithMissingValues(),
        TestNumberOfRowsWithMissingValues(),
        TestShareOfColumnsWithMissingValues(),
        TestShareOfRowsWithMissingValues(),
        TestNumberOfDifferentMissingValues(),
        TestNumberOfConstantColumns(),
        TestNumberOfEmptyRows(),
        TestNumberOfEmptyColumns(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestColumnsType(),
    ]

    integrity_suite = TestSuite(tests=integrity_tests)
    integrity_suite.run(reference_data=data_ref, current_data=data_cur)
    for result in integrity_suite.as_dict()["tests"]:
        if result["status"] != "SUCCESS":
            raise ValueError(f"Integrity test failed: {result['description']}")

    # --- SUBTASK 2: Value Distribution Tests ---
    value_tests_config = [
        ("Age", 18, 100),
        ("Flight Distance", 50, 9500),
        ("Departure Delay", 0, 2880),
        ("Arrival Delay", 0, 2880),
        ("Departure and Arrival Time Convenience", 0, 5),
        ("Ease of Online Booking", 0, 5),
        ("Check-in Service", 0, 5),
        ("Online Boarding", 0, 5),
        ("Gate Location", 0, 5),
        ("On-board Service", 0, 5),
        ("Seat Comfort", 0, 5),
        ("Leg Room Service", 0, 5),
        ("Cleanliness", 0, 5),
        ("Food and Drink", 0, 5),
        ("In-flight Service", 0, 5),
        ("In-flight Wifi Service", 0, 5),
        ("In-flight Entertainment", 0, 5),
        ("Baggage Handling", 0, 5),
    ]

    for col, min_val, max_val in value_tests_config:
        value_test = TestSuite(
            tests=[TestValueRange(column_name=col, left=min_val, right=max_val)]
        )
        value_test.run(reference_data=data_ref, current_data=data_cur)
        report = value_test.as_dict()["tests"][0]
        if report["status"] != "SUCCESS":
            raise ValueError(f"Value range check failed: {report['description']}")

    return data_ref
