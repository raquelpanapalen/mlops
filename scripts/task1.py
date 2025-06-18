import pandas as pd
import pytest
import os
from pathlib import Path
from evidently.test_suite import TestSuite
from evidently.tests import *

# This script is used to test the data integrity and quality of the airline passenger satisfaction dataset.
# The tests are based on the Evidently library, which provides a framework for testing data quality and integrity.
# The tests include checking for missing values, duplicates, and data distribution for both categorical and numerical features.
# The tests are run using pytest, and the results are reported in a structured format.

data_dir = Path(os.path.dirname(__file__)).parent / "data"
# Load the golden set and current set
data_ref = pd.read_csv(data_dir / "processed" / "golden_set.csv")
data_cur = pd.read_csv(data_dir / "processed" / "current_set.csv")


@pytest.fixture
def load_data():
    """Load golden set and current set."""

    return data_ref, data_cur


# SUBTASK 1: Data Integrity and Quality (missing values, duplicates, etc.)
@pytest.mark.parametrize(
    "test_class, test_name",
    [
        (TestNumberOfColumns, "Number of Columns"),
        (TestNumberOfMissingValues, "Number of Missing Values"),
        (TestShareOfMissingValues, "Share of Missing Values"),
        (TestNumberOfColumnsWithMissingValues, "Number of Columns with Missing Values"),
        (TestNumberOfRowsWithMissingValues, "Number of Rows with Missing Values"),
        (TestShareOfColumnsWithMissingValues, "Share of Columns with Missing Values"),
        (TestShareOfRowsWithMissingValues, "Share of Rows with Missing Values"),
        (TestNumberOfDifferentMissingValues, "Number of Different Missing Values"),
        (TestNumberOfConstantColumns, "Number of Constant Columns"),
        (TestNumberOfEmptyRows, "Number of Empty Rows"),
        (TestNumberOfEmptyColumns, "Number of Empty Columns"),
        (TestNumberOfDuplicatedRows, "Number of Duplicated Rows"),
        (TestNumberOfDuplicatedColumns, "Number of Duplicated Columns"),
        (TestColumnsType, "Columns Type"),
    ],
)
def test_data_integrity(test_class, test_name, load_data):
    """Generic function to test data integrity and quality."""
    data_ref, data_cur = load_data
    test = TestSuite(tests=[test_class()])

    # Run the test
    test.run(reference_data=data_ref, current_data=data_cur)
    report = test.as_dict()["tests"][0]

    # Assert if test passed or failed
    assert report["status"] == "SUCCESS", report["description"]


# SUBTASK 2: Data Distribution (categorical and numerical features)

# Define the columns to test
columns_to_test_value_dist = [
    ("Age", {"min": 18, "max": 100}),
    ("Flight Distance", {"min": 50, "max": 9500}),
    ("Departure Delay", {"min": 0, "max": 2880}),  # 2 days of delay
    ("Arrival Delay", {"min": 0, "max": 2880}),
    ("Departure and Arrival Time Convenience", {"min": 0, "max": 5}),
    ("Ease of Online Booking", {"min": 0, "max": 5}),
    ("Check-in Service", {"min": 0, "max": 5}),
    ("Online Boarding", {"min": 0, "max": 5}),
    ("Gate Location", {"min": 0, "max": 5}),
    ("On-board Service", {"min": 0, "max": 5}),
    ("Seat Comfort", {"min": 0, "max": 5}),
    ("Leg Room Service", {"min": 0, "max": 5}),
    ("Cleanliness", {"min": 0, "max": 5}),
    ("Food and Drink", {"min": 0, "max": 5}),
    ("In-flight Service", {"min": 0, "max": 5}),
    ("In-flight Wifi Service", {"min": 0, "max": 5}),
    ("In-flight Entertainment", {"min": 0, "max": 5}),
    ("Baggage Handling", {"min": 0, "max": 5}),
]


# Parametrized test for column value distribution
@pytest.mark.parametrize("column_name, expectation", columns_to_test_value_dist)
def test_column_value_distribution(column_name, expectation, load_data):
    data_ref, data_cur = load_data
    # Expectation is a range (e.g., for numerical columns like Age or Flight Distance)
    test = TestSuite(
        tests=[
            TestValueRange(
                column_name=column_name,
                left=expectation["min"],
                right=expectation["max"],
            )
        ]
    )

    # Run the test
    test.run(reference_data=data_ref, current_data=data_cur)
    report = test.as_dict()["tests"][0]

    # Assert if test passed or failed
    assert report["status"] == "SUCCESS", report["description"]


if __name__ == "__main__":
    pytest.main()
