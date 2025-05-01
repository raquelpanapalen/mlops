from zenml import pipeline, step
from steps.data_validation import data_validation_step
from steps.train_model import train_model_step
from steps.robustness_test import robustness_test_step
import pandas as pd
from pathlib import Path


@step
def load_data_step() -> pd.DataFrame:
    """Load the airline passenger satisfaction dataset."""
    data_path = Path(__file__).parent / "data" / "airline_passenger_satisfaction.csv"
    return pd.read_csv(data_path)


@pipeline
def airline_pipeline():
    """Pipeline for training a model on airline passenger satisfaction data."""
    raw_df = load_data_step()
    validated_df = data_validation_step(raw_df)
    trained_model = train_model_step(validated_df)
    robustness_test_step(trained_model, validated_df)


if __name__ == "__main__":
    airline_pipeline()
