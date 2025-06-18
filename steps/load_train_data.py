from zenml import step
import pandas as pd
from pathlib import Path


@step
def load_train_data_step() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the pre-deployment datasets for data validation."""
    path = Path(__file__).parent.parent / "data" / "processed"
    train_df = pd.read_csv(path / "golden_train.csv")
    X_train = train_df.drop(columns=["ID", "Satisfaction"])
    y_train = train_df["Satisfaction"]

    test_df = pd.read_csv(path / "golden_test.csv")
    X_test = test_df.drop(columns=["ID", "Satisfaction"])
    y_test = test_df["Satisfaction"]

    return X_train, X_test, y_train, y_test
