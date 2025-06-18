from zenml import step
import pandas as pd
from pathlib import Path


@step
def load_unseen_data_step() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the pre-deployment datasets for data validation."""
    path = Path(__file__).parent.parent / "data" / "processed"
    test_df = pd.read_csv(path / "current_set.csv")
    X_test = test_df.drop(columns=["ID", "Satisfaction"])
    y_test = test_df["Satisfaction"]

    return X_test, y_test
