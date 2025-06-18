from zenml import step
import pandas as pd
from pathlib import Path


@step
def load_unseen_data_step() -> pd.DataFrame:
    """Load the pre-deployment datasets for data validation."""
    path = Path(__file__).parent.parent / "data" / "processed"
    test_df = pd.read_csv(path / "current_set.csv")

    return test_df
