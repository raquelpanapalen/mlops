from zenml import step
import pandas as pd
from pathlib import Path


@step
def load_predeploy_data_step() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the pre-deployment datasets for data validation."""
    path = Path(__file__).parent.parent / "data" / "processed"
    golden_set = pd.read_csv(path / "golden_set.csv")
    current_set = pd.read_csv(path / "current_set.csv")

    return golden_set, current_set
