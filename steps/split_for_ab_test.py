from zenml import step
import pandas as pd
from sklearn.utils import shuffle


@step
def split_for_ab_test(
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    test_df = test_df.copy()
    test_df = shuffle(test_df, random_state=42)
    test_df["split_group"] = test_df["ID"] % 2
    a_df = test_df[test_df["split_group"] == 0].drop(columns=["split_group", "ID"])
    b_df = test_df[test_df["split_group"] == 1].drop(columns=["split_group", "ID"])

    X_a = a_df.drop(columns=["Satisfaction"])
    X_b = b_df.drop(columns=["Satisfaction"])
    y_a = a_df["Satisfaction"]
    y_b = b_df["Satisfaction"]

    return X_a, X_b, y_a, y_b
