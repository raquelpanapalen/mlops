import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_golden_set(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    satisfaction_cols = [
        "Departure and Arrival Time Convenience",
        "Ease of Online Booking",
        "Check-in Service",
        "Online Boarding",
        "Gate Location",
        "On-board Service",
        "Seat Comfort",
        "Leg Room Service",
        "Cleanliness",
        "Food and Drink",
        "In-flight Service",
        "In-flight Wifi Service",
        "In-flight Entertainment",
        "Baggage Handling",
    ]
    # make sure all satisfaction columns are numeric --> make them float (to avoid confusion between str and int: evidently doesn't like it)
    df[satisfaction_cols] = df[satisfaction_cols].astype(float)

    very_insatisfied = df[df[satisfaction_cols].le(2).all(axis=1)]

    # Here we have a problem with the data, as passengers under 18 shouldn't be taking a satisfaction survey
    extreme_age = df[(df["Age"] < 18) | (df["Age"] > 80)]

    high_delay = df[(df["Departure Delay"] > 180) | (df["Arrival Delay"] > 180)]
    extreme_distance = df[
        (df["Flight Distance"] < 100) | (df["Flight Distance"] > 4000)
    ]

    # Combine all edge cases (no duplicates)
    edge_cases = pd.concat(
        [high_delay, extreme_distance, very_insatisfied, extreme_age]
    ).drop_duplicates()

    # Remove edge cases from original to avoid duplicate sampling
    df_remaining = df.drop(edge_cases.index)

    # === STRATIFIED SAMPLING FOR REPRESENTATIVE GOLDEN SET ===

    # We’ll sample a balanced stratified subset from remaining data
    # Note: this assumes ~40% of the original dataset size is a reasonable size for golden test set
    strat_cols = [
        "Satisfaction",
        "Customer Type",
        "New Class",
        "Type of Travel",
        "Gender",
    ]

    # To use sklearn’s train_test_split for stratification, combine these cols
    df_remaining["New Class"] = df_remaining["Class"].replace(
        {"Economy Plus": "Economy"}
    )
    df_remaining["strata"] = df_remaining[strat_cols].astype(str).agg("-".join, axis=1)

    # Choose size of golden set excluding edge cases
    n_golden_regular = int(df.shape[0] * 0.4)
    golden_regular, rest = train_test_split(
        df_remaining,
        stratify=df_remaining["strata"],
        test_size=(len(df_remaining) - n_golden_regular),
        random_state=42,
    )

    # Remove the strata column
    golden_regular = golden_regular.drop(columns=["strata", "New Class"])
    rest = rest.drop(columns=["strata", "New Class"])

    # Combine with edge cases to form final golden set
    golden_set = pd.concat([golden_regular, edge_cases]).drop_duplicates()

    return golden_set, rest


def preprocess_data(df: pd.DataFrame):
    """Preprocess the dataset by handling missing values and encoding categorical variables."""
    # Handling missing values on Arrival Delay
    nan_df = df[df["Arrival Delay"].isna()]
    if len(nan_df) > 0:
        # Fill missing values with Departure Delay + some random noise
        # to avoid overfitting (+- * 0.2)
        alpha = np.random.uniform(0.9, 1.1, len(nan_df))
        df.loc[df["Arrival Delay"].isna(), "Arrival Delay"] = (
            df.loc[df["Arrival Delay"].isna(), "Departure Delay"] * alpha
        ).astype(int)

    df["Satisfaction"] = df["Satisfaction"].apply(
        lambda x: 1 if x == "Satisfied" else 0
    )
    categorical_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    features = [col for col in df.columns if col not in ["Satisfaction", "ID"]]
    X = df[features]
    y = df["Satisfaction"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
