import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class AirlineDataManager:
    def __init__(self, data_path: str, output_dir: str = "data/processed"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.df = None
        self.golden_set = None
        self.current_set = None
        self.golden_train = None
        self.golden_test = None
        self._load_data()

    def _load_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Data loaded from {self.data_path}, shape: {self.df.shape}")

    def split_golden_and_current(self):
        """Extract a representative golden set from the dataset."""
        df = self.df.copy()

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
        df_remaining["strata"] = (
            df_remaining[strat_cols].astype(str).agg("-".join, axis=1)
        )

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
        self.golden_set = pd.concat([golden_regular, edge_cases]).drop_duplicates()
        self.current_set = rest

        print(
            f"✅ Golden set: {self.golden_set.shape}, Current set: {self.current_set.shape}"
        )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset by handling missing values and encoding categorical variables."""
        df = df.copy()

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
        for col in ["Gender", "Customer Type", "Type of Travel", "Class"]:
            df[col] = LabelEncoder().fit_transform(df[col])

        return df

    def preprocess_sets(self):
        self.golden_set = self.preprocess(self.golden_set)
        self.current_set = self.preprocess(self.current_set)
        print("✅ Preprocessing complete for golden and current sets.")

    def split_golden_train_test(self):
        df = self.golden_set.copy()
        self.golden_train, self.golden_test = train_test_split(
            df, test_size=0.2, random_state=42
        )
        print(
            f"✅ Golden train: {self.golden_train.shape}, Golden test: {self.golden_test.shape}"
        )

    def save_all(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.golden_train.to_csv(self.output_dir / "golden_train.csv", index=False)
        self.golden_test.to_csv(self.output_dir / "golden_test.csv", index=False)
        self.golden_set.to_csv(self.output_dir / "golden_set.csv", index=False)
        self.current_set.to_csv(self.output_dir / "current_set.csv", index=False)
        print(f"✅ Saved all splits to {self.output_dir}")

    def run_all(self):
        self.split_golden_and_current()
        self.preprocess_sets()
        self.split_golden_train_test()
        self.save_all()


if __name__ == "__main__":
    manager = AirlineDataManager("data/airline_passenger_satisfaction.csv")
    manager.run_all()
