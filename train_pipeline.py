from pipelines.train_pipeline import airline_pipeline
from steps.train_model import TrainConfig
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-version", type=str, required=True)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()

    print(f"Running training flow version: {args.flow_version}")

    config = TrainConfig(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        flow_version=args.flow_version,
    )

    airline_pipeline(config)
