from pipelines.ab_pipeline import ab_test_pipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow_version_a", type=str, required=True)
    parser.add_argument("--flow_version_b", type=str, required=True)
    parser.add_argument("--test_id", type=str, default="abtest_001")
    args = parser.parse_args()

if __name__ == "__main__":
    ab_test_pipeline(
        flow_version_a=args.flow_version_a,
        flow_version_b=args.flow_version_b,
        test_id=args.test_id,
    )
