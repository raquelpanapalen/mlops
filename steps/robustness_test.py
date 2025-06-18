from zenml import step
import mlflow
import pandas as pd
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestAccuracyScore,
    TestPrecisionScore,
    TestRecallScore,
    TestF1Score,
    TestPrecisionByClass,
    TestRecallByClass,
    TestF1ByClass,
)


from scripts.create_baseline import create_baseline


@step(enable_cache=False)
def robustness_test_step(
    model_uri: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """Test the robustness of the model by checking its performance on a perturbed dataset."""

    # Load the model
    model = mlflow.pyfunc.load_model(model_uri)

    # Load the baseline model
    try:
        baseline_model = mlflow.pyfunc.load_model(
            "models:/LogisticRegression_baseline/latest"
        )
    except Exception as e:
        # Create a baseline model if it doesn't exist
        baseline_uri = create_baseline(X_train, X_test, y_train, y_test)
        baseline_model = mlflow.pyfunc.load_model(baseline_uri)

    # Predict using the current model and the baseline model
    y_pred_current = model.predict(X_test)
    y_pred_baseline = baseline_model.predict(X_test)

    # Create a DataFrame to hold the predictions
    data_cur = pd.DataFrame({"target": y_test, "prediction": y_pred_current})
    data_ref = pd.DataFrame({"target": y_test, "prediction": y_pred_baseline})

    classification_tests_suite = TestSuite(
        tests=[
            TestAccuracyScore(),
            TestPrecisionScore(),
            TestRecallScore(),
            TestF1Score(),
            TestPrecisionByClass(label=0),
            TestPrecisionByClass(label=1),
            TestRecallByClass(label=0),
            TestRecallByClass(label=1),
            TestF1ByClass(label=0),
            TestF1ByClass(label=1),
        ]
    )

    classification_tests_suite.run(reference_data=data_ref, current_data=data_cur)
    for result in classification_tests_suite.as_dict()["tests"]:
        # Custom threshold: only fail if accuracy drops by more than 10%
        ref_score = result["parameters"]["condition"]["eq"]["value"] * 0.9
        cur_score = result["parameters"]["value"]
        if cur_score < ref_score:
            raise ValueError(
                f"❌ Robustness test failed: The {result['name']} is {cur_score:.2f}. The test threshold is gt={ref_score:.2f}."
            )
