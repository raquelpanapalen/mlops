from zenml import pipeline
from steps.load_unseen_data import load_unseen_data_step
from steps.split_for_ab_test import split_for_ab_test
from steps.predict_model_version import predict_with_model_versionid
from steps.ab_test import ab_test_step


@pipeline
def ab_test_pipeline(flow_version_a: str, flow_version_b: str, test_id: str):
    """Pipeline for A/B testing of two model versions."""
    test_df = load_unseen_data_step()

    X_a, X_b, y_a, y_b = split_for_ab_test(test_df)

    model_uri_a, preds_a = predict_with_model_versionid(X_a, flow_version_a)
    model_uri_b, preds_b = predict_with_model_versionid(X_b, flow_version_b)

    ab_test_step(
        preds_a=preds_a,
        preds_b=preds_b,
        labels_a=y_a,
        labels_b=y_b,
        model_uri_a=model_uri_a,
        model_uri_b=model_uri_b,
        test_id=test_id,
    )
