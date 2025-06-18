from zenml import pipeline
from steps.load_train_data import load_train_data_step
from steps.load_unseen_data import load_unseen_data_step
from steps.drift_tests import drift_test_step


@pipeline
def monitoring_pipeline():
    train_data, _, _, _ = load_train_data_step()
    unseen_data = load_unseen_data_step()
    drift_test_step(train_data, unseen_data)
