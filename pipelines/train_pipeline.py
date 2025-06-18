from zenml import pipeline
from steps.data_validation import data_validation_step
from steps.train_model import train_model_step, TrainConfig
from steps.robustness_test import robustness_test_step
from steps.load_predeploy_data import load_predeploy_data_step
from steps.load_train_data import load_train_data_step


@pipeline
def airline_pipeline(config: TrainConfig):
    """Pipeline for training a model on airline passenger satisfaction data."""
    data_ref, data_cur = load_predeploy_data_step()
    data_validation_step(data_ref=data_ref, data_cur=data_cur)
    X_train, X_test, y_train, y_test = load_train_data_step()
    trained_model = train_model_step(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=config,
    )
    robustness_test_step(trained_model, X_train, X_test, y_train, y_test)
