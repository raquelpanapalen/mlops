from zenml import step
import mlflow
from scripts.utils import get_golden_set


@step
def drift_test(model_uri: str):
    """Test the robustness of the model by checking its performance on a perturbed dataset."""

    # Load the model
    model = mlflow.pyfunc.load_model(model_uri)

    # Load the current dataset
    _, data_cur = get_golden_set()
