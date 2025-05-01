# TASK 2

This report outlines the design decisions and implementation details of a local machine learning pipeline using ZenML. The pipeline includes four main steps: (0) data loading,  (1) data validation, (2) model training and serialization, and (3) robustness testing. Each step has been modularized and documented to ensure reproducibility, maintainability, and traceability. MLflow is used for model versioning and metadata tracking.

## Step 1: Pre-Deployment Tests (Data Validation)

The data validation step performs the following:
- Schema validation (checking for nulls, type mismatches, etc.)
- Distribution checks

More information on how this validation can be seen [here](Task_1.md).


## Step 2: Model Training and Versioning

- **Model Choice**: We used a Random Forest Classifier for its robustness and good out-of-the-box performance on tabular data.

- **Artifact Output**: The trained model is serialized and stored locally under a version-controlled path using MLflow.


- **Model Versioning with MLflow**: Parameters, metrics, and artifacts (the trained model) are logged to MLflow. The model is saved and registered under a versioned MLflow experiment. Input/output schema and environment dependencies (via conda.yaml or requirements.txt) for each model are stored locally.

- **Simulated Errors**: To test robustness, artificial errors were introduced:
    - Training is aborted if the dataset contains fewer than 1000 records.
    - A 20% chance of simulated random crash mimics unpredictable failures.

- **Error Handling**: The errors are handled via try-except blocks and logged automatically to the ZenML Local Artifact Store, which can be accessed with the command `zenml login --local`. The MLflow run is explicitly ended with a `FAILED` status and no model is saved, ensuring failure is visible in the MLflow UI.

- **Output**: The output of this step is the URI of the trained and serialized model artifact with full metadata (e.g., input/output schema, hyperparameters), which is stored for future use in MLflow. With this URI, in the next step we are able to load the model, deserialize it and use it for inference.

## Step 3: Robustness Tests 

This robustness test step is designed to ensure that your trained Random Forest model not only performs well in isolation but also performs consistently better than a baseline model—in this case, Logistic Regression—under the same conditions. For that, we load our trained Random Forest model (current model), and a baseline Logistic Regression model from the MLflow model registry, and evaluate them both using Evidently on a hold-out segment of data not used in training.


### Performance testing

The **metrics** used to evaluate model performance are:
- Accuracy: Proportion of total correct predictions.
- Precision: How many of the predicted positives are correct.
- Recall: How many actual positives the model correctly identified.
- F1 Score: Harmonic mean of precision and recall.

Evaluated both overall and per class (0 = neutral or insatisfied, 1 = satisfied).


With `TestSuite`, we compare the current model (Random Forest) to the reference (Logistic Regression) with a custom test: For each metric, it checks whether the Random Forest’s score is at least 90% of the Logistic Regression's score. If any metric score from the current model is below that threshold, the pipeline raises an error — signaling a failed robustness test. The Random Forest doesn't have to be better on every metric, but should not degrade by more than 10%.


With this type of test, we're not just checking if the model works — we're checking if it's significantly better than a simple, robust baseline. Additionally, testing as well on per-class scores protects against silent failure of advanced models that might perform worse in subtle ways (e.g., on the minority class).


### Why compare against Logistic Regression?
Logistic Regression is a strong, interpretable baseline:
- It’s simple, well-understood, and fast.
- It often performs surprisingly well on structured tabular datasets.

**Minimum acceptable performance**: If your Random Forest, which is more complex, doesn’t outperform logistic regression, it signals that the additional complexity isn’t adding value. Logistic Regression is less sensitive to overfitting and is often more robust to noise. So if your model performs worse than logistic regression, it might be fragile.