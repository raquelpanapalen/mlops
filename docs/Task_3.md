# Task 3
Firstly, we outline how different data splits were used across the training and evaluation lifecycle and do a recap of Task 1 & 2.

Additionally, this document explains the rationale and strategy behind post-deployment testing, including data drift detection, version-controlled model flows, and offline A/B testing.


## 🪙 Golden Set vs. Current Set
We began by intentionally splitting the full dataset into two parts:

- **Golden Set (Reference Set)**:
    - A clean, curated subset of the dataset.
    - High-quality and trusted data used for training and evaluation.
    - Serves as the **reference** for both model training and drift analysis.

- **Current Set (Unseen or Future Set)**:
    - The remaining data from the full dataset.
    - Represents **production-like or future data** not seen during training.
    - Used for **post-deployment validation**, such as drift detection and A/B testing.

## ✅ Tasks 1 & 2 – Pre-Deployment: Data Validation & Model Training

### 🔍 Data Validation
- Compared **Golden Set** (reference) with the **Current Set** (candidate).
- Checks performed:
  - Schema consistency
  - Missing values
  - Statistical feature drift
- Ensures the model won’t break on production data due to structure or distribution mismatches.

### 🧠 Model Training
- Performed a train/test split **within the Golden Set**:
  - `golden_train` → used to train the model.
  - `golden_test` → used to evaluate the model.
- The **Current Set** was **not used during training**, ensuring a clean separation for unbiased evaluation.

> ✅ This is a best-practice approach because:
> - Models learn from high-quality data only.
> - No leakage of future drifted behavior into training.
> - Evaluation on `golden_test` ensures trustworthy baseline metrics.

---

## 📈 Post-Deployment Testing


### 📊 Data Drift Detection

In our post-deployment pipeline, **data drift is assessed per feature**, not just at the dataset level. This is a more granular and interpretable approach that allows us to reason about which specific features are changing, rather than treating the dataset as a monolith.

We use the [`evidently`](https://docs-old.evidentlyai.com/reference/data-drift-algorithm) library, which follows a well-defined methodology depending on the data type and cardinality:

| Feature Type                                | Drift Metric Used              | Threshold |
|---------------------------------------------|--------------------------------|-----------|
| Numerical with > 5 unique values             | **Wasserstein Distance**       | 0.1       |
| Categorical or numerical with ≤ 5 categories | **Jensen–Shannon Divergence**  | 0.1       |

- **Thresholds**: A feature is considered drifted if the distance metric exceeds **0.1**.
- **Sample size**: These rules apply when the reference dataset (our `golden_train`) has **> 1000 samples**, which it does in our case.
- **Reference**: `golden_train` (data the model was trained on).
- **Current**: `current_set` (simulating real-world production input).

We ran two key `evidently` tests:

1. `TestNumberOfDriftedColumns`
   - Counts how many individual features exceeded the drift threshold.
   - Highlights *which* features changed, useful for debugging.

2. `TestShareOfDriftedColumns`
   - Computes the proportion of drifted columns in the dataset.
   - If this share is high (by default > 50%), the dataset as a whole is flagged as "drifted"

> ✅ This approach sets up a **rule-based aggregation on top of individual feature-level drift results**, providing both local (column-level) and global (dataset-level) drift insights.


# 🏷️ Model Versioning via Flow ID
- Each training run is associated with a `flow_version_id`
- Models are retrieved from MLflow using this ID
- This allows:
    - Accurate tracking of how each model was built
    - Side-by-side comparisons of models trained with different parameters, architectures, or preprocessing
    - Seamless rollback to prior working versions

It’s a reproducibility safeguard — critical in regulated or high-risk environments.


### 🔁 A/B Testing
- **Split logic**: ID % 2
    - Ensures deterministic, reproducible splits using passenger ID
    - Avoids leakage: Each user is consistently routed to the same variant

- **Evaluation**:
    - Performance is measured on both A and B using unseen data (`current_set` split)
    - Accuracy and F1 score are computed to capture both correctness and class balance

- **Tracking**:
    - Results logged to MLflow under a parent A/B test run
    - Nested runs capture metrics for each variant
    - A JSON summary is saved as an artifact for audit and decision-making

This offline A/B test approximates online testing, enabling safe model evaluation before full production rollout.


## Summary Table
| Task            | Input Data                          | Purpose                                 | When / Where                                 |
| --------------- | ----------------------------------- | --------------------------------------- | -------------------------------------------- |
| Data Validation | Golden vs Current                   | Schema/missing/categorical issues       | Pre-deployment pipeline                      |
| Model Training  | Golden (split into train/test)      | Train trusted model                     | Pre-deployment pipeline                      |
| Drift Test      | Golden (train set) vs Current       | Monitor for data shifts post-deployment | Monitoring pipeline (after model deployment) |
| A/B Testing     | Current (split A/B on Passenger ID) | Compare model versions                  | Separate pipeline                            |
