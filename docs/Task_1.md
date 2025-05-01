# TASK 1

## 🔍 Dataset Overview
The dataset contains 129,880 rows and includes:
- Demographics (e.g., Gender, Age),
- Customer metadata (e.g., Customer Type, Class, Type of Travel),
- Satisfaction ratings (1–5 or 0 = not applicable),
- Delay information (departure and arrival delay),
- Binary satisfaction label (Satisfied, Neutral or unsatisfied).

## 1.1.  Constructing a Golden Test Set for Airline Passenger Satisfaction Dataset
To fairly and robustly evaluate machine learning models trained on airline customer satisfaction data, we construct a golden test set. This set serves as a reliable benchmark that:

- Represents typical behavior (expected usage),
- Captures rare or edge cases (stress tests),
- Maintains demographic and behavioral diversity,
- Ensures label fairness across satisfaction levels.


⚠️ No timestamps or temporal data are available, so temporal holdout strategies (e.g., splitting the latest data as test) are not possible. Thus, we opt for a stratified and scenario-based sampling approach, ensuring that the test set fairly reflects real-world diversity and edge scenarios.


### Step 1: Define Stratification Columns
To fairly represent core behavioral and demographic segments, we stratify the data along the following columns:

| Column         | Reason for Inclusion                                                                 |
|----------------|----------------------------------------------------------------------------------------|
| Satisfaction   | Ensure both satisfied and dissatisfied customers are tested.                          |
| Customer Type  | Returning vs first-time travelers may behave differently.                             |
| Type of Travel | Personal vs business travel may affect satisfaction and expectations.                 |
| Class          | Higher-paying customers may be more critical or have different expectations.          |
| Gender         | Check model generalization across demographic groups.                                 |


#### 👉 Why combine Eco and Eco Plus?
While these are technically different fare classes, they reflect similar "economy-level" experience. For test coverage purposes, we merge them to prevent data sparsity during stratification and focus on meaningful class separations (Economy vs Business).

#### 👉 Why stratified sampling?
We use stratified sampling to draw a balanced number of examples from each unique combination of values from the selected columns. This ensures that every meaningful user group is represented, and the model will be evaluated across the full behavioral spectrum.


### Step 2: Add Edge Cases
Beyond the balanced stratified set, we manually include edge cases to test robustness:

| Edge Case                  | Description                                                                                          |
|----------------------------|------------------------------------------------------------------------------------------------------|
| Extreme delays             | Flights with arrival or departure delays > 3 hours. This threshold is legally relevant for compensation. |
| Very short or long flights | E.g., < 100 miles or > 4,000 miles. Covers operational edge cases.                                  |
| All <=2 ratings            | Simulate very dissatisfied customers.                                                               |
| Age extremes               | Very young (e.g., <18) or elderly passengers (e.g., >75). The <18 condition is legally relevant because they shouldn't be allowed to take a survey.                                           |

Real-world users include a wide variety of situations and demographics. If your model only works for typical or majority cases, it's not fair or useful. Edge cases like young travelers, elderly, or extremely dissatisfied customers ensure the model works for everyone.

### 📌 Comparison against other techniques
- **Why not random sampling?** Random splits often overrepresent majority classes and underrepresent rare combinations.

- **Why not temporal holdout?** The dataset lacks timestamps, so we cannot simulate a "future" or "live" setting for evaluation.

- **Why not just a test set from training split?** Typical train-test splits often don’t expose a model's weaknesses in edge scenarios.


## 1.2. Data Distribution Tests

Ensuring that data values fall within reasonable and expected ranges is critical for data quality, model stability, and trustworthy analytics. Below is a breakdown of the attributes tested, why each is important, and how the acceptable range was determined.


| **Column**                             | **Why Test This?**                                                                                                                  | **Expected Range** | **Reasoning Behind the Range**                                                                                          |
|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|--------------------|--------------------------------------------------------------------------------------------------------------------------|
| `Age`                                 | Important for identifying customer personas; extreme values might indicate data errors.                                              | 18 to 100          | To take a passenger survey you have to be an adult; ages over 100 are rare and may signal incorrect entries.                    |
| `Flight Distance`                     | Reflects route length, affecting service expectations and satisfaction.                                                              | 50 to 9500 miles   | Most commercial flights are longer than 50 miles, and 9500 is near the global max for non-stop flights.                 |
| `Departure Delay`                     | Major factor in customer dissatisfaction; extreme values may be invalid.                                                             | 0 to 2880 minutes  | 2 days of delay is an upper bound that covers even severe cases like cancellations or rebookings.                       |
| `Arrival Delay`                       | Same as above; important for validating punctuality-related insights.                                                                | 0 to 2880 minutes  | Same logic as departure delay.                                                                                           |

Additionally, we check that all the ratings columns (e.g., Seat comfort, Food and drink, Gate location) contain values between 0 and 5, ensuring no invalid ratings.

