# MLOps Project

## Environment setup

To set up the environment for running tests, follow these steps:

1. Create a virtual environment:
    ```bash
    python -m venv .venv
    ```

2. Activate the virtual environment:
    - On Linux/MacOS:
      ```bash
      source .venv/bin/activate
      ```
    - On Windows:
      ```bash
      .venv\Scripts\activate
      ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Data
- **Raw Data**: Located in the `data` folder.
- File [`metadata.md`](data/metadata.md) explains the data content.


## TASK 1

Once the environment is set up, you can run the pre-deployment tests using the following command:
```bash
bash run_tests.sh
```
A detailed explanation of all the choices for Task 1 can be found [here](docs/Task_1.md)

### Scripts
- **Notebook/data manager**: A Jupyter notebook (visualization purposes) and `data_manager.py` script for creating the golden set.
- **Task 1 Implementation**: The [`scripts/task1.py`](scripts/task1.py) file contains the Python implementation of all the tests.

## TASK 2 & 3

> 🛠️ **Note**: Task 2 has now been fully integrated into Task 3.  
> The updated implementation for Task 3 includes all functionality and objectives from Task 2.

- 📄 Detailed documentation for both tasks is available:
  - [Task 2 Documentation](docs/Task_2.md)
  - [Task 3 Documentation](docs/Task_3.md)

---

### 🚀 Running the Pipelines

Once the environment is set up, you can execute all the pipelines for Task 3 by running:

```bash
bash run.sh
```

This script orchestrates all required flows (pipelines) and includes post-deployment testing such as data drift detection, versioned inference, and A/B testing.


### 📁 Project Structure
- **Steps**: All modular step scripts used in the pipeline are located in `steps/`
- **Pipelines**: The various ZenML pipelines (flows) for Task 3 are implemented in the pipelines/ directory
- **Runner Script**: The `run.sh` script executes all pipelines in the correct sequence as required by Task 3


