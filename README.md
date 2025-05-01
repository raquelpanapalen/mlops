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
- **Notebook/Utils**: A Jupyter notebook (visualization purposes) and utils script for creating the golden set.
- **Task 1 Implementation**: The [`scripts/task1.py`](scripts/task1.py) file contains the Python implementation of all the tests.

## TASK 2

Once the environment is set up, you can run the Task 2 pipeline using the following command:
```bash
bash run_pipeline.sh
```

A detailed explanation of all the choices for Task 2 can be found [here](docs/Task_2.md)

### Scripts

- **Steps**: You can find all the different step scripts in the folder [`steps`](steps/)
- **Task 2 Implementation**: The main [`zenml_pipeline.py`](zenml_pipeline.py) file contains the Python implementation that runs the ZenML pipeline.


