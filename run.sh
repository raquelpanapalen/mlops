#!/bin/bash
set -e # Exit on error

# Connect to ZenML
echo "🔗 Connecting to ZenML..."
zenml login --local

# Set up paths
MLFLOW_BACKEND_URI="sqlite:///mlflow.db"
MLFLOW_ARTIFACT_URI="./mlruns"
MLFLOW_PORT=5000

# Start the MLflow server in the background
echo "🔧 Starting MLflow server on http://localhost:$MLFLOW_PORT ..."
mlflow server \
  --backend-store-uri $MLFLOW_BACKEND_URI \
  --default-artifact-root $MLFLOW_ARTIFACT_URI \
  --host 127.0.0.1 \
  --port $MLFLOW_PORT > mlflow_server.log 2>&1 &

# Save MLflow server PID
MLFLOW_PID=$!
echo "MLflow server started with PID $MLFLOW_PID"

# Wait a bit to ensure MLflow server starts
sleep 5

# Set the MLFLOW_TRACKING_URI so your code uses the local server
export MLFLOW_TRACKING_URI="http://127.0.0.1:$MLFLOW_PORT"
echo "🔗 MLflow tracking URI set to $MLFLOW_TRACKING_URI"

# Create golden (reference) / train / test datasets
echo "📊 Creating datasets..."
python data_manager.py

flow1="v1"
flow2="v2"

# Run your ZenML pipeline
echo "🚀 Running training pipeline (version 1)..."
python train_pipeline.py --flow-version "$flow1" --max_depth 5 --n_estimators 100

echo "🚀 Running training pipeline (version 2)..."
python train_pipeline.py --flow-version "$flow2" --max_depth 10 --n_estimators 200

echo "🔍 Running monitoring (drift detection) pipeline..."
python monitor_pipeline.py

echo "📈 Running A/B test..."
test_id="${flow1}_${flow2}_$(date +%s)"
python ab_pipeline.py --flow_version_a="$flow1" --flow_version_b="$flow2" --test_id="$test_id"
