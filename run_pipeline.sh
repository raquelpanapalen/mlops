#!/bin/bash

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

# Run your ZenML pipeline
echo "🚀 Running the ZenML pipeline..."
python zenml_pipeline.py

echo "✅ Pipeline run completed."
