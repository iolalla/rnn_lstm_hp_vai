import os
import time
import datetime
import json
import logging
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

# Constants
PROJECT_ID = "banca-march"
LOCATION = "europe-west1"
STAGING_BUCKET = "gs://banca-march-models-hp"
MODEL_BUCKET_NAME = "banca-march-models"
IMAGE_URI = "gcr.io/banca-march/rnn_lstm_vai:hypertune"
JOB_NAME = f"ibex-rnn-lstm-hp-{int(time.time())}"

# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

# Define worker pool specs
worker_pool_specs = [
    {
        "machine_spec": {
            "machine_type": "n1-standard-4",
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": IMAGE_URI,
            "args": [
                "--epochs", "20",
                "--filedata", "https://storage.googleapis.com/ibex35/data/IBEX-1994-2020.csv",
                "--bucket_name", MODEL_BUCKET_NAME,
                "--job_id", JOB_NAME
            ]
        },
    }
]

# Create CustomJob
my_custom_job = aiplatform.CustomJob(
    display_name=f"{JOB_NAME}-custom-job",
    worker_pool_specs=worker_pool_specs,
)

# Define parameter specs for tuning
parameter_spec = {
    "learning_rate": hpt.DoubleParameterSpec(min=0.0001, max=0.01, scale="log"),
    "units": hpt.DiscreteParameterSpec(values=[32, 64, 128, 256, 512], scale=None),
    "activation": hpt.CategoricalParameterSpec(values=['relu', 'tanh', 'sigmoid', 'linear']),
    "dropout_rate": hpt.DoubleParameterSpec(min=0.1, max=0.5, scale="linear"),
    "activation_output": hpt.CategoricalParameterSpec(values=['relu', 'tanh', 'sigmoid', 'linear']),
}

# Define metric spec
metric_spec = {"mse": "minimize"}

# Create and run HyperparameterTuningJob
logging.info(f"Submitting Hyperparameter Tuning Job {JOB_NAME} to Vertex AI...")
hp_job = aiplatform.HyperparameterTuningJob(
    display_name=JOB_NAME,
    custom_job=my_custom_job,
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=15,
    parallel_trial_count=3,
)

hp_job.run()

logging.info("Hyperparameter Tuning Job completed successfully!")

# Retrieve trials and find the best one
trials = hp_job.trials
logging.info(f"Retrieved {len(trials)} trials from completed job.")

best_trial = None
best_mse = float("inf")

for trial in trials:
    # Check if trial succeeded
    if trial.state.name != "SUCCEEDED":
        logging.info(f"Trial {trial.id} state is {trial.state.name}, skipping.")
        continue
    
    # Extract metric value
    for metric in trial.final_measurement.metrics:
        if metric.metric_id == "mse":
            mse_val = metric.value
            logging.info(f"Trial {trial.id} completed with MSE: {mse_val}")
            if mse_val < best_mse:
                best_mse = mse_val
                best_trial = trial

if best_trial:
    logging.info("=" * 50)
    logging.info(f"BEST TRIAL IDENTIFIED: Trial {best_trial.id}")
    logging.info(f"Best MSE: {best_mse}")
    
    # Extract best parameters
    best_params = {}
    for param in best_trial.parameters:
        # Convert parameter value to appropriate type
        if param.parameter_id in ["units"]:
            best_params[param.parameter_id] = int(param.value)
        elif param.parameter_id in ["learning_rate", "dropout_rate"]:
            best_params[param.parameter_id] = float(param.value)
        else:
            best_params[param.parameter_id] = str(param.value)
            
    logging.info(f"Best Parameters: {best_params}")
    logging.info("=" * 50)
    
    # Copy the best trial's model to the final GCS path
    # Our task.py saves the model to: rnn_lstm_hp_vai/{job_id}/trial_{trial_id}/model.h5
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(MODEL_BUCKET_NAME)
    
    best_trial_model_blob_name = f"rnn_lstm_hp_vai/{JOB_NAME}/trial_{best_trial.id}/model.h5"
    final_model_blob_name = "rnn_lstm_hp_vai/best_model.h5"
    
    best_blob = bucket.blob(best_trial_model_blob_name)
    if best_blob.exists():
        logging.info(f"Copying best model from {best_trial_model_blob_name} to {final_model_blob_name}...")
        bucket.copy_blob(best_blob, bucket, final_model_blob_name)
        logging.info(f"Successfully saved best model to gs://{MODEL_BUCKET_NAME}/{final_model_blob_name}!")
    else:
        logging.warning(f"Could not find best trial's model blob at {best_trial_model_blob_name} inside bucket {MODEL_BUCKET_NAME}!")
else:
    logging.error("No successful trials found in the completed job!")
