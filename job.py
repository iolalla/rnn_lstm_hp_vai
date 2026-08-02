import os
import time
import datetime
import json
import logging
import subprocess
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google.cloud import storage
from google.auth.credentials import Credentials as BaseCredentials

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')

# Load .env file manually if it exists
loaded_env = {}
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.strip().startswith("#"):
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    key, val = parts
                    loaded_env[key.strip()] = val.strip().strip('"').strip("'")

# Set loaded env vars in os.environ
for k, v in loaded_env.items():
    os.environ[k] = v

# If GOOGLE_APPLICATION_CREDENTIALS is in os.environ but was NOT in .env,
# remove it so we don't accidentally use credentials from another project
if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ and "GOOGLE_APPLICATION_CREDENTIALS" not in loaded_env:
    logging.info("Clearing global GOOGLE_APPLICATION_CREDENTIALS to prevent project mismatch.")
    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

# Constants
PROJECT_ID = os.getenv("PROJECT_ID", "game-bolsa")
LOCATION = os.getenv("LOCATION", "europe-west1")
STAGING_BUCKET = os.getenv("STAGING_BUCKET", "gs://game-bolsa-models-hp")
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME", "game-bolsa-models")
IMAGE_URI = os.getenv("IMAGE_URI", "gcr.io/game-bolsa/rnn_lstm_vai:hypertune")
SERVICE_ACCOUNT = os.getenv("SERVICE_ACCOUNT")
JOB_NAME = f"ibex-rnn-lstm-hp-{int(time.time())}"

# Dataset and filtering configuration
FILEDATA = os.getenv("FILEDATA", f"{STAGING_BUCKET}/data/reall-complete-2000-2020.csv")
VAL_FILEDATA = os.getenv("VAL_FILEDATA", f"{STAGING_BUCKET}/data/reall-complete-IBEX-2021.csv")

# Define custom credentials class to automatically refresh gcloud access token
class GcloudCredentials(BaseCredentials):
    def __init__(self):
        super().__init__()
        self.token = None
        
    def refresh(self, request):
        logging.info("Refreshing gcloud access token...")
        try:
            self.token = subprocess.check_output(["gcloud", "auth", "print-access-token"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
            logging.info("Successfully refreshed gcloud access token.")
        except Exception as e:
            logging.error(f"Failed to refresh gcloud token: {e}")
            raise e

# Load credentials (first try standard GCP discovery, then fallback to custom gcloud class)
creds = None
try:
    import google.auth
    creds, default_project = google.auth.default()
    logging.info(f"Successfully loaded default Google Cloud credentials (Type: {type(creds).__name__}).")
except Exception as e:
    logging.warning(f"Failed to load default GCP credentials: {e}")

if not creds:
    try:
        logging.info("No standard credentials found. Using self-refreshing gcloud credentials fallback...")
        creds = GcloudCredentials()
    except Exception as e:
        logging.warning(f"Failed to initialize custom gcloud credentials: {e}. Falling back to default auth.")

# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET, credentials=creds)

# Build container arguments
container_args = [
    "--epochs", "20",
    "--filedata", FILEDATA,
    "--bucket_name", MODEL_BUCKET_NAME,
    "--job_id", JOB_NAME
]

if VAL_FILEDATA:
    container_args.extend(["--val_filedata", VAL_FILEDATA])

# Define worker pool specs
worker_pool_specs = [
    {
        "machine_spec": {
            "machine_type": "n1-standard-4",
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": IMAGE_URI,
            "args": container_args
        },
    }
]

# Check if we should resume an existing job
resume_job_id = os.getenv("RESUME_JOB_ID")
if resume_job_id:
    logging.info(f"Resuming polling for existing Hyperparameter Tuning Job: {resume_job_id}...")
    # Get the job resource name
    # If the user passed just the numeric ID, construct the full resource name
    if not resume_job_id.startswith("projects/"):
        resume_job_id = f"projects/{PROJECT_ID}/locations/{LOCATION}/hyperparameterTuningJobs/{resume_job_id}"
    hp_job = aiplatform.HyperparameterTuningJob.get(resume_job_id)
    # Use the existing job's display name as JOB_NAME
    JOB_NAME = hp_job.display_name
    logging.info(f"Loaded existing job display name: {JOB_NAME}")
    # Wait for the job to complete
    hp_job._block_until_complete()
else:
    # Create CustomJob with explicit base_output_dir to preserve trial artifacts
    base_output_dir = f"{STAGING_BUCKET}/hp_tuning/{JOB_NAME}"
    my_custom_job = aiplatform.CustomJob(
        display_name=f"{JOB_NAME}-custom-job",
        worker_pool_specs=worker_pool_specs,
        base_output_dir=base_output_dir,
    )

    # Define parameter specs for tuning
    parameter_spec = {
        "learning_rate": hpt.DoubleParameterSpec(min=0.0001, max=0.01, scale="log"),
        "units": hpt.DiscreteParameterSpec(values=[32, 64, 128, 256, 512], scale=None),
        "activation": hpt.CategoricalParameterSpec(values=['relu', 'tanh', 'sigmoid', 'linear']),
        "dropout_rate": hpt.DoubleParameterSpec(min=0.1, max=0.5, scale="linear"),
        "activation_output": hpt.CategoricalParameterSpec(values=['relu', 'tanh', 'sigmoid', 'linear']),
        "num_layers": hpt.DiscreteParameterSpec(values=[1, 2, 3,4], scale=None),
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
        max_trial_count=150,
        parallel_trial_count=3,
    )

    hp_job.run(service_account=SERVICE_ACCOUNT)

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
    storage_client = storage.Client(project=PROJECT_ID, credentials=creds)
    dest_bucket = storage_client.get_bucket(MODEL_BUCKET_NAME)
    final_model_blob_name = "rnn_lstm_hp_vai/best_model.h5"
    
    # Try to get the base output directory from the job object
    try:
        output_prefix = hp_job.gca_resource.trial_job_spec.base_output_directory.output_uri_prefix
        logging.info(f"Detected job output prefix: {output_prefix}")
        # Parse bucket and blob path from the prefix (e.g., gs://bucket-name/folder-path)
        if output_prefix.startswith("gs://"):
            output_prefix = output_prefix[5:]
        parts = output_prefix.split("/", 1)
        staging_bucket_name = parts[0]
        job_folder = parts[1] if len(parts) > 1 else ""
        
        best_trial_model_blob_name = f"{job_folder}/{best_trial.id}/model/model.h5"
        logging.info(f"Looking for model blob at gs://{staging_bucket_name}/{best_trial_model_blob_name}...")
        
        staging_bucket = storage_client.get_bucket(staging_bucket_name)
        best_blob = staging_bucket.blob(best_trial_model_blob_name)
    except Exception as e:
        logging.warning(f"Failed to get base_output_directory from job spec: {e}. Falling back to default naming convention.")
        # Fallback to default naming convention if proto structure is different
        staging_bucket_name = STAGING_BUCKET.replace("gs://", "")
        staging_bucket = storage_client.get_bucket(staging_bucket_name)
        best_trial_model_blob_name = f"hp_tuning/{JOB_NAME}/{best_trial.id}/model/model.h5"
        best_blob = staging_bucket.blob(best_trial_model_blob_name)
        logging.info(f"Looking for model blob at gs://{staging_bucket_name}/{best_trial_model_blob_name}...")
    
    if best_blob.exists():
        logging.info(f"Processing and downloading best model from gs://{staging_bucket_name}/{best_blob.name}...")
        import tempfile
        from model_metadata import read_metadata, embed_metadata_h5
        
        # Download best blob to a local temp file
        local_temp_model = tempfile.mktemp(suffix=".h5")
        try:
            best_blob.download_to_filename(local_temp_model)
            logging.info(f"Downloaded best model to local path: {local_temp_model}")
            
            # Read existing metadata from the file
            try:
                meta = read_metadata(local_temp_model)
                logging.info("Successfully read existing model metadata.")
            except Exception as e:
                logging.warning(f"Failed to read existing metadata: {e}. Starting with empty metadata.")
                meta = {}
                
            # Update metadata with tuning-specific details
            meta['tuning_job_id'] = JOB_NAME
            meta['best_trial_id'] = best_trial.id
            meta['best_mse'] = best_mse
            meta['is_best_model'] = True
            meta['best_parameters'] = best_params
            
            # Embed updated metadata back into the file
            embed_metadata_h5(local_temp_model, meta)
            
            # Upload the modified file back to GCS
            logging.info(f"Uploading updated best model to gs://{MODEL_BUCKET_NAME}/{final_model_blob_name}...")
            dest_blob = dest_bucket.blob(final_model_blob_name)
            dest_blob.upload_from_filename(local_temp_model)
            logging.info(f"Successfully saved updated best model to gs://{MODEL_BUCKET_NAME}/{final_model_blob_name}!")
        finally:
            # Clean up local temp file
            if os.path.exists(local_temp_model):
                os.remove(local_temp_model)
    else:
        logging.warning(f"Could not find best trial's model blob at gs://{staging_bucket_name}/{best_blob.name}!")
else:
    logging.error("No successful trials found in the completed job!")
