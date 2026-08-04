# RNN LSTM Hyperparameter Tuning on Google Cloud Vertex AI

This repository contains a complete, production-ready pipeline for training Recurrent Neural Network (RNN) Long Short-Term Memory (LSTM) models on financial market data. It supports local training, local validation, and fully-orchestrated cloud-based hyperparameter tuning using **Google Cloud Vertex AI (VAI)**, with automatic model provenance tracking and result logging.

---

## 🚀 Key Features

- **Flexible Execution**: Run training locally as a Python script or containerized, or scale up to Google Cloud Vertex AI with a single command.
- **Automated Hyperparameter Optimization**: Uses Vertex AI's `HyperparameterTuningJob` to optimize:
  - Learning rate (log-scaled search space)
  - Number of LSTM units
  - Dropout rates
  - Hidden and output layer activation functions
- **Model Provenance & Metadata**: Automatically collects and embeds rich system, git, and dataset metadata (including row/column counts, date ranges, and CRC32 checksums) directly into the trained `.h5` model file.
- **Enterprise-Grade Logging**: Logs all training runs, parameters, and final evaluation metrics (MSE, MAE, RMSE, Directional Accuracy) directly to **Google BigQuery** for auditing and analysis.
- **Local Environment Optimization**: Seamless integration with Python's fast dependency manager `uv` and local datasets.

---

## 🛠️ Repository Structure

- `trainer/`: Contains the training package executed by Vertex AI.
  - [task.py](rnn_lstm_hp_vai/trainer/task.py): Main training entrypoint (handles data loading, normalization, sequence creation, model training, evaluation, metadata embedding, GCS upload, and BigQuery logging).
- [job.py](rnn_lstm_hp_vai/job.py): Orchestrates and submits the hyperparameter tuning job to Vertex AI, tracks trials, identifies the best trial, and copies the best model to GCS.
- [model_metadata.py](rnn_lstm_hp_vai/model_metadata.py): Shared utility library for collecting metadata, embedding it into `.h5` files, and logging evaluation summaries.
- [test_dataset.py](rnn_lstm_hp_vai/test_dataset.py): Diagnostic script to verify sequence generation and timeseries dataset compatibility.
- [Makefile](rnn_lstm_hp_vai/Makefile): Simplifies building, pushing, running local training, and submitting jobs.
- [Dockerfile](rnn_lstm_hp_vai/Dockerfile): Container image definition for local and cloud training.

---

## 📦 Local Setup & Installation

### 1. Prerequisites
- Python 3.10+ (Python 3.13 recommended)
- [uv](https://github.com/astral-sh/uv) (fast Python package manager) or standard virtual environment tool.
- Docker (optional, for local container runs)

### 2. Install Dependencies
Initialize a virtual environment and install the required packages:
```bash
# Using uv (Recommended)
uv venv
uv pip install -r requirements.txt

# Or using standard pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory to configure your Google Cloud project and bucket names:
```env
PROJECT_ID=my-gcp-project-id
LOCATION=europe-west1
STAGING_BUCKET=gs://my-staging-bucket-hp
MODEL_BUCKET_NAME=my-model-bucket
IMAGE_URI=gcr.io/my-gcp-project-id/rnn_lstm_vai:hypertune
BIGQUERY_DATASET=ml_training
MODEL_NAME=rnn_lstm_hp_model
```

---

## 🏃 Running training locally

To quickly verify that the training script works and train a model on local data, use the preconfigured Makefile target:
```bash
make run-local-script
```
This runs the training script using `uv run python3 trainer/task.py` with:
- 2 epochs
- Hyperparameters: 32 units, relu activation, 0.1 dropout, linear output, 0.001 learning rate
- Dataset: `data/val-dataset-2021.csv`
- Ticker: `SAN.MC` (Banco Santander)

To run with custom parameters:
```bash
uv run python3 trainer/task.py \
  --epochs=5 \
  --learning_rate=0.0005 \
  --units=64 \
  --activation=tanh \
  --dropout_rate=0.2 \
  --filedata=data/val-dataset-2021.csv \
  --ticker=SAN.MC
```

---

## ☁️ Running on Google Cloud Vertex AI

### 1. Build and Push the Container Image

You can build and push your container image using either Google Cloud Build (no local Docker required) or local Docker:

#### Option A: Using Google Cloud Build (Recommended — No local Docker required)
If you do not have Docker installed locally, or prefer to build in the cloud, run:
```bash
make cloud-build
```
This builds your container image using Google Cloud Build and automatically pushes it to the registry path defined by `IMAGE_URI` in `.env`.

#### Option B: Using local Docker
If you have Docker running locally and prefer to build and push manually:
```bash
# Build the Docker image locally
make build

# Push the Docker image to the registry
make push
```

### 2. Submit the Hyperparameter Tuning Job
Submit the job to Vertex AI by running the orchestrator script:
```bash
make job
```
This script will:
1. Load configurations from your `.env` file.
2. Initialize the Vertex AI SDK.
3. Define the search space for hyperparameters (learning rate, units, activations, dropout).
4. Submit a `HyperparameterTuningJob` (runs up to 15 trials, 3 in parallel).
5. Poll the job until completion.
6. Identify the best trial based on the minimum Validation Mean Squared Error (MSE).
7. Copy the best trial's model file (`model.h5`) to a date-versioned GCS folder as `rnn_lstm_hp_vai/YYYY-MM-DD_HHMMSS/best_model.h5` and update the top-level pointer at `rnn_lstm_hp_vai/best_model.h5`.

---

## 📊 Model Provenance & Results Auditing

### Embedded Metadata
When a model finishes training, `model_metadata.py` embeds crucial metadata directly into the HDF5 file (`model.h5`). You can read this metadata at prediction time to guarantee reproducibility:
- **Git State**: Commit hash and branch name.
- **Dataset Stats**: Row counts, date range, and CRC32 checksum of the dataset.
- **Runtime Environment**: Python version, platform, TensorFlow, and Keras versions.
- **Evaluation Summary**: Final loss, MSE, MAE, RMSE, and Directional Accuracy.

### BigQuery Integration
Training results are automatically logged to the `trains` table inside your configured BigQuery dataset. This allows you to write SQL queries to compare different training runs and trials over time:
```sql
SELECT 
  job_id, 
  trial_id, 
  training_date, 
  JSON_VALUE(parameters, '$.ticker') as ticker,
  JSON_VALUE(metrics, '$.mse') as val_mse,
  JSON_VALUE(metrics, '$.directional_accuracy') as dir_accuracy,
  model_path 
FROM `my-gcp-project-id.ml_training.trains` 
ORDER BY training_date DESC;
```
