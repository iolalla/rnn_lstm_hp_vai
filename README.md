# RNN LSTM Hyperparameter Tuning on Google Cloud Vertex AI

This repository contains a complete, production-ready pipeline for training Recurrent Neural Network (RNN) Long Short-Term Memory (LSTM) models on financial market data. It supports local training, local validation, and automated hyperparameter tuning using **Google Cloud Vertex AI (VAI)**.

---

## 🚀 Key Features

- **Flexible Execution**: Run training locally as a Python script or containerized, or scale up to Google Cloud Vertex AI with a single command.
- **Automated Hyperparameter Optimization**: Uses Vertex AI's `HyperparameterTuningJob` to optimize:
  - Learning rate (log-scaled search space)
  - Number of LSTM units
  - Dropout rates
  - Hidden and output layer activation functions
- **Model Provenance & Metadata**: Automatically collects and embeds rich system, git, and dataset metadata (including row/column counts, date ranges, and CRC32 checksums) directly into the trained `.h5` model file.
- **Enterprise-Grade Logging**: Logs all training runs, parameters, and evaluation metrics directly to **Google BigQuery** (`bolsa.trains`) for auditing and analysis.
- **Local Environment Optimization**: Seamless integration with Python's fast dependency manager `uv` and local datasets.

---

## 🛠️ Repository Structure

- `trainer/`: Contains the training package executed by Vertex AI.
  - [task.py](trainer/task.py): Main training entrypoint (handles data loading, normalization, sequence creation, model training, evaluation, metadata embedding, GCS upload, and BigQuery logging).
- [job.py](job.py): Orchestrates and submits the hyperparameter tuning job to Vertex AI.
- [model_metadata.py](model_metadata.py): Shared utility library for collecting metadata and embedding it into `.h5` files.
- [test_dataset.py](test_dataset.py): Diagnostic script to verify sequence generation and timeseries dataset compatibility.
- [Makefile](Makefile): Simplifies building, pushing, running local training, and submitting hyperparameter tuning jobs.
- [Dockerfile](Dockerfile): Container image definition for local and cloud training.

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
Create or verify `.env` in the root directory:
```env
PROJECT_ID=my-gcp-project
LOCATION=europe-west1
STAGING_BUCKET=gs://my-staging-bucket/vertex-ai
MODEL_BUCKET_NAME=my-model-bucket
IMAGE_URI=gcr.io/my-gcp-project/vertexai/rnn_lstm_vai:hypertune
SERVICE_ACCOUNT=vertex-ai-training@my-gcp-project.iam.gserviceaccount.com
BIGQUERY_DATASET=ml_training
MODEL_NAME=rnn_lstm_hp_model

# Optional
JOB_NAME=rnn-lstm-hp-tuning
FILEDATA=gs://my-staging-bucket/vertex-ai/data/train-dataset.csv
VAL_FILEDATA=gs://my-staging-bucket/vertex-ai/data/val-dataset.csv
RESUME_JOB_ID=
```

- `PROJECT_ID`: Google Cloud project ID.
- `LOCATION`: Vertex AI region.
- `STAGING_BUCKET`: GCS path for staging artifacts and trial outputs.
- `MODEL_BUCKET_NAME`: Destination GCS bucket for the best trained model.
- `IMAGE_URI`: Container image used for training trials.
- `SERVICE_ACCOUNT`: Email of the service account that runs the training job.
- `BIGQUERY_DATASET`: BigQuery dataset where training results are logged.
- `MODEL_NAME`: Name used for the model and BigQuery records.
- `JOB_NAME`: Optional display name for the tuning job (defaults to `rnn-lstm-hp-tuning`).
- `FILEDATA`: Optional path to the training CSV (defaults to `<STAGING_BUCKET>/data/train-dataset.csv`).
- `VAL_FILEDATA`: Optional path to the validation CSV (defaults to `<STAGING_BUCKET>/data/val-dataset.csv`).
- `RESUME_JOB_ID`: Optional existing Vertex AI hyperparameter tuning job ID to resume polling.

---

## 🏃 Running Training Locally

### Run Local Training Script
```bash
make run-local
```

---

## ☁️ Running Hyperparameter Tuning on Google Cloud Vertex AI

### 1. Build and Push Container Image
```bash
make cloud-build
```

### 2. Submit Hyperparameter Tuning Job
```bash
make job
```
This script initializes Vertex AI, submits a `HyperparameterTuningJob`, retrieves the best trial based on minimum MSE, embeds tuning metadata into the model, and copies the best model artifact to Google Cloud Storage.

---

## 📊 Model Provenance & Results Auditing

- **Vertex AI Console**: Track and compare tuning trial progress, metric charts, and parameters in real time.
- **BigQuery Logging**: Results are automatically appended to `bolsa.trains` in BigQuery for SQL auditing.


---

## 📝 Notes

If you want to see the model with best results you can use : 
```bash
RESUME_JOB_ID=4687416015661826048 uv run python3 job.py
```

