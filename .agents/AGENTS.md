# Project Rules and Guidelines - rnn_lstm_hp_vai

This document defines the project-specific rules, style guidelines, and development patterns for the `rnn_lstm_hp_vai` repository.

## 1. Development Stack & Environment
- **Runtime**: Python 3.13+ managed via `uv` or standard virtual environments (`.venv`).
- **Machine Learning**: TensorFlow 2.x and Keras.
- **Cloud Platform**: Google Cloud Platform (GCP) - Vertex AI for Hyperparameter Tuning and Custom Training Jobs.
- **Data Warehousing**: Google BigQuery (for logging training results/metrics) and Google Cloud Storage (for model artifacts and dataset hosting).

## 2. Python Coding Standards
- Follow PEP 8 style guidelines.
- Always include type hints where appropriate for readability and maintenance.
- Use explicit logging via the standard `logging` module rather than raw `print` statements (except in simple test scripts).
- Maintain robust exception handling, especially when interacting with cloud APIs (GCS, BigQuery, Vertex AI).

## 3. Vertex AI & Hyperparameter Tuning Patterns
- **Training Script (`trainer/task.py`)**:
  - Must accept hyperparameters as command-line arguments (using `fire` or `argparse`).
  - Must report the tuning metric (e.g., `mse`) to Vertex AI using the `hypertune` library:
    ```python
    import hypertune
    hpt_client = hypertune.HyperTune()
    hpt_client.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='mse',
        metric_value=val_mse,
        global_step=epochs
    )
    ```
  - Model saving: Save the trained model to `model.h5` locally, embed metadata, and then upload to GCS (either to `AIP_MODEL_DIR` if provided, or to a custom bucket path).
- **Orchestrator Script (`job.py`)**:
  - Configures and submits the `HyperparameterTuningJob` to Vertex AI.
  - Retrieves completed trials, identifies the best trial based on the target metric, and copies its model artifact to a "best model" path in GCS.

## 4. Model Metadata Provenance
- All trained models must have runtime and data provenance metadata embedded directly into the HDF5 file using `model_metadata.py`'s `embed_metadata_h5()` or written as a sidecar `.meta.json` file.
- The metadata must include:
  - Training timestamp.
  - Git commit hash and branch.
  - Python, TensorFlow, and Keras versions.
  - Dataset source URL and CRC32 checksum.
  - Dataset statistics (number of rows, columns, date range).
  - Hyperparameters used.
  - Final evaluation metrics (MSE, MAE, RMSE, Directional Accuracy).

## 5. BigQuery Results Logging
- Every successful training run (local or cloud-based) should log its results to the `play.trains` table in BigQuery.
- The table schema includes:
  - `job_id` (STRING, REQUIRED)
  - `trial_id` (STRING, NULLABLE)
  - `model_name` (STRING, REQUIRED)
  - `training_date` (TIMESTAMP, REQUIRED)
  - `parameters` (JSON, NULLABLE)
  - `metrics` (JSON, NULLABLE)
  - `model_path` (STRING, NULLABLE)
  - `git_commit` (STRING, NULLABLE)
  - `git_branch` (STRING, NULLABLE)
  - `dataset_file` (STRING, NULLABLE)
- When running locally, support connecting to a local BigQuery emulator via the `BIGQUERY_EMULATOR_HOST` environment variable.

## 6. Git Workflow
- Keep `.gitignore` updated to prevent committing local virtual environments (`.venv`), local configuration secrets (`.env`), temporary model files (`*.h5`, `model/`), and cache directories (`__pycache__/`, `.pytest_cache/`).
- Commit messages should be clear and descriptive, following the conventional commits specification where applicable (e.g., `feat:`, `fix:`, `docs:`, `chore:`).
