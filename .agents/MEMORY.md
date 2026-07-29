# Project Memory & Knowledge Base - rnn_lstm_hp_vai

This file serves as a persistent memory and project state tracker for the `rnn_lstm_hp_vai` repository. It is updated continuously as the project progresses to ensure seamless handoffs and consistent design decisions.

## 1. Project Overview
`rnn_lstm_hp_vai` is a machine learning pipeline designed to train Recurrent Neural Network (RNN) Long Short-Term Memory (LSTM) models on financial market data (such as the Spanish IBEX 35 index or individual stock tickers). 
The training and hyperparameter optimization are orchestrated using **Google Cloud Vertex AI (VAI)**, with results logged to **Google BigQuery** and model artifacts stored in **Google Cloud Storage (GCS)**.

### Key Capabilities:
- **Local & Cloud Training**: Supports running training containers locally (with a BigQuery emulator) or submitting jobs to Vertex AI.
- **Hyperparameter Tuning**: Utilizes Vertex AI's `HyperparameterTuningJob` to optimize key parameters (learning rate, LSTM units, activation functions, dropout rate).
- **Model Provenance**: Embeds comprehensive training and dataset metadata directly into the trained model HDF5 file (`.h5`) or a sidecar JSON file.
- **Automated Logging**: Logs all training trials, hyperparameters, and evaluation metrics to a BigQuery dataset (`play.trains`) for analysis and auditability.

## 2. Repository Structure
- `trainer/`: Contains the training package executed by Vertex AI.
  - [task.py](file:///home/iolalla/src/rnn_lstm_hp_vai/trainer/task.py): The main training entrypoint. Handles data loading, normalization, sequence creation, model training, evaluation, metadata embedding, GCS upload, and BigQuery logging.
- [job.py](file:///home/iolalla/src/rnn_lstm_hp_vai/job.py): Orchestrates and submits the hyperparameter tuning job to Vertex AI, tracks completed trials, finds the best trial, and copies the best model to a designated GCS path.
- [model_metadata.py](file:///home/iolalla/src/rnn_lstm_hp_vai/model_metadata.py): Shared utility library for collecting system/dataset metadata, embedding it into `.h5` files, and logging evaluation summaries.
- [test_dataset.py](file:///home/iolalla/src/rnn_lstm_hp_vai/test_dataset.py): A test script to verify that the dataset sequence creation and timeseries generation work as expected.
- [Makefile](file:///home/iolalla/src/rnn_lstm_hp_vai/Makefile): Simplifies common tasks like building/pushing docker images, submitting jobs, and running local training.
- [Dockerfile](file:///home/iolalla/src/rnn_lstm_hp_vai/Dockerfile): Defines the container image used for both local training and Vertex AI training.
- [requirements.txt](file:///home/iolalla/src/rnn_lstm_hp_vai/requirements.txt): Python dependencies.
- [data/](file:///home/iolalla/src/rnn_lstm_hp_vai/data/): Directory containing local CSV datasets for development and testing.
- [.agents/](file:///home/iolalla/src/rnn_lstm_hp_vai/.agents/): Agent workspace customizations directory.
  - [AGENTS.md](file:///home/iolalla/src/rnn_lstm_hp_vai/.agents/AGENTS.md): Project-scoped rules and development guidelines.
  - [MEMORY.md](file:///home/iolalla/src/rnn_lstm_hp_vai/.agents/MEMORY.md): This file (project state, progress, and roadmap).

## 3. Current Project State

### Completed Actions:
1. **Git Repository Initialized**: Set up a local git repository.
2. **Workspace Customizations Folder Set Up**: Created `.agents/` directory with `AGENTS.md` and `MEMORY.md`.
3. **Virtual Environment Fixes**: Resolved file permissions on the local virtual environment (`.venv/bin`) to allow execution of python and helper binaries.
4. **Local Dataset Discovery**: Discovered a rich set of local CSV datasets in the `data/` folder, including `reall-complete-IBEX-2021.csv` and ticker-specific files like `reall-SAN-2000-2021.csv`.
5. **Test Script Enhancement**: Updated `test_dataset.py` to utilize the local `data/reall-complete-IBEX-2021.csv` dataset and filter for a single ticker (`SAN.MC`) to prevent mixing different stock price series.
6. **Import Path Correction in `task.py`**: Fixed an issue where `sys.path` insertion was going up three levels instead of two, causing `ModuleNotFoundError: No module named 'model_metadata'`.
7. **Robust Ticker Support Added**: Added a `ticker` command-line argument to `trainer/task.py` to filter multi-ticker datasets (like `reall-complete-IBEX-2021.csv`) before training. If no ticker is specified, it defaults to the first ticker in the dataset.
8. **Environment Variable Integration in `job.py`**: Modified `job.py` to automatically load settings from the local `.env` file using `os.getenv` with sensible defaults. This prevents hardcoded configs from overriding local configuration settings. Also added support for loading and passing `SERVICE_ACCOUNT` to the Vertex AI `HyperparameterTuningJob` run method so that training trials run with correct GCP permissions.
9. **Convenient Local Run & Env Integration in Makefile**: Added a `run-local-script` target to the `Makefile` to run the training script locally using python and `uv` with the local dataset. Also updated the `Makefile` to automatically load and export environment variables from the local `.env` file if it exists, ensuring consistent image URIs and project configurations across build, push, and job submission commands.
10. **Dockerfile Customization**: User added `ENV GCLOUD_PROJECT=banca-march` to the `Dockerfile` to default the GCP project name when running inside the container. This change has been committed to git.
11. **Local Configurations Updated**: User updated the local `.env` file to set `GCLOUD_PROJECT="banca-march-379915"`, `PROJECT_ID="banca-march-379915"`, and configure the `SERVICE_ACCOUNT` value.

### Verification Status:
- **test_dataset.py**: Successfully ran and verified that TensorFlow's `timeseries_dataset_from_array` creates proper input sequences and target labels.
- **trainer/task.py (Local Script Run)**: Successfully executed local training runs (1 and 2 epochs) using the local dataset and the new ticker filtering option. Saved the model to `model.h5` and embedded metadata successfully.
  - **Epochs**: 2
  - **Loss (MSE)**: `0.2407`
  - **Directional Accuracy**: `48.59%`
  - *Note*: Both the agent and the user have executed this target successfully in the local shell.

## 4. Architectural Diagram
```mermaid
graph TD
    subgraph Local Environment
        DS[Local CSV Data] --> |Filter Ticker| TD[test_dataset.py]
        TD --> |Verify Sequences| TF[TensorFlow Dataset]
        M[Makefile] --> |Run Script| TS[trainer/task.py]
        TS --> |Train Model| L[Local model.h5]
        TS --> |Embed Metadata| L
        M --> |Build/Push| DK[Docker Image]
    end
    subgraph Google Cloud Platform (GCP)
        DK --> |Host Image| GCR[Google Container Registry]
        J[job.py] --> |Submit Job| VAI[Vertex AI Hyperparameter Tuning]
        GCR --> |Execute Container| VAI
        VAI --> |Train Model| T[trainer/task.py]
        T --> |Report Metrics| VAI
        T --> |Upload Models| GCS[Google Cloud Storage]
        T --> |Log Results| BQ[Google BigQuery]
    end
```

## 5. Next Steps & Roadmap
1. **Review Dockerfile**: Ensure the Dockerfile is optimized and includes all necessary files (completed, user added `ENV GCLOUD_PROJECT=banca-march`).
2. **Commit and Prepare for Push**:
   - Stage all relevant files (completed).
   - Create a clean git commit history (completed).
   - Set up instructions for pushing to GitHub (completed).
3. **Verify job.py Configurations**:
   - Check if `job.py` constants match the user's active GCP project and bucket configurations.
   - Ensure credentials and environment variables are set up correctly.
