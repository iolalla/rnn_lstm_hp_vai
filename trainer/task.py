#!/usr/bin/env python3
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import fire
import warnings
import time
import pandas as pd
import numpy as np
import gc
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Input
from sklearn.preprocessing import MinMaxScaler
import logging
import pickle
import glob
import re
import json

# Add parent directories to path to import model_metadata
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import h5py
from model_metadata import build_training_metadata, embed_metadata_h5, log_and_evaluate_supervised, serialize_scaler_map, crc32_file, _data_stats

from google.cloud import storage
from google.cloud import bigquery
from google.api_core.client_options import ClientOptions
from google.cloud.exceptions import NotFound
import google.auth
import hypertune

warnings.filterwarnings('ignore')

def get_bigquery_client(project: str = None):
    if not project:
        project = os.getenv("PROJECT_ID", os.getenv("GCLOUD_PROJECT", os.getenv("GCP_PROJECT")))
    is_local = os.getenv("ISLOCAL", "false").lower() == "true"
    if is_local:
        endpoint = os.getenv("BIGQUERY_EMULATOR_HOST", "http://localhost:9050")
        if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
            endpoint = f"http://{endpoint}"
        client_options = ClientOptions(api_endpoint=endpoint)
        from google.auth.credentials import AnonymousCredentials
        return bigquery.Client(
            project=project or "test-project",
            client_options=client_options,
            credentials=AnonymousCredentials()
        )
    else:
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and os.path.exists(creds_path):
            return bigquery.Client.from_service_account_json(creds_path, project=project)
        else:
            return bigquery.Client(project=project)

def ensure_trains_table_exists(client: bigquery.Client, dataset_id: str = "ml_training"):
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "EU"
        client.create_dataset(dataset)
        print(f"Created dataset {dataset_id}")

    table_ref = dataset_ref.table("trains")
    try:
        client.get_table(table_ref)
    except NotFound:
        schema = [
            bigquery.SchemaField("job_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("trial_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("model_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("training_date", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("parameters", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("metrics", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("model_path", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("git_commit", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("git_branch", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("dataset_file", "STRING", mode="NULLABLE"),
        ]
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
        print(f"Created table {dataset_id}.trains")

def insert_train_result(client: bigquery.Client, row_data: dict, dataset_id: str = "ml_training"):
    table_ref = client.dataset(dataset_id).table("trains")
    errors = client.insert_rows_json(table_ref, [row_data])
    if errors:
        raise RuntimeError(f"Failed to insert row into BigQuery: {errors}")

def upload_to_bucket(blob_name, path_to_file, bucket_name):
    """ Upload data to a bucket"""
    creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if creds_path and os.path.exists(creds_path):
        storage_client = storage.Client.from_service_account_json(creds_path)
    else:
        storage_client = storage.Client()
    # bucket_name may include a path prefix (e.g. "my-bucket/some/prefix");
    # GCS bucket names cannot contain slashes, so split it out here.
    bucket_name = bucket_name.replace("gs://", "").strip("/")
    bare_bucket_name, _, prefix = bucket_name.partition("/")
    if prefix:
        blob_name = f"{prefix}/{blob_name}"
    bucket = storage_client.get_bucket(bare_bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path_to_file)
    return blob.public_url

def build_model(units, activation, dropout_rate, activation_output, learning_rate, num_layers=1):
    """Creating LSTM model"""
    input_shape = (10, 1)
    model = Sequential()
    model.add(Input(shape=input_shape))
    for i in range(num_layers):
        return_seq = (i < num_layers - 1)
        model.add(Bidirectional(
            LSTM(
                units=units,
                activation=activation,
                return_sequences=return_seq
            )
        ))
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation=activation_output))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    return model

def create_sequences(data: pd.DataFrame, seq_length: int) -> tuple:
    xs, ys = [], []
    for i in range(len(data) - seq_length - 9):  # Asegura 10 días adicionales
        x = data.iloc[i:(i + seq_length)]
        y = data.iloc[i + seq_length: i + seq_length + 10]  # 10 días futuros
        xs.append(x.values)
        ys.append(y.values.flatten())  # Aplana para forma (10,)
    return np.array(xs), np.array(ys)

def train_evaluate(filedata=None,
                   modelpath=None,
                   epochs=10,
                   learning_rate=0.001,
                   units=50,
                   activation="relu",
                   dropout_rate=0.1,
                   activation_output="linear",
                   num_layers=1,
                   job_id=None,
                   trial_id=None,
                   bucket_name=None,
                   ticker=None,
                   val_filedata=None,
                   bq_dataset=None,
                   model_name=None,
                   project_id=None
                   ):
    # Resolve configuration from environment if not explicitly passed as arguments
    if not project_id:
        project_id = os.getenv("PROJECT_ID", os.getenv("GCLOUD_PROJECT"))
    if not filedata:
        filedata = os.getenv("FILEDATA", "data/reall-complete-2000-2020.csv")
    if not val_filedata:
        val_filedata = os.getenv("VAL_FILEDATA")
    if not modelpath:
        modelpath = os.getenv("MODEL_PATH", "model/rnn_lstm_hp_model.h5")
    if not bucket_name:
        bucket_name = os.getenv("MODEL_BUCKET_NAME", "my-model-bucket")
    if not bq_dataset:
        bq_dataset = os.getenv("BIGQUERY_DATASET", "ml_training")
    if not model_name:
        model_name = os.getenv("MODEL_NAME", "rnn_lstm_hp_model")

    # Set job_id and trial_id from env if not passed
    if not job_id:
        job_id = os.getenv("AIP_JOB_ID", "local_job_" + str(round(time.time())))
    if not trial_id:
        trial_id = os.getenv("AIP_TRIAL_ID")

    logging.info("Tensorflow version " + tf.__version__)
    
    # Save original paths for metadata
    original_filedata = filedata
    original_val_filedata = val_filedata

    # Helper function to download from GCS if needed
    def download_from_gcs_if_needed(path, local_path):
        if path.startswith("gs://"):
            logging.info(f"Downloading data from GCS path: {path}...")
            from google.cloud import storage
            path_parts = path[5:].split("/", 1)
            b_name = path_parts[0]
            blob_name = path_parts[1]
            storage_client = storage.Client()
            bucket = storage_client.bucket(b_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            logging.info(f"Successfully downloaded to local path: {local_path}")
            return local_path
        return path

    # Download datasets if they are GCS paths
    local_filedata = download_from_gcs_if_needed(filedata, "/tmp/train_dataset.csv")
    local_val_filedata = None
    if val_filedata:
        local_val_filedata = download_from_gcs_if_needed(val_filedata, "/tmp/val_dataset.csv")

    # If ticker is not specified, determine a smart default ticker
    if not ticker:
        try:
            # Read only Ticker column to find unique tickers quickly
            train_cols = pd.read_csv(local_filedata, nrows=1).columns
            if 'Ticker' in train_cols:
                train_tickers = set(pd.read_csv(local_filedata, usecols=['Ticker'])['Ticker'].unique())
                
                # If validation file is provided, check its tickers too
                if local_val_filedata:
                    val_cols = pd.read_csv(local_val_filedata, nrows=1).columns
                    if 'Ticker' in val_cols:
                        val_tickers = set(pd.read_csv(local_val_filedata, usecols=['Ticker'])['Ticker'].unique())
                        common_tickers = train_tickers.intersection(val_tickers)
                    else:
                        common_tickers = set()
                else:
                    common_tickers = train_tickers
                
                # Choose the best default ticker
                if 'SAN.MC' in common_tickers:
                    ticker = 'SAN.MC'
                    logging.info("No ticker specified. Defaulting to 'SAN.MC' (found in both datasets).")
                elif common_tickers:
                    ticker = sorted(list(common_tickers))[0]
                    logging.info(f"No ticker specified. Defaulting to first common ticker: '{ticker}'.")
                else:
                    ticker = sorted(list(train_tickers))[0]
                    logging.info(f"No ticker specified. Defaulting to first training ticker: '{ticker}'.")
        except Exception as e:
            logging.warning(f"Failed to determine smart default ticker: {e}. Falling back to standard behavior.")

    # Helper function to load and preprocess a dataset
    def load_and_preprocess(filepath, ticker_to_filter, label="dataset"):
        logging.info(f"Loading {label} from: {filepath}...")
        df = pd.read_csv(filepath)
        
        # Filter by ticker if 'Ticker' column exists and ticker_to_filter is provided
        if 'Ticker' in df.columns and ticker_to_filter:
            df = df[df['Ticker'] == ticker_to_filter]
            logging.info(f"Filtered {label} for ticker: {ticker_to_filter}")
        
        df = df[['Date', 'Close']]  # Extracting required columns
        df.dropna(inplace=True)
        
        # Try multiple date formats
        try:
            df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.split()[0]), format='%Y-%m-%d')
        except Exception:
            try:
                df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.split()[0]), format='%d/%m/%Y')
            except Exception:
                df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.split()[0]))
                
        df.set_index('Date', drop=True, inplace=True)
        df.sort_index(inplace=True)
            
        return df

    # Load and preprocess training data
    train_df = load_and_preprocess(local_filedata, ticker, "training data")

    # Load and preprocess validation data if provided
    if local_val_filedata:
        val_df = load_and_preprocess(local_val_filedata, ticker, "validation data")
    else:
        val_df = None

    # Normalization
    mms = MinMaxScaler()
    if val_df is not None:
        train_data = train_df.copy()
        test_data = val_df.copy()
        train_data[['Close']] = mms.fit_transform(train_data[['Close']])
        test_data[['Close']] = mms.transform(test_data[['Close']])
    else:
        # Fallback to 80/20 split of train_df
        training_size = round(len(train_df) * 0.80)
        train_data = train_df[:training_size].copy()
        test_data = train_df[training_size:].copy()
        train_data[['Close']] = mms.fit_transform(train_data[['Close']])
        test_data[['Close']] = mms.transform(test_data[['Close']])

    train_seq, train_label = create_sequences(train_data, 10)
    test_seq, test_label = create_sequences(test_data, 10)

    # Build model
    model_rnn = build_model(
        units=int(units),
        activation=activation,
        dropout_rate=float(dropout_rate),
        activation_output=activation_output,
        learning_rate=float(learning_rate),
        num_layers=int(num_layers)
    )

    # Train model
    model_rnn.fit(train_seq,
                  train_label,
                  epochs=int(epochs),
                  validation_data=(test_seq, test_label),
                  verbose=1
                  )

    logging.info(model_rnn.summary())
    
    # Evaluate model
    local_model_path = "model.h5"
    model_rnn.save(local_model_path)
    
    final_metrics = log_and_evaluate_supervised(model_rnn, test_seq, test_label, local_model_path, train_seq=train_seq)
    
    # Report metric to Vertex AI Hypertune
    val_mse = final_metrics.get('mse', 0.0)
    hpt_client = hypertune.HyperTune()
    hpt_client.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='mse',
        metric_value=val_mse,
        global_step=int(epochs)
    )
    logging.info(f"Reported MSE to Vertex AI: {val_mse}")

    # Build and embed metadata
    meta = build_training_metadata(local_filedata, {
        'epochs': int(epochs),
        'learning_rate': float(learning_rate),
        'units': int(units),
        'activation': activation,
        'dropout_rate': float(dropout_rate),
        'activation_output': activation_output,
        'num_layers': int(num_layers),
        'ticker': ticker if ticker else "N/A",
        'final_metrics': final_metrics,
    })
    
    # Overwrite data_file with original path
    meta['data_file'] = original_filedata
    
    # Add scaler to metadata
    meta['scalers'] = serialize_scaler_map({ticker: mms}) if ticker else serialize_scaler_map({None: mms})
    
    # Add validation metadata if val_filedata is provided
    if val_filedata:
        meta['val_data_file'] = original_val_filedata
        meta['val_data_file_crc32'] = crc32_file(local_val_filedata)
        meta['val_data_stats'] = _data_stats(local_val_filedata)
        
    embed_metadata_h5(local_model_path, meta)

    # Save model to GCS
    model_gcs_url = None
    # If AIP_MODEL_DIR is set (Vertex AI custom job), save there
    aip_model_dir = os.getenv("AIP_MODEL_DIR")
    if aip_model_dir:
        # Save directly to Vertex AI model dir
        gcs_save_path = os.path.join(aip_model_dir, "model.h5")
        # Copy local model (with embedded metadata) directly to AIP_MODEL_DIR in GCS
        tf.io.gfile.copy(local_model_path, gcs_save_path, overwrite=True)
        model_gcs_url = gcs_save_path
        logging.info(f"Copied metadata-enriched model to AIP_MODEL_DIR: {gcs_save_path}")
    else:
        # Upload to custom bucket path
        trial_suffix = f"/trial_{trial_id}" if trial_id else ""
        experiment_name = f"rnn_lstm_hp_vai/{job_id}{trial_suffix}"
        model_file_name = f"{experiment_name}/model.h5"
        try:
            model_gcs_url = upload_to_bucket(model_file_name, local_model_path, bucket_name)
            logging.info(f"Uploaded model to GCS: {model_gcs_url}")
        except Exception as e:
            logging.error(f"Failed to upload model to GCS: {e}")

    # Write results to BigQuery
    try:
        bq_client = get_bigquery_client(project=project_id)
        ensure_trains_table_exists(bq_client, dataset_id=bq_dataset)
        
        # Prepare row data
        parameters_json = json.dumps({
            'learning_rate': float(learning_rate),
            'units': int(units),
            'activation': activation,
            'dropout_rate': float(dropout_rate),
            'activation_output': activation_output,
            'num_layers': int(num_layers),
            'epochs': int(epochs),
            'ticker': ticker if ticker else "N/A",
            'val_dataset_file': original_val_filedata if original_val_filedata else "N/A"
        })
        metrics_json = json.dumps(final_metrics)
        
        row_data = {
            "job_id": str(job_id),
            "trial_id": str(trial_id) if trial_id else None,
            "model_name": str(model_name),
            "training_date": datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            "parameters": parameters_json,
            "metrics": metrics_json,
            "model_path": model_gcs_url if model_gcs_url else local_model_path,
            "git_commit": meta.get("git_commit"),
            "git_branch": meta.get("git_branch"),
            "dataset_file": original_filedata
        }
        
        insert_train_result(bq_client, row_data, dataset_id=bq_dataset)
        logging.info(f"Successfully wrote training results to BigQuery {bq_dataset}.trains table!")
    except Exception as e:
        logging.error(f"Failed to write results to BigQuery: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
    fire.Fire(train_evaluate)
