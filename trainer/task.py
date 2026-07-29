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
from model_metadata import build_training_metadata, embed_metadata_h5, log_and_evaluate_supervised

from google.cloud import storage
from google.cloud import bigquery
from google.api_core.client_options import ClientOptions
from google.cloud.exceptions import NotFound
import google.auth
import hypertune

warnings.filterwarnings('ignore')

def get_bigquery_client():
    is_local = os.getenv("ISLOCAL", "false").lower() == "true"
    if is_local:
        endpoint = os.getenv("BIGQUERY_EMULATOR_HOST", "http://localhost:9050")
        if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
            endpoint = f"http://{endpoint}"
        client_options = ClientOptions(api_endpoint=endpoint)
        from google.auth.credentials import AnonymousCredentials
        return bigquery.Client(
            project="game-bolsa",
            client_options=client_options,
            credentials=AnonymousCredentials()
        )
    else:
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path and os.path.exists(creds_path):
            return bigquery.Client.from_service_account_json(creds_path)
        else:
            return bigquery.Client()

def ensure_trains_table_exists(client: bigquery.Client):
    dataset_ref = client.dataset("play")
    try:
        client.get_dataset(dataset_ref)
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "EU"
        client.create_dataset(dataset)
        print("Created dataset play")

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
        print("Created table play.trains")

def insert_train_result(client: bigquery.Client, row_data: dict):
    table_ref = client.dataset("play").table("trains")
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
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path_to_file)
    return blob.public_url

def build_model(units, activation, dropout_rate, activation_output, learning_rate):
    """Creating LSTM model"""
    input_shape = (10, 1)
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(
            LSTM(
                units=units,
                activation=activation
            )
        ),
        Dropout(dropout_rate),
        Dense(10, activation=activation_output)
    ])
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

def train_evaluate(filedata="https://storage.googleapis.com/ibex35/data/IBEX-1994-2020.csv",
                   modelpath="model/ibex_rnn_lstm_hp_model.h5",
                   epochs=10,
                   learning_rate=0.001,
                   units=50,
                   activation="relu",
                   dropout_rate=0.1,
                   activation_output="linear",
                   job_id=None,
                   trial_id=None,
                   bucket_name="game-bolsa-models",
                   ticker=None
                   ):
    # Set job_id and trial_id from env if not passed
    if not job_id:
        job_id = os.getenv("AIP_JOB_ID", "local_job_" + str(round(time.time())))
    if not trial_id:
        trial_id = os.getenv("AIP_TRIAL_ID")

    logging.info("Tensorflow version " + tf.__version__)
    logging.info('> Loading data... ')

    data = pd.read_csv(filedata)
    logging.info(data.head())

    # Filter by ticker if 'Ticker' column exists
    if 'Ticker' in data.columns:
        if ticker:
            data = data[data['Ticker'] == ticker]
            logging.info(f"Filtered dataset for ticker: {ticker}")
        else:
            unique_tickers = data['Ticker'].unique()
            if len(unique_tickers) > 0:
                ticker = unique_tickers[0]
                data = data[data['Ticker'] == ticker]
                logging.info(f"No ticker specified. Defaulting to first ticker: {ticker}")

    data = data[['Date', 'Close']]  # Extracting required columns
    data.dropna(inplace=True)
    
    # Try multiple date formats
    try:
        data['Date'] = pd.to_datetime(data['Date'].apply(lambda x: x.split()[0]), format='%Y-%m-%d')
    except Exception:
        try:
            data['Date'] = pd.to_datetime(data['Date'].apply(lambda x: x.split()[0]), format='%d/%m/%Y')
        except Exception:
            data['Date'] = pd.to_datetime(data['Date'].apply(lambda x: x.split()[0]))

    data.set_index('Date', drop=True, inplace=True)
    logging.info(data.head())

    # Normalization
    mms = MinMaxScaler()
    data[['Close']] = mms.fit_transform(data[['Close']])

    training_size = round(len(data) * 0.80)
    train_data = data[:training_size]
    test_data = data[training_size:]

    train_seq, train_label = create_sequences(train_data, 10)
    test_seq, test_label = create_sequences(test_data, 10)

    # Build model
    model_rnn = build_model(
        units=int(units),
        activation=activation,
        dropout_rate=float(dropout_rate),
        activation_output=activation_output,
        learning_rate=float(learning_rate)
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
    meta = build_training_metadata(filedata, {
        'epochs': int(epochs),
        'learning_rate': float(learning_rate),
        'units': int(units),
        'activation': activation,
        'dropout_rate': float(dropout_rate),
        'activation_output': activation_output,
        'ticker': ticker if ticker else "N/A",
        'final_metrics': final_metrics,
    })
    embed_metadata_h5(local_model_path, meta)

    # Save model to GCS
    model_gcs_url = None
    # If AIP_MODEL_DIR is set (Vertex AI custom job), save there
    aip_model_dir = os.getenv("AIP_MODEL_DIR")
    if aip_model_dir:
        # Save directly to Vertex AI model dir
        gcs_save_path = os.path.join(aip_model_dir, "model.h5")
        # TensorFlow can save directly to gs:// paths
        model_rnn.save(gcs_save_path)
        model_gcs_url = gcs_save_path
        logging.info(f"Saved model directly to AIP_MODEL_DIR: {gcs_save_path}")
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
        bq_client = get_bigquery_client()
        ensure_trains_table_exists(bq_client)
        
        # Prepare row data
        parameters_json = json.dumps({
            'learning_rate': float(learning_rate),
            'units': int(units),
            'activation': activation,
            'dropout_rate': float(dropout_rate),
            'activation_output': activation_output,
            'epochs': int(epochs),
            'ticker': ticker if ticker else "N/A"
        })
        metrics_json = json.dumps(final_metrics)
        
        row_data = {
            "job_id": str(job_id),
            "trial_id": str(trial_id) if trial_id else None,
            "model_name": "ibex_rnn_lstm_hp_model",
            "training_date": datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            "parameters": parameters_json,
            "metrics": metrics_json,
            "model_path": model_gcs_url if model_gcs_url else local_model_path,
            "git_commit": meta.get("git_commit"),
            "git_branch": meta.get("git_branch"),
            "dataset_file": filedata
        }
        
        insert_train_result(bq_client, row_data)
        logging.info("Successfully wrote training results to BigQuery play.trains table!")
    except Exception as e:
        logging.error(f"Failed to write results to BigQuery: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
    fire.Fire(train_evaluate)
