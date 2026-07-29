"""Shared model metadata utilities for all gostocks training and prediction scripts.

Training scripts call build_training_metadata() and embed_metadata_h5() / write_metadata_json().
Predict/validation scripts call log_model_info() right after loading a model.
"""
import base64
import json
import logging
import os
import pickle
import platform
import subprocess
import zlib
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def crc32_file(path: str) -> str:
    """Return the CRC32 hex digest of a file, or 'N/A' if unreadable."""
    crc = 0
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                crc = zlib.crc32(chunk, crc)
    except OSError:
        return 'N/A'
    return format(crc & 0xFFFFFFFF, '08x')


def serialize_scaler_map(scaler_map: dict) -> dict:
    """Serialize a dict of ticker -> MinMaxScaler to JSON-safe base64 strings."""
    serialized = {}
    for ticker, scaler in scaler_map.items():
        serialized[ticker if ticker is not None else '__none__'] = base64.b64encode(
            pickle.dumps(scaler)
        ).decode('utf-8')
    return serialized


def deserialize_scaler_map(serialized: dict) -> dict:
    """Reconstruct a dict of ticker -> MinMaxScaler from base64 strings."""
    scaler_map = {}
    for ticker, encoded in serialized.items():
        key = None if ticker == '__none__' else ticker
        scaler_map[key] = pickle.loads(base64.b64decode(encoded.encode('utf-8')))
    return scaler_map


def git_commit() -> str:
    """Return the current git commit hash, or 'N/A' if not in a repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else 'N/A'
    except Exception:
        return 'N/A'


def git_branch() -> str:
    """Return the current git branch name, or 'N/A'."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else 'N/A'
    except Exception:
        return 'N/A'


def _data_stats(local_path: str) -> dict:
    """Return basic statistics about the data file (row/column counts, date range)."""
    stats = {'rows': 'N/A', 'columns': 'N/A', 'date_min': 'N/A', 'date_max': 'N/A', 'tickers': 'N/A'}
    try:
        import pandas as pd
        df = pd.read_csv(local_path, nrows=None)
        stats['rows'] = int(len(df))
        stats['columns'] = list(df.columns)
    except Exception as e:
        logging.warning("data_stats: could not read CSV rows/columns from %s: %s", local_path, e)
        return stats

    if 'Date' in df.columns:
        try:
            dates = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True).dropna()
            if not dates.empty:
                stats['date_min'] = str(dates.min().date())
                stats['date_max'] = str(dates.max().date())
        except Exception as e:
            logging.warning("data_stats: could not parse Date column from %s: %s", local_path, e)

    if 'Ticker' in df.columns:
        try:
            stats['tickers'] = sorted(df['Ticker'].unique().tolist())
        except Exception as e:
            logging.warning("data_stats: could not extract tickers from %s: %s", local_path, e)

    return stats


# ---------------------------------------------------------------------------
# Build metadata
# ---------------------------------------------------------------------------

def build_training_metadata(data_file: str, params: dict,
                            framework: str = 'tensorflow') -> dict:
    """Collect full runtime + data provenance information.

    Args:
        data_file:  URL or path to the training data file.
        params:     Dict of training hyperparameters.
        framework:  'tensorflow' | 'pytorch' — adds the relevant version field.
    """
    local_path = data_file.replace('file://', '') if data_file.startswith('file://') else data_file

    meta = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'git_commit': git_commit(),
        'git_branch': git_branch(),
        'data_file': data_file,
        'data_file_crc32': crc32_file(local_path),
        'data_stats': _data_stats(local_path),
        'parameters': params,
    }

    if framework == 'tensorflow':
        try:
            import tensorflow as tf
            meta['tensorflow_version'] = tf.__version__
        except ImportError:
            pass
        try:
            import keras
            meta['keras_version'] = keras.__version__
        except ImportError:
            pass

    elif framework == 'pytorch':
        try:
            import torch
            meta['torch_version'] = torch.__version__
        except ImportError:
            pass
        try:
            import tensorflow as tf
            meta['tensorflow_version'] = tf.__version__
        except ImportError:
            pass

    return meta


# ---------------------------------------------------------------------------
# Embed / write metadata
# ---------------------------------------------------------------------------

def embed_metadata_h5(model_path: str, meta: dict) -> None:
    """Write metadata as HDF5 root attributes into an existing .h5 model file."""
    import h5py
    with h5py.File(model_path, 'a') as hf:
        for k, v in meta.items():
            hf.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
    logging.info("Model metadata embedded in %s", model_path)


def write_metadata_json(model_path: str, meta: dict) -> None:
    """Write metadata as a sidecar .meta.json beside the model file.

    Used for .keras format (ZIP-based, not writable by h5py).
    """
    ext = os.path.splitext(model_path)[1]
    meta_path = model_path.replace(ext, '.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logging.info("Model metadata saved to %s", meta_path)


# ---------------------------------------------------------------------------
# Read metadata
# ---------------------------------------------------------------------------

def read_metadata(model_path: str) -> dict:
    """Read training metadata from a model file.

    Supports .h5 (h5py attrs), .keras (sidecar .meta.json), .pth (state_dict key).
    Returns an empty dict if no training metadata is found.
    """
    ext = os.path.splitext(model_path)[1].lower()

    if ext == '.h5':
        return _read_h5(model_path)
    elif ext == '.keras':
        return _read_keras(model_path)
    elif ext == '.pth':
        return _read_pth(model_path)
    else:
        logging.warning("read_metadata: unsupported extension '%s'", ext)
        return {}


def _read_h5(path: str) -> dict:
    import h5py
    meta = {}
    with h5py.File(path, 'r') as f:
        for k, v in f.attrs.items():
            try:
                meta[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                meta[k] = v
    return meta


def _read_keras(path: str) -> dict:
    ext = os.path.splitext(path)[1]
    meta_path = path.replace(ext, '.meta.json')
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, 'r') as f:
        return json.load(f)


def _read_pth(path: str) -> dict:
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    return state.get('training_metadata', {})


# ---------------------------------------------------------------------------
# Log model info at predict / validation time
# ---------------------------------------------------------------------------

def log_model_info(model_path: str) -> None:
    """Read and log training metadata for a model file.

    Call this at the start of any predict / validation script right after
    loading the model, so provenance is always visible in the logs.
    """
    meta = read_metadata(model_path)
    if not meta:
        logging.info("=== Model Info: no training metadata found in %s ===", model_path)
        return

    logging.info("=== Model Info: %s ===", os.path.basename(model_path))
    _interesting = [
        'training_date', 'git_commit', 'git_branch',
        'tensorflow_version', 'keras_version', 'torch_version',
        'python_version', 'platform',
        'data_file', 'data_file_crc32', 'data_stats', 'parameters',
    ]
    for key in _interesting:
        if key not in meta:
            continue
        val = meta[key]
        if isinstance(val, dict):
            logging.info("  %s:", key)
            for k2, v2 in val.items():
                logging.info("    %s: %s", k2, v2)
        else:
            logging.info("  %s: %s", key, val)
    logging.info("=" * 50)


def log_and_evaluate_supervised(model, test_seq, test_label, model_name: str, train_seq=None) -> dict:
    """Run final evaluation on test sequences for supervised forecasting models,
    log an === EVALUATION SUMMARY ===, and return a dictionary of metrics.
    """
    logging.info("Starting final evaluation on test set...")
    try:
        eval_res = model.evaluate(test_seq, test_label, verbose=0)
        if isinstance(eval_res, (list, tuple)):
            test_loss = float(eval_res[0])
            test_mse = float(eval_res[1]) if len(eval_res) > 1 else test_loss
        else:
            test_loss = float(eval_res)
            test_mse = float(test_loss)
    except Exception as e:
        logging.warning("Failed running model.evaluate: %s", e)
        test_loss, test_mse = 0.0, 0.0

    try:
        import numpy as np
        predictions = model.predict(test_seq, verbose=0)
        test_mae = float(np.mean(np.abs(test_label - predictions)))
        test_rmse = float(np.sqrt(np.mean((test_label - predictions) ** 2)))

        y_true_flat = np.array(test_label).flatten()
        y_pred_flat = np.array(predictions).flatten()
        if len(y_true_flat) > 1:
            diff_true = np.diff(y_true_flat)
            diff_pred = np.diff(y_pred_flat)
            directional_acc = float(np.mean(np.equal(np.sign(diff_true), np.sign(diff_pred))) * 100.0)
        else:
            directional_acc = 0.0
    except Exception as e:
        logging.warning("Failed computing detailed test metrics: %s", e)
        test_mae, test_rmse, directional_acc = 0.0, 0.0, 0.0

    logging.info("=== EVALUATION SUMMARY ===")
    logging.info("Model File           : %s", os.path.basename(model_name))
    if train_seq is not None:
        logging.info("Train Sequences      : %d", len(train_seq))
    logging.info("Test Sequences       : %d", len(test_seq))
    logging.info("Test Loss (MSE)      : %.6f", test_loss)
    logging.info("Test MAE             : %.6f", test_mae)
    logging.info("Test RMSE            : %.6f", test_rmse)
    logging.info("Directional Accuracy : %.2f%%", directional_acc)
    logging.info("=" * 50)

    return {
        'loss': test_loss,
        'mse': test_mse,
        'mae': test_mae,
        'rmse': test_rmse,
        'directional_accuracy': directional_acc
    }

