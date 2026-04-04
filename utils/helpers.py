import zipfile
from pathlib import Path

from .config import DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR, RAW_DATA_PATH, KAGGLE_DATASET


def ensure_directories():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def download_telco_dataset():
    ensure_directories()
    if RAW_DATA_PATH.exists():
        return RAW_DATA_PATH

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as error:
        raise ImportError(
            "Kaggle API not installed. Install the kaggle package or place the dataset file in data/raw/."
        ) from error

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(str(KAGGLE_DATASET), path=str(DATA_RAW_DIR), unzip=True, force=False)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Downloaded dataset not found. Please place WA_Fn-UseC_-Telco-Customer-Churn.csv inside {DATA_RAW_DIR}."
        )

    return RAW_DATA_PATH
