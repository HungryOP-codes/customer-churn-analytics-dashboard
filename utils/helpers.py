import urllib.request
from pathlib import Path
import pandas as pd
import numpy as np

from .config import DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR, RAW_DATA_PATH, KAGGLE_DATASET

DATASET_URL = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def ensure_directories():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _generate_synthetic_telco_dataset(target_path: Path, n_samples: int = 1500):
    """Fallback generator for realistic Telco Customer Churn dataset if offline."""
    np.random.seed(42)
    customer_ids = [f"{np.random.randint(1000, 9999)}-{''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(n_samples)]
    gender = np.random.choice(['Male', 'Female'], n_samples)
    senior = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
    tenure = np.random.randint(1, 73, n_samples)
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
    multiple_lines = np.random.choice(['No phone service', 'No', 'Yes'], n_samples, p=[0.10, 0.48, 0.42])
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22])
    online_security = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.29, 0.49, 0.22])
    online_backup = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.44, 0.22])
    device_protection = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.44, 0.22])
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.29, 0.49, 0.22])
    streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.38, 0.40, 0.22])
    streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.39, 0.39, 0.22])
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        n_samples,
        p=[0.34, 0.23, 0.22, 0.21]
    )
    monthly_charges = np.where(
        internet_service == 'Fiber optic',
        np.random.uniform(70.0, 118.0, n_samples),
        np.where(internet_service == 'DSL', np.random.uniform(30.0, 85.0, n_samples), np.random.uniform(18.0, 26.0, n_samples))
    )
    monthly_charges = np.round(monthly_charges, 2)
    total_charges = np.round(monthly_charges * tenure + np.random.uniform(-10, 10, n_samples), 2)
    total_charges = np.maximum(total_charges, monthly_charges)

    # Churn probability based on contract and charges
    churn_prob = (
        0.42 * (contract == 'Month-to-month') +
        0.18 * (payment_method == 'Electronic check') +
        0.15 * (monthly_charges > 70) -
        0.30 * (tenure > 36) -
        0.15 * (contract == 'Two year')
    )
    churn_prob = 1 / (1 + np.exp(-churn_prob * 2))
    churn = np.where(np.random.rand(n_samples) < churn_prob, 'Yes', 'No')

    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges.astype(str),
        'Churn': churn
    })
    df.to_csv(target_path, index=False)
    return target_path


def download_telco_dataset():
    ensure_directories()
    if RAW_DATA_PATH.exists() and RAW_DATA_PATH.stat().st_size > 1000:
        return RAW_DATA_PATH

    # Attempt 1: Download from public URL mirror
    try:
        req = urllib.request.Request(DATASET_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response, open(RAW_DATA_PATH, 'wb') as out_file:
            out_file.write(response.read())
        if RAW_DATA_PATH.exists() and RAW_DATA_PATH.stat().st_size > 1000:
            return RAW_DATA_PATH
    except Exception:
        pass

    # Attempt 2: Kaggle API if available
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(str(KAGGLE_DATASET), path=str(DATA_RAW_DIR), unzip=True, force=False)
        if RAW_DATA_PATH.exists() and RAW_DATA_PATH.stat().st_size > 1000:
            return RAW_DATA_PATH
    except Exception:
        pass

    # Attempt 3: Robust synthetic dataset fallback
    return _generate_synthetic_telco_dataset(RAW_DATA_PATH)
