from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

RAW_DATA_PATH = DATA_RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEANED_DATA_PATH = DATA_PROCESSED_DIR / "cleaned_data.csv"
SEGMENTED_DATA_PATH = DATA_PROCESSED_DIR / "segmented_customers.csv"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
METRICS_PATH = MODEL_DIR / "model_performance.json"
KAGGLE_DATASET = "blastchar/telco-customer-churn"
