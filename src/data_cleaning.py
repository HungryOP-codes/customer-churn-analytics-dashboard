import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import RAW_DATA_PATH, CLEANED_DATA_PATH
from utils.helpers import ensure_directories, download_telco_dataset


def load_raw_data():
    ensure_directories()
    if not RAW_DATA_PATH.exists():
        try:
            download_telco_dataset()
        except Exception as error:
            raise FileNotFoundError(
                f"Could not locate or download dataset. Please place WA_Fn-UseC_-Telco-Customer-Churn.csv in {RAW_DATA_PATH.parent}."
            ) from error
    return pd.read_csv(RAW_DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    binary_columns = [
        'gender',
        'SeniorCitizen',
        'Partner',
        'Dependents',
        'PhoneService',
        'PaperlessBilling',
        'Churn',
    ]

    for column in binary_columns:
        if column in df.columns:
            if df[column].dtype == object:
                df[column] = df[column].replace(binary_map)
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)

    return df


def show_dataset_summary(df: pd.DataFrame):
    print('\n=== Dataset Summary ===')
    print('First 5 rows:')
    print(df.head().to_string(index=False))
    print('\nShape:', df.shape)
    print('\nColumns:', df.columns.tolist())
    print('\nData types:')
    print(df.dtypes)


def save_cleaned_data(df: pd.DataFrame):
    ensure_directories()
    df.to_csv(CLEANED_DATA_PATH, index=False)
    return CLEANED_DATA_PATH


def main():
    df = load_raw_data()
    cleaned = clean_data(df)
    save_cleaned_data(cleaned)
    show_dataset_summary(cleaned)


if __name__ == '__main__':
    main()
