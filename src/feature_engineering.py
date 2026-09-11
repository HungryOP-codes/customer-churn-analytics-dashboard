import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import CLEANED_DATA_PATH
from utils.helpers import ensure_directories
from src.data_cleaning import load_raw_data, clean_data


def load_clean_data():
    ensure_directories()
    if not CLEANED_DATA_PATH.exists():
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)
        cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
        return cleaned_df
    return pd.read_csv(CLEANED_DATA_PATH)


def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'TotalCharges' not in df.columns:
        df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']

    tenure_bins = [0, 6, 12, 24, 48, df['tenure'].max() + 1]
    tenure_labels = ['0-6', '7-12', '13-24', '25-48', '49+']
    df['Tenure_Group'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels, include_lowest=True)

    charge_bins = [0, 35, 70, 100, df['MonthlyCharges'].max() + 1]
    charge_labels = ['Low', 'Moderate', 'High', 'Very High']
    df['Monthly_Charges_Group'] = pd.cut(df['MonthlyCharges'], bins=charge_bins, labels=charge_labels, include_lowest=True)

    df['Customer_Lifetime_Value'] = df['MonthlyCharges'] * df['tenure']

    def assign_risk(row):
        # Eliminate target leakage: compute risk using only customer behavioral attributes
        if row['tenure'] <= 6 or row['MonthlyCharges'] > 90:
            return 'High Risk'
        if row['tenure'] <= 18 or row['MonthlyCharges'] > 70:
            return 'Medium Risk'
        return 'Low Risk'

    df['Risk_Category'] = df.apply(assign_risk, axis=1)
    return df


def encode_features(df: pd.DataFrame, encoders: dict = None, fit_encoders: bool = False):
    df = df.copy()
    object_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if fit_encoders:
        encoders = {}
        for column in object_columns:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].astype(str))
            encoders[column] = encoder
        return df, encoders
    for column in object_columns:
        if column not in encoders:
            raise ValueError(f"Missing encoder for column {column}")
        df[column] = encoders[column].transform(df[column].astype(str))
    return df


def prepare_data(test_size: float = 0.2, random_state: int = 42):
    df = load_clean_data()
    df = build_derived_features(df)

    feature_columns = [
        'gender',
        'SeniorCitizen',
        'tenure',
        'MonthlyCharges',
        'TotalCharges',
        'Contract',
        'PaymentMethod',
        'Tenure_Group',
        'Monthly_Charges_Group',
        'Customer_Lifetime_Value',
        'Risk_Category',
    ]

    available_columns = [col for col in feature_columns if col in df.columns]
    X = df[available_columns].copy()
    y = df['Churn'].copy()
    X, encoders = encode_features(X, fit_encoders=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, encoders, available_columns


def main():
    X_train, X_test, y_train, y_test, encoders, columns = prepare_data()
    print('Train shape:', X_train.shape)
    print('Test shape:', X_test.shape)
    print('Encoded features:', columns)


if __name__ == '__main__':
    main()
