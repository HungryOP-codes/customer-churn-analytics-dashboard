import sys
from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import MODEL_PATH


def load_model_package():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model package not found at {MODEL_PATH}. Train the model first.")
    return joblib.load(MODEL_PATH)


def build_input_dataframe(customer_info: dict) -> pd.DataFrame:
    df = pd.DataFrame([customer_info])
    df['Gender'] = df['Gender'].astype(str)
    df['gender'] = df['Gender'].replace({'Male': 1, 'Female': 0}).fillna(0).astype(int)
    df['SeniorCitizen'] = int(df.get('SeniorCitizen', 0))
    df['tenure'] = int(df['Tenure'])
    df['MonthlyCharges'] = float(df['MonthlyCharges'])
    df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']

    tenure = df['tenure'].iloc[0]
    monthly = df['MonthlyCharges'].iloc[0]

    if tenure <= 6:
        tenure_group = '0-6'
    elif tenure <= 12:
        tenure_group = '7-12'
    elif tenure <= 24:
        tenure_group = '13-24'
    elif tenure <= 48:
        tenure_group = '25-48'
    else:
        tenure_group = '49+'

    if monthly <= 35:
        monthly_group = 'Low'
    elif monthly <= 70:
        monthly_group = 'Moderate'
    elif monthly <= 100:
        monthly_group = 'High'
    else:
        monthly_group = 'Very High'

    df['Tenure_Group'] = tenure_group
    df['Monthly_Charges_Group'] = monthly_group
    df['Customer_Lifetime_Value'] = df['MonthlyCharges'] * df['tenure']

    if df['tenure'].iloc[0] <= 6 or df['MonthlyCharges'].iloc[0] > 90:
        risk = 'High Risk'
    elif df['tenure'].iloc[0] <= 18 or df['MonthlyCharges'].iloc[0] > 70:
        risk = 'Medium Risk'
    else:
        risk = 'Low Risk'

    df['Risk_Category'] = risk
    df['Contract'] = df['Contract'].astype(str)
    df['PaymentMethod'] = df['PaymentMethod'].astype(str)

    return df


def encode_input(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for column, encoder in encoders.items():
        if column not in df.columns:
            raise ValueError(f"Expected feature column '{column}' not found in input.")
        df[column] = encoder.transform(df[column].astype(str))
    return df


def predict_churn(customer_info: dict) -> dict:
    package = load_model_package()
    model = package['model']
    encoders = package['encoders']
    feature_columns = package['feature_columns']

    df = build_input_dataframe(customer_info)
    input_df = encode_input(df, encoders)
    X = input_df[feature_columns]

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    risk_level = df['Risk_Category'].iloc[0]

    return {
        'prediction': 'Yes' if prediction == 1 else 'No',
        'probability': round(probability, 4),
        'risk_level': risk_level,
    }


def main():
    sample = {
        'Gender': 'Female',
        'SeniorCitizen': 0,
        'Tenure': 12,
        'MonthlyCharges': 70.0,
        'Contract': 'Month-to-month',
        'PaymentMethod': 'Electronic check',
    }
    result = predict_churn(sample)
    print('Prediction result:')
    print(result)


if __name__ == '__main__':
    main()
