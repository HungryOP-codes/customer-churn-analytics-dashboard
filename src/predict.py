import sys
from pathlib import Path
import numpy as np
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
    df['SeniorCitizen'] = int(customer_info.get('SeniorCitizen', 0))
    df['tenure'] = int(customer_info.get('Tenure', 0))
    df['MonthlyCharges'] = float(customer_info.get('MonthlyCharges', 0.0))
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
        known_classes = set(encoder.classes_)
        df[column] = df[column].astype(str).map(
            lambda val: val if val in known_classes else encoder.classes_[0]
        )
        df[column] = encoder.transform(df[column])
    return df


def generate_retention_recommendations(customer_info: dict, probability: float) -> list:
    recommendations = []
    contract = str(customer_info.get('Contract', ''))
    payment = str(customer_info.get('PaymentMethod', ''))
    tenure = float(customer_info.get('Tenure', 0))
    monthly = float(customer_info.get('MonthlyCharges', 0))

    if probability >= 0.5:
        if contract == 'Month-to-month':
            recommendations.append("Offer a 15% discount on an annual contract commitment.")
        if 'Electronic check' in payment:
            recommendations.append("Incentivize switching to automatic ACH or credit card payment with a one-time $10 credit.")
        if monthly > 80:
            recommendations.append("Propose service bundle optimization or loyalty pricing adjustment.")
        if tenure <= 12:
            recommendations.append("Assign a dedicated onboarding customer success specialist.")
        if not recommendations:
            recommendations.append("Initiate targeted proactive customer success outreach.")
    else:
        recommendations.append("Customer shows high retention affinity. Recommend cross-sell / up-sell opportunities.")

    return recommendations


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
        'model_name': package.get('model_name', 'Trained Classifier'),
        'recommendations': generate_retention_recommendations(customer_info, probability),
    }


def predict_batch(df_input: pd.DataFrame) -> pd.DataFrame:
    package = load_model_package()
    model = package['model']
    encoders = package['encoders']
    feature_columns = package['feature_columns']

    df = df_input.copy()
    # Normalize column casing if necessary
    cols_map = {c.lower(): c for c in df.columns}
    gender_col = cols_map.get('gender', 'gender')
    if gender_col in df.columns:
        df['gender'] = df[gender_col].replace({'Male': 1, 'Female': 0}).fillna(0).astype(int)
    else:
        df['gender'] = 1

    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce').fillna(0).astype(int)
    elif 'seniorcitizen' in cols_map and cols_map['seniorcitizen'] in df.columns:
        df['SeniorCitizen'] = pd.to_numeric(df[cols_map['seniorcitizen']], errors='coerce').fillna(0).astype(int)
    else:
        df['SeniorCitizen'] = 0

    tenure_col = cols_map.get('tenure', 'tenure')
    df['tenure'] = pd.to_numeric(df[tenure_col], errors='coerce').fillna(1).astype(int)

    monthly_col = cols_map.get('monthlycharges', 'MonthlyCharges')
    df['MonthlyCharges'] = pd.to_numeric(df[monthly_col], errors='coerce').fillna(50.0).astype(float)
    df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']

    tenure_bins = [0, 6, 12, 24, 48, 1000]
    tenure_labels = ['0-6', '7-12', '13-24', '25-48', '49+']
    df['Tenure_Group'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels, include_lowest=True).astype(str)

    charge_bins = [0, 35, 70, 100, 10000]
    charge_labels = ['Low', 'Moderate', 'High', 'Very High']
    df['Monthly_Charges_Group'] = pd.cut(df['MonthlyCharges'], bins=charge_bins, labels=charge_labels, include_lowest=True).astype(str)

    df['Customer_Lifetime_Value'] = df['MonthlyCharges'] * df['tenure']

    def calc_risk(r):
        if r['tenure'] <= 6 or r['MonthlyCharges'] > 90:
            return 'High Risk'
        if r['tenure'] <= 18 or r['MonthlyCharges'] > 70:
            return 'Medium Risk'
        return 'Low Risk'

    df['Risk_Category'] = df.apply(calc_risk, axis=1)

    contract_col = cols_map.get('contract', 'Contract')
    payment_col = cols_map.get('paymentmethod', 'PaymentMethod')
    df['Contract'] = df[contract_col].astype(str) if contract_col in df.columns else 'Month-to-month'
    df['PaymentMethod'] = df[payment_col].astype(str) if payment_col in df.columns else 'Electronic check'

    processed_df = encode_input(df[feature_columns].copy(), encoders)
    probabilities = model.predict_proba(processed_df[feature_columns])[:, 1]
    predictions = model.predict(processed_df[feature_columns])

    result_df = df_input.copy()
    result_df['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
    result_df['Churn_Probability'] = np.round(probabilities, 4)
    result_df['Risk_Category'] = df['Risk_Category']
    return result_df


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
