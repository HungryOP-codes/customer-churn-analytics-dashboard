import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import MODEL_PATH, METRICS_PATH, CLEANED_DATA_PATH
from utils.helpers import ensure_directories
from src.feature_engineering import prepare_data


def save_eda_plots():
    df = pd.read_csv(CLEANED_DATA_PATH)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
    df['Customer_Lifetime_Value'] = df['MonthlyCharges'] * df['tenure']

    plots_dir = CLEANED_DATA_PATH.parent / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    df['Churn'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, labels=['No', 'Yes'])
    plt.title('Churn Distribution')
    plt.ylabel('')
    plt.savefig(plots_dir / 'churn_distribution.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='gender', hue='Churn')
    plt.title('Gender vs Churn')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.savefig(plots_dir / 'gender_vs_churn.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Contract', hue='Churn')
    plt.title('Contract vs Churn')
    plt.xlabel('Contract')
    plt.ylabel('Count')
    plt.xticks(rotation=30)
    plt.savefig(plots_dir / 'contract_vs_churn.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='PaymentMethod', hue='Churn')
    plt.title('Payment Method vs Churn')
    plt.xlabel('Payment Method')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(plots_dir / 'payment_method_vs_churn.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df['MonthlyCharges'], kde=True, bins=30)
    plt.title('Monthly Charges Distribution')
    plt.xlabel('Monthly Charges')
    plt.savefig(plots_dir / 'monthly_charges_distribution.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Churn', y='tenure')
    plt.title('Tenure vs Churn')
    plt.xlabel('Churn')
    plt.ylabel('Tenure (months)')
    plt.savefig(plots_dir / 'tenure_vs_churn.png', bbox_inches='tight')
    plt.close()


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': float(accuracy_score(y_test, predictions)),
        'precision': float(precision_score(y_test, predictions, zero_division=0)),
        'recall': float(recall_score(y_test, predictions, zero_division=0)),
        'f1_score': float(f1_score(y_test, predictions, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, probabilities)),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
    }
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    metrics['roc_curve'] = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
    }
    return metrics


def extract_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return {name: 0.0 for name in feature_names}

    sorted_indices = np.argsort(importances)[::-1]
    return {feature_names[i]: float(importances[i]) for i in sorted_indices}


def train_and_save_models():
    ensure_directories()
    X_train, X_test, y_train, y_test, encoders, feature_columns = prepare_data()

    candidates = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42, solver='lbfgs'),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0),
    }

    best_score = -1
    best_name = None
    best_model = None
    results = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        if metrics['f1_score'] > best_score:
            best_score = metrics['f1_score']
            best_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError('No model was trained successfully.')

    save_eda_plots()

    package = {
        'model_name': best_name,
        'model': best_model,
        'encoders': encoders,
        'feature_columns': feature_columns,
    }
    joblib.dump(package, MODEL_PATH)

    best_importance = extract_feature_importance(best_model, feature_columns)
    model_performance = {
        'best_model': best_name,
        'model_results': results,
        'feature_importance': best_importance,
    }
    model_performance['best_model_metrics'] = results[best_name]

    with open(METRICS_PATH, 'w', encoding='utf-8') as file:
        json.dump(model_performance, file, indent=2)

    return MODEL_PATH, METRICS_PATH


def main():
    model_path, metrics_path = train_and_save_models()
    print(f'Best model saved to {model_path}')
    print(f'Model metrics saved to {metrics_path}')


if __name__ == '__main__':
    main()
