import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'dashboard') not in sys.path:
    sys.path.insert(0, str(ROOT / 'dashboard'))

from utils.config import (
    CLEANED_DATA_PATH,
    SEGMENTED_DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    RAW_DATA_PATH
)
from src.predict import predict_churn, predict_batch
from streamlit.testing.v1 import AppTest

def test_files_and_paths():
    print("=== TEST 1: Files, Paths, and Artifact Integrity ===")
    assert RAW_DATA_PATH.exists(), f"Missing raw data: {RAW_DATA_PATH}"
    assert CLEANED_DATA_PATH.exists(), f"Missing cleaned data: {CLEANED_DATA_PATH}"
    assert SEGMENTED_DATA_PATH.exists(), f"Missing segmented data: {SEGMENTED_DATA_PATH}"
    assert MODEL_PATH.exists(), f"Missing model: {MODEL_PATH}"
    assert METRICS_PATH.exists(), f"Missing metrics: {METRICS_PATH}"
    print("  [PASS] All expected data and model files exist.")

def test_dynamic_kpi_calculations():
    print("\n=== TEST 2: Dynamic KPI Calculation vs Ground Truth ===")
    df = pd.read_csv(CLEANED_DATA_PATH)
    total_customers = len(df)
    churned_customers = int(df['Churn'].sum())
    active_customers = int((df['Churn'] == 0).sum())
    churn_rate = churned_customers / total_customers
    total_rev = float((df['MonthlyCharges'] * df['tenure']).sum())
    avg_monthly = float(df['MonthlyCharges'].mean())
    avg_tenure = float(df['tenure'].mean())

    assert total_customers == 7043, f"Expected 7043, got {total_customers}"
    assert churned_customers == 1869, f"Expected 1869, got {churned_customers}"
    assert active_customers == 5174, f"Expected 5174, got {active_customers}"
    assert abs(churn_rate - 0.2653698) < 1e-4
    assert abs(avg_monthly - 64.76169) < 1e-2
    assert abs(avg_tenure - 32.37114) < 1e-2
    print(f"  [PASS] Ground truth verified: {total_customers} rows, {churned_customers} churned ({churn_rate:.2%}), Total Rev: ${total_rev:,.0f}")

def test_single_and_batch_prediction():
    print("\n=== TEST 3: Churn Prediction Inference Workflow ===")
    high_risk_input = {
        'Gender': 'Male',
        'SeniorCitizen': 0,
        'Tenure': 1,
        'MonthlyCharges': 95.0,
        'Contract': 'Month-to-month',
        'PaymentMethod': 'Electronic check'
    }
    high_res = predict_churn(high_risk_input)
    assert 'prediction' in high_res
    assert 'probability' in high_res
    assert 'risk_level' in high_res
    assert 'recommendations' in high_res
    assert high_res['probability'] > 0.5, f"Expected >0.5, got {high_res['probability']}"
    assert high_res['prediction'] == 'Yes'
    print(f"  [PASS] High risk customer: Prediction={high_res['prediction']}, Prob={high_res['probability']:.2%}, Risk={high_res['risk_level']}")

    low_risk_input = {
        'Gender': 'Female',
        'SeniorCitizen': 0,
        'Tenure': 65,
        'MonthlyCharges': 25.0,
        'Contract': 'Two year',
        'PaymentMethod': 'Credit card (automatic)'
    }
    low_res = predict_churn(low_risk_input)
    assert low_res['probability'] < 0.2, f"Expected <0.2, got {low_res['probability']}"
    assert low_res['prediction'] == 'No'
    print(f"  [PASS] Low risk customer: Prediction={low_res['prediction']}, Prob={low_res['probability']:.2%}, Risk={low_res['risk_level']}")

    batch_df = pd.DataFrame([high_risk_input, low_risk_input])
    scored = predict_batch(batch_df)
    assert len(scored) == 2
    assert 'Churn_Prediction' in scored.columns
    assert 'Churn_Probability' in scored.columns
    assert scored['Churn_Prediction'].iloc[0] == 'Yes'
    assert scored['Churn_Prediction'].iloc[1] == 'No'
    print("  [PASS] Batch prediction correctly scored multiple rows.")

def test_clustering_workflow():
    print("\n=== TEST 4: Customer Segmentation Workflow ===")
    seg_df = pd.read_csv(SEGMENTED_DATA_PATH)
    assert len(seg_df) == 7043
    assert 'Cluster' in seg_df.columns
    assert 'Risk_Group' in seg_df.columns
    assert 'Segment_Persona' in seg_df.columns
    clusters = set(seg_df['Cluster'].unique())
    assert clusters == {0, 1, 2}, f"Expected clusters {0, 1, 2}, got {clusters}"
    personas = set(seg_df['Segment_Persona'].unique())
    assert len(personas) == 3
    print(f"  [PASS] Segmentation verified: 3 clusters mapped to personas: {personas}")

def test_model_performance_math():
    print("\n=== TEST 5: Model Performance Math & Benchmark Verification ===")
    with open(METRICS_PATH, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    
    best_model = metrics['best_model']
    results = metrics['model_results']
    assert best_model in results, f"Champion {best_model} not in results"

    for model_name, m in results.items():
        cm = m['confusion_matrix']
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        total_test = tn + fp + fn + tp
        assert total_test == 1409, f"{model_name}: Expected test set size 1409, got {total_test}"
        
        calc_acc = (tp + tn) / total_test
        calc_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        calc_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        calc_f1 = 2 * calc_prec * calc_rec / (calc_prec + calc_rec) if (calc_prec + calc_rec) > 0 else 0
        
        assert abs(m['accuracy'] - calc_acc) < 1e-4, f"Accuracy mismatch for {model_name}"
        assert abs(m['precision'] - calc_prec) < 1e-4, f"Precision mismatch for {model_name}"
        assert abs(m['recall'] - calc_rec) < 1e-4, f"Recall mismatch for {model_name}"
        assert abs(m['f1_score'] - calc_f1) < 1e-4, f"F1 mismatch for {model_name}"
        print(f"  [PASS] {model_name}: Acc={m['accuracy']:.2%}, Prec={m['precision']:.2%}, Rec={m['recall']:.2%}, F1={m['f1_score']:.4f}, ROC AUC={m['roc_auc']:.4f}")

def test_data_explorer_operations():
    print("\n=== TEST 6: Data Explorer Search, Filter & Export Logic ===")
    df = pd.read_csv(CLEANED_DATA_PATH)
    search_target = '7590-VHVEG'
    matched = df[df['customerID'] == search_target]
    assert len(matched) == 1
    retained = df[df['Churn'] == 0]
    churned = df[df['Churn'] == 1]
    assert len(retained) == 5174
    assert len(churned) == 1869
    m2m = df[df['Contract'] == 'Month-to-month']
    one_yr = df[df['Contract'] == 'One year']
    two_yr = df[df['Contract'] == 'Two year']
    assert len(m2m) + len(one_yr) + len(two_yr) == 7043
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    assert len(csv_bytes) > 10000
    print(f"  [PASS] Search, filtering, and CSV serialization ({len(csv_bytes):,} bytes) verified.")

def test_all_seven_streamlit_pages_load():
    print("\n=== TEST 7: Automated Streamlit AppTest for All 7 Pages ===")
    pages = [
        'Executive Summary',
        'Customer Analytics',
        'Sales Dashboard',
        'Prediction Center',
        'Customer Segmentation',
        'Model Performance',
        'Data Explorer'
    ]
    for page in pages:
        at = AppTest.from_file(str(ROOT / 'dashboard' / 'app.py'), default_timeout=30)
        at.query_params['page'] = page
        at.run()
        exceptions = [e for e in at.exception]
        assert len(exceptions) == 0, f"Page '{page}' threw exception: {exceptions}"
        print(f"  [PASS] Page '{page}' loaded with 0 exceptions.")

if __name__ == '__main__':
    test_files_and_paths()
    test_dynamic_kpi_calculations()
    test_single_and_batch_prediction()
    test_clustering_workflow()
    test_model_performance_math()
    test_data_explorer_operations()
    test_all_seven_streamlit_pages_load()
    print("\n=========================================")
    print("ALL 7 TEST SUITES PASSED WITH 0 FAILURES!")
    print("=========================================")
