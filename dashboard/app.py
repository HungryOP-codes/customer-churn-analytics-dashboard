import sys
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# -------------------------
# PATH SETUP
# -------------------------
ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent

for p in [str(ROOT), str(DASHBOARD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.config import (
    CLEANED_DATA_PATH,
    SEGMENTED_DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
)

# -------------------------
# PAGE CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="Customer Churn & Sales Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# MODERN DESIGN SYSTEM & CSS
# -------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #888888;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 500;
    }

    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }

    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-high { background-color: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-med { background-color: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-low { background-color: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }

    div[data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }

    .sidebar-brand {
        padding: 0.5rem 0 1.25rem 0;
        text-align: left;
    }
    .brand-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #6366f1;
        margin: 0;
    }
    .brand-sub {
        font-size: 0.75rem;
        color: #9ca3af;
        margin: 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------
# IMPORT PAGES DIRECTLY
# -------------------------
from views import (
    _1_Executive_Summary,
    _2_Customer_Analytics,
    _3_Sales_Dashboard,
    _4_Prediction_Center,
    _5_Customer_Segmentation,
    _6_Model_Performance,
    _7_Data_Explorer,
)

# -------------------------
# LOAD OR INITIALIZE DATA
# -------------------------
@st.cache_data
def load_data():
    if not (CLEANED_DATA_PATH.exists() and SEGMENTED_DATA_PATH.exists() and MODEL_PATH.exists() and METRICS_PATH.exists()):
        from src.data_cleaning import load_raw_data, clean_data, save_cleaned_data
        from src.train_model import train_and_save_models
        from src.clustering import perform_clustering

        raw = load_raw_data()
        cleaned = clean_data(raw)
        save_cleaned_data(cleaned)
        train_and_save_models()
        perform_clustering()

    df = pd.read_csv(CLEANED_DATA_PATH)
    segmented = pd.read_csv(SEGMENTED_DATA_PATH)
    model_package = joblib.load(MODEL_PATH)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        model_metrics = json.load(f)

    return df, segmented, model_package, model_metrics


# -------------------------
# MAIN APPLICATION
# -------------------------
def main():
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <p class="brand-title">⚡ ChurnIQ & Sales</p>
                <p class="brand-sub">Predictive Retention & Analytics</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page_options = [
            "Executive Summary",
            "Customer Analytics",
            "Sales Dashboard",
            "Prediction Center",
            "Customer Segmentation",
            "Model Performance",
            "Data Explorer",
        ]
        requested_page = st.query_params.get("page", None)
        default_index = 0
        if requested_page in page_options:
            default_index = page_options.index(requested_page)

        selected = option_menu(
            menu_title=None,
            options=page_options,
            icons=[
                "speedometer2",
                "people-fill",
                "currency-dollar",
                "lightning-charge-fill",
                "pie-chart-fill",
                "cpu-fill",
                "table",
            ],
            default_index=default_index,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"font-size": "0.95rem"},
                "nav-link": {
                    "font-size": "0.88rem",
                    "text-align": "left",
                    "margin": "3px 0",
                    "border-radius": "8px",
                    "padding": "0.6rem 0.8rem",
                },
                "nav-link-selected": {
                    "background-color": "#4f46e5",
                    "font-weight": "600",
                },
            },
        )
        if requested_page in page_options:
            selected = requested_page

        st.markdown("---")
        st.caption("🚀 **Engine**: Streamlit + Scikit-Learn + XGBoost")
        st.caption("📊 **Dataset**: Telco Customer Churn (7,043 rows)")

    try:
        df, segmented_df, model_package, model_metrics = load_data()
    except Exception as e:
        st.error(f"Error loading pipeline data: {e}")
        st.info("Please run `python src/train_model.py` and `python src/clustering.py` from your terminal.")
        return

    if selected == "Executive Summary":
        _1_Executive_Summary.app(df, segmented_df, model_package, model_metrics)
    elif selected == "Customer Analytics":
        _2_Customer_Analytics.app(df, segmented_df, model_package, model_metrics)
    elif selected == "Sales Dashboard":
        _3_Sales_Dashboard.app(df, segmented_df, model_package, model_metrics)
    elif selected == "Prediction Center":
        _4_Prediction_Center.app(df, segmented_df, model_package, model_metrics)
    elif selected == "Customer Segmentation":
        _5_Customer_Segmentation.app(df, segmented_df, model_package, model_metrics)
    elif selected == "Model Performance":
        _6_Model_Performance.app(df, segmented_df, model_package, model_metrics)
    elif selected == "Data Explorer":
        _7_Data_Explorer.app(df, segmented_df, model_package, model_metrics)


if __name__ == "__main__":
    main()