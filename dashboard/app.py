import sys
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Customer Churn Prediction and Sales Dashboard",
    page_icon="📊",
    layout="wide"
)
# -------------------------
# PATH SETUP
# -------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import (
    CLEANED_DATA_PATH,
    SEGMENTED_DATA_PATH,
    MODEL_PATH,
    METRICS_PATH
)

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
    _7_Data_Explorer
)

# -------------------------
# LOAD DATA
# -------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(CLEANED_DATA_PATH)

    segmented = pd.read_csv(SEGMENTED_DATA_PATH)

    model_package = joblib.load(MODEL_PATH)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        model_metrics = json.load(f)

    return df, segmented, model_package, model_metrics

# -------------------------
# MAIN
# -------------------------

def main():

    with st.sidebar:
        st.title("Customer Churn Dashboard")
        selected = option_menu(
            "Navigation",
            [
                "Executive Summary",
                "Customer Analytics",
                "Sales Dashboard",
                "Prediction Center",
                "Customer Segmentation",
                "Model Performance",
                "Data Explorer",
            ],
            icons=[
                "bar-chart",
                "people",
                "graph-up",
                "cpu",
                "diagram-3",
                "robot",
                "table"
            ],
            menu_icon="cast",
            default_index=0
        )

    df, segmented_df, model_package, model_metrics = load_data()

    if selected == "Executive Summary":
        _1_Executive_Summary.app(
            df,
            segmented_df,
            model_package,
            model_metrics
        )

    elif selected == "Customer Analytics":
        _2_Customer_Analytics.app(
            df,
            segmented_df,
            model_package,
            model_metrics
        )

    elif selected == "Sales Dashboard":
        _3_Sales_Dashboard.app(
            df,
            segmented_df,
            model_package,
            model_metrics
        )

    elif selected == "Prediction Center":
        _4_Prediction_Center.app(
            df,
            segmented_df,
            model_package,
            model_metrics
        )

    elif selected == "Customer Segmentation":
        _5_Customer_Segmentation.app(
            df,
            segmented_df,
            model_package,
            model_metrics
        )

    elif selected == "Model Performance":
        _6_Model_Performance.app(
            df,
            segmented_df,
            model_package,
            model_metrics
        )

    elif selected == "Data Explorer":
        _7_Data_Explorer.app(
            df,
            segmented_df,
            model_package,
            model_metrics
        )

if __name__ == "__main__":
    main()