import sys
import json
import importlib.util
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# -----------------------------
# PATH SETUP
# -----------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import (
    CLEANED_DATA_PATH,
    SEGMENTED_DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
)

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Customer Churn Prediction and Sales Dashboard",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# PAGE FILES
# -----------------------------

PAGES_DIR = Path(__file__).resolve().parent / "pages"

PAGE_FILES = [
    "1_Executive_Summary.py",
    "2_Customer_Analytics.py",
    "3_Sales_Dashboard.py",
    "4_Prediction_Center.py",
    "5_Customer_Segmentation.py",
    "6_Model_Performance.py",
    "7_Data_Explorer.py",
]

PAGE_TITLES = [
    "Executive Summary",
    "Customer Analytics",
    "Sales Dashboard",
    "Prediction Center",
    "Customer Segmentation",
    "Model Performance",
    "Data Explorer",
]

ICONS = [
    "bar-chart-line",
    "people-fill",
    "graph-up",
    "cpu",
    "diagram-3",
    "robot",
    "table",
]

# -----------------------------
# LOAD PAGE MODULE
# -----------------------------

def load_page_module(filename):

    module_path = PAGES_DIR / filename

    spec = importlib.util.spec_from_file_location(
        filename.replace(".py", ""),
        module_path,
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# -----------------------------
# LOAD DATA
# -----------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(CLEANED_DATA_PATH)

    segmented_df = pd.read_csv(SEGMENTED_DATA_PATH)

    model_package = joblib.load(MODEL_PATH)

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        model_metrics = json.load(file)

    return df, segmented_df, model_package, model_metrics


# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------

st.sidebar.title("Customer Churn Dashboard")

selected = option_menu(
    menu_title="Pages",
    options=PAGE_TITLES,
    icons=ICONS,
    menu_icon="cast",
    default_index=0,
)

# -----------------------------
# LOAD DATA
# -----------------------------

try:

    df, segmented_df, model_package, model_metrics = load_data()

except Exception as e:

    st.error("Data loading failed")

    st.exception(e)

    st.stop()

# -----------------------------
# LOAD SELECTED PAGE
# -----------------------------

try:

    page_index = PAGE_TITLES.index(selected)

    page_module = load_page_module(
        PAGE_FILES[page_index]
    )

    page_module.app(
        df,
        segmented_df,
        model_package,
        model_metrics,
    )

except Exception as e:

    st.error("Page failed to load")

    st.exception(e)