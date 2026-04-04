# Customer Churn Prediction and Sales Dashboard

## Project Description

This repository contains a complete end-to-end customer churn prediction and sales analytics project built in Python. It includes data cleaning, feature engineering, machine learning model training, customer segmentation, and a multi-page Streamlit dashboard for business-facing insights.

## Features

- Data cleaning and preprocessing pipeline
- Feature engineering with derived customer metrics
- Machine learning model training with Logistic Regression, Random Forest, and XGBoost
- Automatic model selection based on F1 score
- Customer segmentation using K-Means clustering
- Prediction center for live churn scoring
- Multi-page Streamlit dashboard with KPIs, analytics, sales views, segmentation, model performance, and an interactive data explorer
- Prepared for Streamlit Cloud deployment

## Installation

1. Clone or download this project into the directory `Customer_Churn_Project`.
2. Create a Python environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Information

The project uses the Kaggle dataset `Telco Customer Churn Dataset` with file name `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

### Dataset placement

- Preferred: place the dataset in `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Optional: if Kaggle API is installed and configured, the pipeline will attempt to download the dataset automatically.

## How to Run

### 1. Clean the data

```bash
python src/data_cleaning.py
```

### 2. Prepare features and train models

```bash
python src/train_model.py
```

### 3. Generate customer segmentation

```bash
python src/clustering.py
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

## Streamlit Dashboard Pages

1. Executive Summary
2. Customer Analytics
3. Sales Dashboard
4. Prediction Center
5. Customer Segmentation
6. Model Performance
7. Data Explorer

## Deployment Instructions

- Ensure `requirements.txt` is present
- Verify `dashboard/app.py` runs locally
- For deployment to Streamlit Cloud, push this repository to GitHub and connect the repo to Streamlit Cloud
- Set the startup command: `streamlit run dashboard/app.py`

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- plotly
- streamlit
- streamlit-option-menu
- streamlit-aggrid
- joblib

## GitHub Integration

```bash
git init
git add .
git commit -m "Customer Churn Prediction and Sales Dashboard project"
git branch -M main
git remote add origin REPOSITORY_URL
git push -u origin main
```

> Replace `REPOSITORY_URL` with your GitHub repository URL.
