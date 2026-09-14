# Customer Churn Prediction & Commercial Sales Dashboard

An end-to-end Big Data Analytics and Machine Learning application designed to identify customer attrition risk, segment accounts by commercial behavior, and deliver actionable retention intelligence through an interactive Streamlit dashboard.

> **Project Summary**: An independent Customer Churn Prediction & Commercial Sales Dashboard project featuring a leakage-free machine learning pipeline, multi-model classification benchmarks, unsupervised customer segmentation, and an interactive 7-page Streamlit analytics application.

---

## 📌 Project Overview

Customer retention is critical to the recurring revenue model of subscription businesses. Acquiring a new customer often costs significantly more than retaining an existing one. This project provides a complete analytics pipeline that translates raw customer telecom records into strategic retention decisions:

* **Data Cleaning & Ingestion**: Automated handling of missing charges, robust type conversion, and customer identifier tracking.
* **Leakage-Free Feature Engineering**: Derivation of tenure cohorts, charge brackets, and Customer Lifetime Value (CLV) without contaminating input features with target labels.
* **Predictive Machine Learning**: Training and benchmarking multiple classification models to accurately identify at-risk customers.
* **Customer Segmentation**: Unsupervised K-Means clustering to categorize customers into distinct behavioral personas.
* **Interactive Dashboard**: A modular 7-page Streamlit web application providing executives, analysts, and sales teams with dynamic insights and live scoring tools.

---

## ✨ Features

* **Zero-Leakage ML Pipeline**: Derived risk indicators are computed strictly from behavioral attributes (`tenure`, `MonthlyCharges`), ensuring unbiased evaluation and true real-world generalization.
* **Multi-Model Benchmark**: Evaluates Logistic Regression, Random Forest, and XGBoost on identical train/test splits and persists the best performing model.
* **Unsupervised Clustering ($K=3$)**: Groups accounts into three distinct personas: *High Spend Flight Risk*, *Balanced Moderate Users*, and *Long-Term Loyal Champions*.
* **Live Prediction Center**:
  * **Single Customer Simulator**: Enter customer parameters to view real-time churn probability on an interactive gauge with customized retention recommendations.
  * **Batch CSV Scoring**: Upload a customer CSV file (or sample the existing dataset) to score multiple accounts at once and download the resulting predictions.
* **Executive & Sales Analytics**: Visualizes Customer Lifetime Value, revenue lost to churn, contract mix, and customer tenure trends.
* **Data Warehouse Explorer**: Search records by Customer ID, filter across multiple dimensions, and export sanitized datasets directly to CSV.
* **Automated Audit Suite**: Includes an automated Streamlit `AppTest` suite verifying that all pages load with zero exceptions and that all metrics are dynamically calculated.

---

## 🛠️ Tech Stack

* **Language**: Python 3.10+
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn, XGBoost, Joblib
* **Data Visualization**: Plotly, Matplotlib, Seaborn
* **Web Dashboard**: Streamlit, Streamlit Option Menu
* **Testing & Quality**: Streamlit Testing Framework (`AppTest`)

---

## 🧠 Machine Learning Models & Verified Performance

Models were trained on an 80/20 stratified split of the 7,043 Telco customer records ($N_{\text{train}} = 5,634$, $N_{\text{test}} = 1,409$). Model selection was evaluated using $F_1$-score to balance precision and recall given the class distribution (~26.5% churn).

The following metrics were verified directly from the serialized project metrics file (`models/model_performance.json`):

| Model Architecture | Accuracy | Precision | Recall | $F_1$ Score | ROC AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **79.28%** | **65.07%** | **47.33%** | **0.5480** | **0.8342** | **Selected Champion** |
| **Random Forest** | 77.86% | 60.40% | 48.13% | 0.5357 | 0.8104 | Evaluated Candidate |
| **XGBoost** | 77.71% | 59.43% | 50.53% | 0.5462 | 0.8230 | Evaluated Candidate |

### Top Predictive Feature Drivers
Based on model coefficients and feature importance rankings:
1. `Contract`: Strongest retention indicator (annual and multi-year contracts show minimal churn).
2. `SeniorCitizen`: Demographic sensitivity signal.
3. `MonthlyCharges` & `TotalCharges`: Direct indicator of fee sensitivity.
4. `tenure`: Accounts beyond 24 months demonstrate significantly higher retention stability.

---

## 🖥️ Dashboard Pages

The Streamlit dashboard (`dashboard/app.py`) is organized into seven specialized views:

1. **Executive Summary** (`dashboard/views/_1_Executive_Summary.py`): Overview of core business metrics including Total Customers (7,043), Active Accounts (5,174), Churned Accounts (1,869), Churn Rate (26.54%), Total Realized Revenue ($16.05M), Contract Distribution, and Cohort Revenue Trends.
2. **Customer Analytics** (`dashboard/views/_2_Customer_Analytics.py`): Cohort filter bar (Contract, Payment Method, Senior Citizen status) with baseline comparisons and a bivariate tenure vs. monthly charges risk map.
3. **Sales Dashboard** (`dashboard/views/_3_Sales_Dashboard.py`): Tracks revenue at risk from churned customers ($2.86M), Average Revenue Per User (ARPU: $64.76/mo), payment channel breakdown, and the Top-10 Customer Lifetime Value leaderboard.
4. **Prediction Center** (`dashboard/views/_4_Prediction_Center.py`):
   * *Single Account Simulator*: Interactive inputs returning churn probability, risk classification badge, and prescriptive retention actions.
   * *Bulk Batch Scoring*: Scorer for customer CSV files with batch summaries and CSV export.
5. **Customer Segmentation** (`dashboard/views/_5_Customer_Segmentation.py`): K-Means cluster analysis displaying financial profiles, account volume, and marketing playbooks for each persona.
6. **Model Performance** (`dashboard/views/_6_Model_Performance.py`): Side-by-side benchmark table, confusion matrix heatmap, ROC curve, and feature importance rankings.
7. **Data Explorer** (`dashboard/views/_7_Data_Explorer.py`): Searchable table with customer ID lookup, churn filters, contract filters, progress bars, and CSV data export.

---

## 📊 Dataset Information

* **Dataset Name**: Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
* **Source**: Kaggle / IBM Business Analytics
* **Volume**: 7,043 rows, 21 columns
* **Features Included**:
  * *Demographics*: Gender, Senior Citizen, Partner, Dependents
  * *Services*: Phone Service, Multiple Lines, Internet Service, Online Security, Tech Support, Streaming Services
  * *Account Information*: Tenure (months), Contract, Paperless Billing, Payment Method, Monthly Charges, Total Charges
  * *Target*: Churn (Yes / No)
* **Automated Provisioning**: The ingestion pipeline automatically downloads the dataset if not locally present, with an offline synthetic generator fallback so setup never fails.

---

## 📁 Project Structure

```
Customer-Churn-prediction-and-sales-dashboard/
├── README.md                           # Project documentation
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git ignore rules
│
├── utils/                              # Utility configurations and helpers
│   ├── config.py                       # Central file paths and constants
│   └── helpers.py                      # Multi-source dataset downloader and fallback generator
│
├── src/                                # Machine learning & data pipeline
│   ├── data_cleaning.py                # Data ingestion, missing value imputation, type mapping
│   ├── feature_engineering.py          # Derived features, binning, and label encoding
│   ├── train_model.py                  # Model training, F1 evaluation, and artifact saving
│   ├── clustering.py                   # K-Means clustering and persona assignment
│   └── predict.py                      # Single and batch prediction inference logic
│
├── dashboard/                          # Streamlit user interface
│   ├── __init__.py
│   ├── app.py                          # Streamlit application entry point
│   └── views/                          # Modular dashboard views
│       ├── __init__.py
│       ├── _1_Executive_Summary.py     # Executive KPI cockpit
│       ├── _2_Customer_Analytics.py    # Demographic and behavioral cohort views
│       ├── _3_Sales_Dashboard.py       # Revenue-at-risk and CLV leaderboard
│       ├── _4_Prediction_Center.py     # Real-time simulator and batch CSV scoring
│       ├── _5_Customer_Segmentation.py # K-Means cluster personas
│       ├── _6_Model_Performance.py     # Model benchmarks and confusion matrix
│       └── _7_Data_Explorer.py         # Searchable table and CSV export
│
└── tests/                              # Automated testing suite
    ├── test_audit.py                   # Complete 7-suite verification test
    └── audit_code_scan.py              # Anti-hardcoding metric code scanner
```

---

## ⚙️ Installation (Windows)

### 1. Clone the Repository
```powershell
git clone https://github.com/HungryOP-codes/Customer-Churn-prediction-and-sales-dashboard.git
cd Customer-Churn-prediction-and-sales-dashboard
```

### 2. Create and Activate a Virtual Environment
```powershell
# Using the Python launcher
py -m venv venv

# Activate in PowerShell
.\venv\Scripts\Activate.ps1
```
*(If using Command Prompt, run `.\venv\Scripts\activate.bat` instead)*

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1: Clean and Prepare Data
```powershell
py src/data_cleaning.py
```

### Step 2: Train Machine Learning Models
```powershell
py src/train_model.py
```

### Step 3: Run Customer Segmentation
```powershell
py src/clustering.py
```

### Step 4: Launch the Streamlit Dashboard
```powershell
py -m streamlit run dashboard/app.py
```
The application will open in your default browser at `http://localhost:8501`.

### Step 5: Run Automated Verification Tests (Optional)
To verify that all 7 pages load cleanly and that all dynamic KPI calculations match the dataset:
```powershell
py tests/test_audit.py
```

---

## 📸 Screenshots

### 1. Executive Summary
*High-level KPI cockpit, contract distribution donut, and annualized churn rates.*

![Executive Summary](screenshots/Screenshot%202026-09-11%20234548.png)

---

### 2. Sales Dashboard
*Commercial revenue intelligence, contract revenue concentration, payment channels, and Customer Lifetime Value (CLV) trajectory.*

#### Commercial Revenue & Payment Channels
![Sales Revenue & Payment Channels](screenshots/Screenshot%202026-09-11%20234919.png)

#### Revenue Trajectory & High-Value Accounts Leaderboard
![Revenue Trajectory & High-Value Accounts](screenshots/Screenshot%202026-09-11%20234935.png)

---

### 3. Prediction Center
*Real-time churn risk simulator, dynamic probability gauge, risk classification badge, and prescriptive retention playbook.*

#### Customer Simulation Input Form
![Prediction Simulator Input](screenshots/Screenshot%202026-09-11%20234629.png)

#### Real-Time Churn Gauge & Retention Playbook
![Prediction Results Gauge & Playbook](screenshots/Screenshot%202026-09-11%20234648.png)

---

### 4. Customer Segmentation
*Unsupervised K-Means clustering ($K=3$), behavioral persona cards, and bivariate cluster scatter analysis.*

#### Segment Persona Profiles & Cluster Scatter Plot
![Customer Segmentation Personas](screenshots/Screenshot%202026-09-11%20234843.png)

#### Cluster Distribution & Segment Volume Analysis
![Customer Segmentation Scatter & Volume](screenshots/Screenshot%202026-09-11%20234902.png)

---

### 5. Model
*Cross-algorithm leaderboard benchmark, confusion matrix heatmap, ROC curve, and global feature importance ranking.*

#### Model Benchmark Leaderboard & Confusion Matrix
![Model Benchmarks Leaderboard](screenshots/Screenshot%202026-09-11%20234718.png)

#### Confusion Matrix & Top Predictive Feature Drivers
![Feature Importance & Model Evaluation](screenshots/Screenshot%202026-09-11%20234811.png)

---

## 📈 Key Business Insights

1. **Contract Type is the Primary Churn Driver**:
   * Customers on **Month-to-month contracts** exhibit a **42.71% churn rate**, accounting for over 75% of total lost revenue.
   * Customers with **One-year (11.27%)** and **Two-year (2.83%)** contracts exhibit high stability, demonstrating that long-term incentives significantly reduce attrition.
2. **Payment Channel Sensitivity**:
   * Customers using **Electronic check** churn at **45.29%**, compared to under **17%** for those using automated credit card or bank transfer payments.
3. **The 24-Month Retention Milestone**:
   * Over 50% of customer churn occurs within the first 12 months.
   * Once an account reaches **24 months of tenure**, retention stabilizes above **80%**, and lifetime value increases by more than 3x.

---

## 🔮 Future Improvements

* **Explainable AI (XAI)**: Integrate SHAP waterfall plots for individual prediction explanations in the Prediction Center.
* **Real-Time Event Streaming**: Connect ingestion to Apache Kafka or Spark Streaming to monitor live usage drops and payment failures.
* **Survival Analysis**: Implement Cox Proportional Hazards modeling to predict expected time-to-churn in addition to binary probability.
* **Automated Webhooks**: Connect high-risk prediction triggers to CRM endpoints (e.g., Slack or email alerts) for immediate customer success outreach.
