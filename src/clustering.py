import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import SEGMENTED_DATA_PATH, CLEANED_DATA_PATH
from utils.helpers import ensure_directories
from src.data_cleaning import load_raw_data, clean_data


def load_cleaned_for_clustering():
    ensure_directories()
    try:
        df = pd.read_csv(CLEANED_DATA_PATH)
    except FileNotFoundError:
        raw_df = load_raw_data()
        df = clean_data(raw_df)
        df.to_csv(CLEANED_DATA_PATH, index=False)
    return df


def assign_risk_groups(cluster_df: pd.DataFrame) -> pd.DataFrame:
    cluster_summary = cluster_df.groupby('Cluster').agg(
        MonthlyCharges=('MonthlyCharges', 'mean'),
        tenure=('tenure', 'mean'),
        TotalCharges=('TotalCharges', 'mean'),
    )
    score = (cluster_summary['MonthlyCharges'] / (cluster_summary['tenure'] + 1)).sort_values(ascending=False)
    risk_order = ['High Risk', 'Medium Risk', 'Low Risk']
    mapping = {cluster: risk_order[idx] for idx, cluster in enumerate(score.index)}
    cluster_df['Risk_Group'] = cluster_df['Cluster'].map(mapping)
    return cluster_df


def perform_clustering(n_clusters: int = 3):
    df = load_cleaned_for_clustering()
    clean_df = df.copy()
    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X = clean_df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clean_df['Cluster'] = kmeans.fit_predict(X_scaled)
    clean_df = assign_risk_groups(clean_df)
    ensure_directories()
    clean_df.to_csv(SEGMENTED_DATA_PATH, index=False)
    return clean_df


def main():
    segmented = perform_clustering()
    print('Cluster counts:')
    print(segmented['Risk_Group'].value_counts())
    print(f'Segmented customers saved to {SEGMENTED_DATA_PATH}')


if __name__ == '__main__':
    main()
