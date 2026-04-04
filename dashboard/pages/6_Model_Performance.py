import pandas as pd
import plotly.express as px
import streamlit as st


def app(df, segmented_df, model_package, model_metrics: dict):
    st.title('Model Performance')
    best = model_metrics.get('best_model_metrics', {})
    st.subheader(f"Best model: {model_metrics.get('best_model', 'N/A')}")

    metrics = {
        'Accuracy': best.get('accuracy', 0),
        'Precision': best.get('precision', 0),
        'Recall': best.get('recall', 0),
        'F1 Score': best.get('f1_score', 0),
        'ROC AUC': best.get('roc_auc', 0),
    }

    cols = st.columns(5)
    for idx, (label, value) in enumerate(metrics.items()):
        cols[idx].metric(label, f"{value:.2f}")

    st.subheader('Confusion Matrix')
    matrix = best.get('confusion_matrix', [[0, 0], [0, 0]])
    cm_df = pd.DataFrame(matrix, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])
    st.table(cm_df)

    roc = best.get('roc_curve', {})
    if roc:
        roc_df = pd.DataFrame({'False Positive Rate': roc.get('fpr', []), 'True Positive Rate': roc.get('tpr', [])})
        fig_roc = px.line(roc_df, x='False Positive Rate', y='True Positive Rate', title='ROC Curve')
        fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
        st.plotly_chart(fig_roc, width="stretch")

    st.subheader('Feature Importance')
    importance = model_metrics.get('feature_importance', {})
    importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
    if not importance_df.empty:
        fig_importance = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance')
        st.plotly_chart(fig_importance, width="stretch")

    st.markdown('---')
    st.write('Model metrics and importance scores are automatically generated from the trained churn models. Review this page before running the prediction center.')
