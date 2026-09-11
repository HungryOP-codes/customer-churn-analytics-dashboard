import pandas as pd
import plotly.express as px
import streamlit as st


def app(df, segmented_df, model_package, model_metrics: dict):
    st.markdown("## 🤖 Machine Learning Model Benchmarks & Explainability")
    st.markdown(
        "<p style='color: #6b7280; font-size: 0.95rem;'>Evaluation metrics, cross-model benchmark comparisons, confusion matrices, and predictive feature importance.</p>",
        unsafe_allow_html=True,
    )

    best_name = model_metrics.get('best_model', 'Logistic Regression')
    best = model_metrics.get('best_model_metrics', {})
    model_results = model_metrics.get('model_results', {})

    # Top Champion Ribbon
    st.markdown(
        f"""
        <div class="metric-card" style="margin-bottom: 1.25rem;">
            <span class="badge-pill badge-high">Production Champion</span>
            <span style="font-size: 1.15rem; font-weight: 700; color: #4f46e5;">{best_name}</span>
            <p style="font-size: 0.85rem; color: #6b7280; margin-top: 0.35rem;">
                Selected based on optimal balance of precision and recall (highest validation F1 Score: <b>{best.get('f1_score', 0):.3f}</b>).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Multi-Model Benchmark Comparison Table
    st.markdown("<div class='section-header'>Cross-Algorithm Leaderboard</div>", unsafe_allow_html=True)
    benchmark_rows = []
    for model_title, m_data in model_results.items():
        benchmark_rows.append({
            'Model Architecture': f"🏆 {model_title}" if model_title == best_name else model_title,
            'Accuracy': f"{m_data.get('accuracy', 0):.2%}",
            'Precision': f"{m_data.get('precision', 0):.2%}",
            'Recall': f"{m_data.get('recall', 0):.2%}",
            'F1 Score': f"{m_data.get('f1_score', 0):.4f}",
            'ROC AUC': f"{m_data.get('roc_auc', 0):.4f}",
        })

    if benchmark_rows:
        bench_df = pd.DataFrame(benchmark_rows)
        st.dataframe(bench_df, width="stretch", hide_index=True)

    st.markdown("<div style='margin: 1.5rem 0 0.5rem 0;'></div>", unsafe_allow_html=True)

    # Evaluation Charts: Confusion Matrix & ROC Curve
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<div class='section-header'>Confusion Matrix ({best_name})</div>", unsafe_allow_html=True)
        matrix = best.get('confusion_matrix', [[0, 0], [0, 0]])
        fig_cm = px.imshow(
            matrix,
            text_auto=True,
            x=['Predicted Retained (0)', 'Predicted Churned (1)'],
            y=['Actual Retained (0)', 'Actual Churned (1)'],
            color_continuous_scale='Blues',
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        )
        fig_cm.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, width="stretch")

    with col2:
        st.markdown(f"<div class='section-header'>ROC Curve (AUC: {best.get('roc_auc', 0):.3f})</div>", unsafe_allow_html=True)
        roc = best.get('roc_curve', {})
        if roc and roc.get('fpr'):
            roc_df = pd.DataFrame({'False Positive Rate': roc.get('fpr', []), 'True Positive Rate': roc.get('tpr', [])})
            fig_roc = px.line(roc_df, x='False Positive Rate', y='True Positive Rate', color_discrete_sequence=['#4f46e5'])
            fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='#9ca3af'))
            fig_roc.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
            st.plotly_chart(fig_roc, width="stretch")
        else:
            st.info("ROC Curve coordinates available upon model retraining.")

    # Feature Importance
    st.markdown("<div class='section-header'>Top Predictive Feature Drivers (Global Explainability)</div>", unsafe_allow_html=True)
    importance = model_metrics.get('feature_importance', {})
    if importance:
        importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale=['#c7d2fe', '#4338ca'],
        )
        fig_importance.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_importance, width="stretch")

    st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
    st.caption("ℹ️ **Feature Insight**: Contract type and tenure demonstrate the strongest predictive signals for determining whether an account is likely to churn.")
