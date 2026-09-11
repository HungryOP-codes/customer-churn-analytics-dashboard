import plotly.express as px
import pandas as pd
import streamlit as st


def app(df, segmented_df, model_package, model_metrics):
    st.markdown("## 🧩 Customer Clustering & Persona Segmentation")
    st.markdown(
        "<p style='color: #6b7280; font-size: 0.95rem;'>Unsupervised machine learning (K-Means) clustering based on customer tenure, monthly charges, and lifetime billing patterns.</p>",
        unsafe_allow_html=True,
    )

    segmented = segmented_df.copy()
    if 'Risk_Group' not in segmented.columns:
        segmented['Risk_Group'] = 'Standard'

    # Compute Cluster Profiles
    summary = segmented.groupby('Cluster').agg(
        Count=('tenure', 'count'),
        AvgTenure=('tenure', 'mean'),
        AvgMonthly=('MonthlyCharges', 'mean'),
        AvgTotal=('TotalCharges', 'mean'),
        ChurnRate=('Churn', 'mean'),
    ).reset_index()

    persona_map = (
        segmented.groupby('Cluster')['Segment_Persona'].first().to_dict()
        if 'Segment_Persona' in segmented.columns
        else {c: f"Cluster {c}" for c in summary['Cluster']}
    )
    risk_map = (
        segmented.groupby('Cluster')['Risk_Group'].first().to_dict()
        if 'Risk_Group' in segmented.columns
        else {c: "Standard" for c in summary['Cluster']}
    )

    # Persona Cards
    st.markdown("#### 👥 Segment Persona Profiles")
    cols = st.columns(len(summary))

    for idx, row in summary.iterrows():
        c_id = int(row['Cluster'])
        name = persona_map.get(c_id, f"Cluster {c_id}")
        risk_text = risk_map.get(c_id, "Standard")
        badge = "badge-high" if "High" in risk_text else ("badge-med" if "Medium" in risk_text else "badge-low")
        badge_icon = "⚠️" if "High" in risk_text else ("🟡" if "Medium" in risk_text else "🟢")

        with cols[idx]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span class="badge-pill {badge}">{badge_icon} {risk_text}</span>
                    <p class="metric-title" style="margin-top: 0.5rem;">{name}</p>
                    <p style="font-size: 1.35rem; font-weight: 700; margin: 0.2rem 0;">{int(row['Count']):,} Accounts</p>
                    <p style="font-size: 0.85rem; color: #6b7280; margin: 0;">
                        ⏱️ Avg Tenure: <b>{row['AvgTenure']:.1f} mos</b><br/>
                        💵 Avg Monthly: <b>${row['AvgMonthly']:.2f}</b><br/>
                        📉 Churn Rate: <b>{row['ChurnRate']:.1%}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

    # Cluster Visualizations
    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown("<div class='section-header'>Cluster Scatter (Tenure vs. Monthly Charges)</div>", unsafe_allow_html=True)
        segmented['Cluster_Label'] = segmented['Cluster'].apply(lambda x: f"Cluster {x}: {persona_map.get(x, f'Cluster {x}')}")
        fig_scatter = px.scatter(
            segmented,
            x='tenure',
            y='MonthlyCharges',
            color='Cluster_Label',
            opacity=0.65,
            color_discrete_sequence=['#ef4444', '#f59e0b', '#10b981'],
            labels={'tenure': 'Tenure (Months)', 'MonthlyCharges': 'Monthly Charges ($)', 'Cluster_Label': 'Segment'},
        )
        fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_scatter, width="stretch")

    with col2:
        st.markdown("<div class='section-header'>Customer Volume by Segment</div>", unsafe_allow_html=True)
        fig_bar = px.bar(
            summary,
            x='Cluster',
            y='Count',
            color='Cluster',
            text=summary['Count'].apply(lambda x: f"{x:,}"),
            color_discrete_sequence=['#ef4444', '#f59e0b', '#10b981'],
            labels={'Count': 'Customer Count', 'Cluster': 'Cluster ID'},
        )
        fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=380, showlegend=False)
        st.plotly_chart(fig_bar, width="stretch")

    with st.expander("📌 Marketing & Product Personalization Strategy", expanded=False):
        st.markdown(
            """
            * **Cluster 0 (High Spend, Flight Risk)**: Highly engaged users paying premium rates on short tenure. Target with dedicated proactive onboarding calls and annual contract discounts.
            * **Cluster 1 (Balanced Users)**: Stable, moderate-spend accounts. Ideal target for value-add cross-selling (online security, cloud backup).
            * **Cluster 2 (Loyal Champions)**: High tenure and lifetime value. Enroll in VIP loyalty reward programs to maximize brand advocacy and reduce passive slippage.
            """
        )
