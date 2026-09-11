import pandas as pd
import plotly.express as px
import streamlit as st


def app(df: pd.DataFrame, segmented_df: pd.DataFrame, model_package, model_metrics: dict):
    st.markdown("## 👥 Customer Cohort & Behavioral Analytics")
    st.markdown(
        "<p style='color: #6b7280; font-size: 0.95rem;'>Deep-dive analysis of customer behavior, billing methods, demographic patterns, and attrition correlation.</p>",
        unsafe_allow_html=True,
    )

    df = df.copy()
    if 'Gender' not in df.columns:
        df['Gender'] = df['gender'].map({0: 'Female', 1: 'Male'})
    if 'Customer_Lifetime_Value' not in df.columns:
        df['Customer_Lifetime_Value'] = df['MonthlyCharges'] * df['tenure']
    df['Churn_Status'] = df['Churn'].map({0: 'Retained', 1: 'Churned'})

    # Filter Bar
    with st.expander("🔍 Filter Customer Cohort", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            contract_options = ['All Contracts'] + sorted(df['Contract'].dropna().unique().tolist())
            selected_contract = st.selectbox('Contract Type', contract_options, index=0)
        with col2:
            payment_options = ['All Payment Methods'] + sorted(df['PaymentMethod'].dropna().unique().tolist())
            selected_payment = st.selectbox('Payment Method', payment_options, index=0)
        with col3:
            senior_options = ['All Customers', 'Non-Senior Citizens (0)', 'Senior Citizens (1)']
            selected_senior = st.selectbox('Demographic Segment', senior_options, index=0)

    filtered = df.copy()
    if selected_contract != 'All Contracts':
        filtered = filtered[filtered['Contract'] == selected_contract]
    if selected_payment != 'All Payment Methods':
        filtered = filtered[filtered['PaymentMethod'] == selected_payment]
    if selected_senior == 'Senior Citizens (1)':
        filtered = filtered[filtered['SeniorCitizen'] == 1]
    elif selected_senior == 'Non-Senior Citizens (0)':
        filtered = filtered[filtered['SeniorCitizen'] == 0]

    # Cohort KPI Ribbon
    cohort_count = len(filtered)
    cohort_churn_rate = filtered['Churn'].mean() if cohort_count > 0 else 0
    baseline_churn_rate = df['Churn'].mean()
    churn_diff = cohort_churn_rate - baseline_churn_rate
    avg_cohort_monthly = filtered['MonthlyCharges'].mean() if cohort_count > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Cohort Size", f"{cohort_count:,} accounts", delta=f"{cohort_count / len(df):.1%} of Total")
    with m2:
        st.metric("Cohort Churn Rate", f"{cohort_churn_rate:.1%}", delta=f"{churn_diff:+.1%} vs Overall", delta_color="inverse")
    with m3:
        st.metric("Cohort Avg Monthly", f"${avg_cohort_monthly:,.2f}")
    with m4:
        st.metric("Total Cohort Revenue", f"${filtered['Customer_Lifetime_Value'].sum():,.0f}")

    st.markdown("<div style='margin: 1.25rem 0;'></div>", unsafe_allow_html=True)

    # Visualizations Row 1
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("<div class='section-header'>Attrition by Payment Method</div>", unsafe_allow_html=True)
        churn_payment = filtered.groupby('PaymentMethod')['Churn'].mean().reset_index().sort_values(by='Churn', ascending=False)
        fig_pay = px.bar(
            churn_payment,
            x='PaymentMethod',
            y='Churn',
            color='Churn',
            color_continuous_scale=['#10b981', '#ef4444'],
            text=churn_payment['Churn'].apply(lambda x: f"{x:.1%}"),
            labels={'Churn': 'Churn Rate', 'PaymentMethod': 'Payment Method'},
        )
        fig_pay.update_yaxes(tickformat='.0%')
        fig_pay.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_pay, width="stretch")

    with c_right:
        st.markdown("<div class='section-header'>Attrition by Gender & Senior Citizen Status</div>", unsafe_allow_html=True)
        churn_demo = filtered.groupby(['Gender', 'SeniorCitizen'])['Churn'].mean().reset_index()
        churn_demo['SeniorCitizen'] = churn_demo['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior'})
        fig_demo = px.bar(
            churn_demo,
            x='Gender',
            y='Churn',
            color='SeniorCitizen',
            barmode='group',
            text=churn_demo['Churn'].apply(lambda x: f"{x:.1%}"),
            color_discrete_sequence=['#6366f1', '#ec4899'],
            labels={'Churn': 'Churn Rate'},
        )
        fig_demo.update_yaxes(tickformat='.0%')
        fig_demo.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_demo, width="stretch")

    # Visualizations Row 2
    st.markdown("<div class='section-header'>Tenure vs Monthly Charges (Bivariate Risk Map)</div>", unsafe_allow_html=True)
    hover_cols = ['customerID', 'Contract', 'PaymentMethod', 'Customer_Lifetime_Value']
    available_hover = [c for c in hover_cols if c in filtered.columns]

    fig_scatter = px.scatter(
        filtered,
        x='tenure',
        y='MonthlyCharges',
        color='Churn_Status',
        color_discrete_map={'Retained': '#10b981', 'Churned': '#ef4444'},
        opacity=0.65,
        hover_data=available_hover,
        labels={'tenure': 'Tenure (Months)', 'MonthlyCharges': 'Monthly Charges ($)', 'Churn_Status': 'Status'},
    )
    fig_scatter.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=380)
    st.plotly_chart(fig_scatter, width="stretch")

    st.info(
        "💡 **Key Observation**: Notice the dense cluster of red points (Churned) in the upper-left quadrant: tenure < 12 months with monthly charges > $70. This confirms high upfront fee sensitivity."
    )