import pandas as pd
import plotly.express as px
import streamlit as st


def app(df: pd.DataFrame, segmented_df: pd.DataFrame, model_package, model_metrics: dict):
    st.markdown("## 💰 Sales & Revenue Intelligence")
    st.markdown(
        "<p style='color: #6b7280; font-size: 0.95rem;'>Commercial performance, customer lifetime value (CLV) realizations, and revenue-at-risk analysis.</p>",
        unsafe_allow_html=True,
    )

    df = df.copy()
    if 'Customer_Lifetime_Value' not in df.columns:
        df['Customer_Lifetime_Value'] = df['MonthlyCharges'] * df['tenure']
    df['Revenue'] = df['Customer_Lifetime_Value']

    total_revenue = float(df['Revenue'].sum())
    arpu = float(df['MonthlyCharges'].mean())
    revenue_churned = float(df[df['Churn'] == 1]['Revenue'].sum())
    pct_revenue_lost = revenue_churned / total_revenue if total_revenue > 0 else 0

    # Sales KPI Cards
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Cumulative Revenue", f"${total_revenue:,.0f}")
    with s2:
        st.metric("Average Revenue Per User (ARPU)", f"${arpu:,.2f}/mo")
    with s3:
        st.metric("Lost Revenue (Churned)", f"${revenue_churned:,.0f}", delta=f"{pct_revenue_lost:.1%} of Total", delta_color="inverse")
    with s4:
        active_rev = total_revenue - revenue_churned
        st.metric("Retained Revenue Base", f"${active_rev:,.0f}", delta=f"{(1 - pct_revenue_lost):.1%} Secured")

    st.markdown("<div style='margin: 1.25rem 0;'></div>", unsafe_allow_html=True)

    # Revenue Breakdown Charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-header'>Revenue Concentration by Contract</div>", unsafe_allow_html=True)
        revenue_contract = df.groupby('Contract')['Revenue'].sum().reset_index().sort_values(by='Revenue', ascending=False)
        revenue_contract['Pct'] = revenue_contract['Revenue'] / total_revenue
        fig_contract = px.bar(
            revenue_contract,
            x='Contract',
            y='Revenue',
            color='Contract',
            color_discrete_sequence=['#4f46e5', '#06b6d4', '#10b981'],
            text=revenue_contract['Revenue'].apply(lambda x: f"${x:,.0f}"),
        )
        fig_contract.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, showlegend=False)
        st.plotly_chart(fig_contract, width="stretch")

    with col2:
        st.markdown("<div class='section-header'>Revenue Share by Payment Channel</div>", unsafe_allow_html=True)
        revenue_payment = df.groupby('PaymentMethod')['Revenue'].sum().reset_index().sort_values(by='Revenue', ascending=False)
        fig_payment = px.pie(
            revenue_payment,
            names='PaymentMethod',
            values='Revenue',
            hole=0.5,
            color_discrete_sequence=['#6366f1', '#3b82f6', '#14b8a6', '#f59e0b'],
        )
        fig_payment.update_traces(textposition='inside', textinfo='percent+label')
        fig_payment.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, showlegend=False)
        st.plotly_chart(fig_payment, width="stretch")

    # Revenue by Tenure Cohort Line Chart
    st.markdown("<div class='section-header'>Revenue Trajectory Across Customer Lifespan</div>", unsafe_allow_html=True)
    tenure_bins = [0, 6, 12, 24, 48, df['tenure'].max() + 1]
    tenure_labels = ['0-6 mos', '7-12 mos', '13-24 mos', '25-48 mos', '49+ mos']
    df['Tenure_Group'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels, include_lowest=True)
    revenue_trend = df.groupby('Tenure_Group', observed=False)['Revenue'].sum().reset_index()
    fig_trend = px.line(
        revenue_trend,
        x='Tenure_Group',
        y='Revenue',
        markers=True,
        labels={'Revenue': 'Total Revenue ($)', 'Tenure_Group': 'Tenure Cohort'},
        color_discrete_sequence=['#10b981'],
    )
    fig_trend.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
    st.plotly_chart(fig_trend, width="stretch")

    # Top Revenue Accounts Leaderboard
    st.markdown("<div class='section-header'>🏆 Top 10 High-Value Accounts (CLV)</div>", unsafe_allow_html=True)
    display_cols = ['customerID', 'Contract', 'PaymentMethod', 'tenure', 'MonthlyCharges', 'Customer_Lifetime_Value', 'Churn']
    available_cols = [c for c in display_cols if c in df.columns]
    top_customers = df.nlargest(10, 'Customer_Lifetime_Value')[available_cols].copy()
    if 'Churn' in top_customers.columns:
        top_customers['Churn'] = top_customers['Churn'].map({0: '✅ Retained', 1: '⚠️ Churned'})

    st.dataframe(
        top_customers.reset_index(drop=True),
        column_config={
            "Customer_Lifetime_Value": st.column_config.NumberColumn("Lifetime Value", format="$%.2f"),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly Charges", format="$%.2f"),
            "tenure": st.column_config.NumberColumn("Tenure (Months)"),
            "customerID": st.column_config.TextColumn("Customer ID"),
        },
        width="stretch",
    )
