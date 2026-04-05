import pandas as pd
import plotly.express as px
import streamlit as st


def app(df: pd.DataFrame, segmented_df: pd.DataFrame, model_package, model_metrics: dict):
    st.title('Executive Summary')
    df = df.copy()
    df['Customer_Lifetime_Value'] = df['MonthlyCharges'] * df['tenure']
    df['Active'] = df['Churn'].apply(lambda x: 0 if x == 1 else 1)

    total_customers = len(df)
    churned_customers = int(df['Churn'].sum())
    active_customers = int(df['Active'].sum())
    churn_rate = churned_customers / total_customers if total_customers else 0
    retention_rate = 1 - churn_rate
    total_revenue = float((df['MonthlyCharges'] * df['tenure']).sum())
    average_monthly = float(df['MonthlyCharges'].mean())
    average_tenure = float(df['tenure'].mean())
    average_ltv = float(df['Customer_Lifetime_Value'].mean())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Customers', total_customers)
    col2.metric('Active Customers', active_customers)
    col3.metric('Churned Customers', churned_customers)
    col4.metric('Churn Rate', f'{churn_rate:.2%}')

    col5, col6, col7, col8 = st.columns(4)
    col5.metric('Total Revenue', f'${total_revenue:,.0f}')
    col6.metric('Avg Monthly Charges', f'${average_monthly:,.2f}')
    col7.metric('Avg Tenure (months)', f'{average_tenure:.1f}')
    col8.metric('Avg Customer LTV', f'${average_ltv:,.0f}')

    st.markdown('---')
    st.subheader('Customer Distribution')
    distribution = df.groupby('Contract').size().reset_index(name='Count')
    fig_distribution = px.pie(distribution, names='Contract', values='Count', title='Customer Contract Distribution')
    st.plotly_chart(fig_distribution, width="stretch")

    st.subheader('Churn Trend by Contract')
    churn_trend = df.groupby('Contract')['Churn'].mean().reset_index()
    fig_churn = px.bar(churn_trend, x='Contract', y='Churn', title='Churn Rate by Contract', labels={'Churn': 'Churn Rate'})
    fig_churn.update_yaxes(tickformat='.0%')
    st.plotly_chart(fig_churn, width="stretch")

    st.subheader('Revenue Trend by Tenure Group')
    df['Tenure_Group'] = pd.cut(
        df['tenure'], bins=[0, 6, 12, 24, 48, df['tenure'].max() + 1], labels=['0-6', '7-12', '13-24', '25-48', '49+'], include_lowest=True
    )
    revenue_trend = df.groupby(
    'Tenure_Group', observed=False
    )['Customer_Lifetime_Value'].sum().reset_index()
    fig_revenue = px.line(revenue_trend, x='Tenure_Group', y='Customer_Lifetime_Value', markers=True, title='Total Revenue by Tenure Group')
    st.plotly_chart(fig_revenue, width="stretch")

    st.markdown('---')
    st.subheader('Executive Dashboard Notes')
    st.write(
        'This page highlights core business KPIs and revenue patterns. Churn rate, contract mix, and customer lifetime value are key signals for retention strategy.'
    )
