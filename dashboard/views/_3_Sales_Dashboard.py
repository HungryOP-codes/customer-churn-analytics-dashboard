import pandas as pd
import plotly.express as px
import streamlit as st


def app(df: pd.DataFrame, segmented_df: pd.DataFrame, model_package, model_metrics: dict):
    st.title('Sales Dashboard')
    df = df.copy()
    df['Customer_Lifetime_Value'] = df['MonthlyCharges'] * df['tenure']
    df['Revenue'] = df['Customer_Lifetime_Value']

    st.subheader('Revenue by Contract')
    revenue_contract = df.groupby('Contract')['Revenue'].sum().reset_index().sort_values(by='Revenue', ascending=False)
    fig_contract = px.bar(revenue_contract, x='Contract', y='Revenue', title='Total Revenue by Contract')
    st.plotly_chart(fig_contract,width="stretch")

    st.subheader('Revenue by Payment Method')
    revenue_payment = df.groupby('PaymentMethod')['Revenue'].sum().reset_index().sort_values(by='Revenue', ascending=False)
    fig_payment = px.bar(revenue_payment, x='PaymentMethod', y='Revenue', title='Total Revenue by Payment Method')
    st.plotly_chart(fig_payment, width="stretch")

    st.subheader('Top Revenue Customers')
    top_customers = df.nlargest(10, 'Customer_Lifetime_Value')[['tenure', 'MonthlyCharges', 'Customer_Lifetime_Value', 'Contract', 'PaymentMethod']]
    st.dataframe(top_customers.reset_index(drop=True))

    st.subheader('Revenue Trend by Tenure Group')
    tenure_bins = [0, 6, 12, 24, 48, df['tenure'].max() + 1]
    tenure_labels = ['0-6', '7-12', '13-24', '25-48', '49+']
    df['Tenure_Group'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels, include_lowest=True)
    revenue_trend = df.groupby('Tenure_Group')['Revenue'].sum().reset_index()
    fig_trend = px.line(revenue_trend, x='Tenure_Group', y='Revenue', markers=True, title='Revenue Trend by Tenure Group')
    st.plotly_chart(fig_trend, width="stretch")

    st.markdown('---')
    st.write('Sales performance is driven by contract type, payment method, and customer tenure. Use the top revenue customers list to identify high-value retention opportunities.')
