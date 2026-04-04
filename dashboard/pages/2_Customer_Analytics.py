import pandas as pd
import plotly.express as px
import streamlit as st


def app(df: pd.DataFrame, segmented_df: pd.DataFrame, model_package, model_metrics: dict):
    st.title('Customer Analytics')
    df = df.copy()
    df['Gender'] = df['gender'].map({0: 'Female', 1: 'Male'})
    df['Customer_Lifetime_Value'] = df['MonthlyCharges'] * df['tenure']

    st.sidebar.header('Filters')
    contract_options = ['All'] + sorted(df['Contract'].unique().tolist())
    payment_options = ['All'] + sorted(df['PaymentMethod'].unique().tolist())
    selected_contract = st.sidebar.selectbox('Contract', contract_options, index=0)
    selected_payment = st.sidebar.selectbox('Payment Method', payment_options, index=0)

    filtered = df.copy()
    if selected_contract != 'All':
        filtered = filtered[filtered['Contract'] == selected_contract]
    if selected_payment != 'All':
        filtered = filtered[filtered['PaymentMethod'] == selected_payment]

    churn_gender = filtered.groupby('Gender')['Churn'].mean().reset_index()
    churn_contract = filtered.groupby('Contract')['Churn'].mean().reset_index()
    churn_payment = filtered.groupby('PaymentMethod')['Churn'].mean().reset_index()

    fig_gender = px.bar(churn_gender, x='Gender', y='Churn', title='Churn Rate by Gender', labels={'Churn': 'Churn Rate'})
    fig_gender.update_yaxes(tickformat='.0%')
    fig_contract = px.bar(churn_contract, x='Contract', y='Churn', title='Churn Rate by Contract', labels={'Churn': 'Churn Rate'})
    fig_contract.update_yaxes(tickformat='.0%')
    fig_payment = px.bar(churn_payment, x='PaymentMethod', y='Churn', title='Churn Rate by Payment Method', labels={'Churn': 'Churn Rate'})
    fig_payment.update_yaxes(tickformat='.0%')

    st.plotly_chart(fig_gender, width="stretch")
    st.plotly_chart(fig_contract, width="stretch")
    st.plotly_chart(fig_payment, width="stretch")

    fig_hist = px.histogram(filtered, x='MonthlyCharges', nbins=30, title='Monthly Charges Distribution')
    st.plotly_chart(fig_hist, width="stretch")

    fig_scatter = px.scatter(
        filtered,
        x='tenure',
        y='MonthlyCharges',
        color=filtered['Churn'].map({0: 'Retained', 1: 'Churned'}),
        title='Tenure vs Monthly Charges by Churn',
        labels={'tenure': 'Tenure (months)', 'MonthlyCharges': 'Monthly Charges'},
    )
    st.plotly_chart(fig_scatter, width="stretch")

    st.markdown('---')
    st.write('Use the sidebar filters to compare churn patterns across contract types and payment methods. The scatter plot helps identify high-value customers with churn risk.')
