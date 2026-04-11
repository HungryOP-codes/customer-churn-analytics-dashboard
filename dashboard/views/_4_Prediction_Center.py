import streamlit as st
from src.predict import predict_churn


def app(df, segmented_df, model_package, model_metrics):
    st.title('Prediction Center')
    st.write('Use the form below to simulate a customer churn prediction.')

    with st.form('prediction_form'):
        gender = st.selectbox('Gender', ['Female', 'Male'])
        senior_citizen = st.selectbox('Senior Citizen', [0, 1])
        tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=500.0, value=70.0)
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        payment_method = st.selectbox(
            'Payment Method',
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        )
        predict_button = st.form_submit_button('Predict Churn')

    if predict_button:
        customer_info = {
            'Gender': gender,
            'SeniorCitizen': senior_citizen,
            'Tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'Contract': contract,
            'PaymentMethod': payment_method,
        }
        with st.spinner('Predicting churn...'):
            result = predict_churn(customer_info)

        st.success('Prediction complete')
        st.metric('Churn Prediction', result['prediction'])
        st.metric('Churn Probability', f"{result['probability']:.2%}")
        st.metric('Risk Level', result['risk_level'])

        st.write('### Input Summary')
        st.json(customer_info)
