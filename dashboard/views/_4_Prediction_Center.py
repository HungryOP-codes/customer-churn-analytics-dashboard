import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.predict import predict_churn, predict_batch


def app(df, segmented_df, model_package, model_metrics):
    st.markdown("## ⚡ Live Churn Prediction & Retention Center")
    st.markdown(
        "<p style='color: #6b7280; font-size: 0.95rem;'>Simulate real-time churn risk for individual customer profiles or perform bulk batch scoring for retention campaigns.</p>",
        unsafe_allow_html=True,
    )

    tab_single, tab_batch = st.tabs(["🎯 Single Account Simulator", "📂 Bulk Batch Scoring"])

    with tab_single:
        with st.form('prediction_form'):
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox('Gender', ['Female', 'Male'])
                senior_citizen = st.selectbox('Senior Citizen Status', [0, 1], format_func=lambda x: 'Yes (1)' if x == 1 else 'No (0)')
            with c2:
                tenure = st.slider('Customer Tenure (Months)', min_value=0, max_value=72, value=12, help="Number of months customer has stayed with company")
                monthly_charges = st.number_input('Monthly Charges ($)', min_value=15.0, max_value=250.0, value=75.0, step=2.5)
            with c3:
                contract = st.selectbox('Contract Term', ['Month-to-month', 'One year', 'Two year'])
                payment_method = st.selectbox(
                    'Payment Method',
                    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                )

            submitted = st.form_submit_button('⚡ Calculate Churn Probability', width="stretch")

        if submitted:
            customer_info = {
                'Gender': gender,
                'SeniorCitizen': senior_citizen,
                'Tenure': tenure,
                'MonthlyCharges': monthly_charges,
                'Contract': contract,
                'PaymentMethod': payment_method,
            }

            with st.spinner('Evaluating ML model decision boundaries...'):
                result = predict_churn(customer_info)

            prob = result['probability']
            pred = result['prediction']
            risk = result['risk_level']
            model_name = result.get('model_name', 'Trained Classifier')

            st.markdown("---")
            st.markdown(f"#### 🔍 Prediction Results *(Engine: {model_name})*")

            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                # Plotly Gauge Chart for Churn Probability
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number={'suffix': "%", 'font': {'size': 32}},
                        title={'text': "Calculated Churn Probability", 'font': {'size': 18}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#ef4444" if prob >= 0.5 else "#10b981"},
                            'steps': [
                                {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.15)"},
                                {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.15)"},
                                {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.15)"},
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(margin=dict(t=30, b=10, l=30, r=30), height=260)
                st.plotly_chart(fig_gauge, width="stretch")

            with res_col2:
                badge_class = "badge-high" if risk == "High Risk" else ("badge-med" if risk == "Medium Risk" else "badge-low")
                st.markdown(
                    f"""
                    <div class="metric-card" style="margin-top: 1rem;">
                        <p class="metric-title">Model Verdict</p>
                        <p class="metric-value" style="color: {'#ef4444' if pred == 'Yes' else '#10b981'};">
                            {'⚠️ Likely to Churn' if pred == 'Yes' else '✅ Retained Customer'}
                        </p>
                        <p style="margin-top: 0.5rem;">
                            Risk Classification: <span class="badge-pill {badge_class}">{risk}</span>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("##### 🎯 Recommended Retention Playbook")
                recommendations = result.get('recommendations', [])
                for rec in recommendations:
                    st.markdown(f"- 💡 **Action**: {rec}")

    with tab_batch:
        st.markdown("#### 📂 Score Customer Batches via CSV")
        st.markdown("Upload any CSV with customer fields (`tenure`, `MonthlyCharges`, `Contract`, `PaymentMethod`, etc.) or generate a demo batch from our existing database.")

        demo_col, upload_col = st.columns([1, 2])
        with demo_col:
            score_demo = st.button("🚀 Score 50 Sample Accounts", width="stretch")

        uploaded_file = st.file_uploader("Or upload your CSV file", type=['csv'])

        batch_df = None
        if score_demo:
            batch_df = df.sample(min(50, len(df)), random_state=42)
        elif uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)

        if batch_df is not None:
            with st.spinner("Scoring batch records..."):
                scored_df = predict_batch(batch_df)

            churn_count = (scored_df['Churn_Prediction'] == 'Yes').sum()
            avg_prob = scored_df['Churn_Probability'].mean()

            b1, b2, b3 = st.columns(3)
            with b1:
                st.metric("Total Accounts Scored", len(scored_df))
            with b2:
                st.metric("Accounts Flagged for Churn", f"{churn_count}", delta=f"{churn_count / len(scored_df):.1%}")
            with b3:
                st.metric("Average Churn Risk", f"{avg_prob:.1%}")

            st.dataframe(
                scored_df[['customerID', 'Contract', 'PaymentMethod', 'MonthlyCharges', 'tenure', 'Churn_Probability', 'Churn_Prediction', 'Risk_Category']]
                if 'customerID' in scored_df.columns
                else scored_df,
                width="stretch",
            )

            csv_data = scored_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Scored Customer List (CSV)",
                data=csv_data,
                file_name="churn_scored_customers.csv",
                mime="text/csv",
                width="stretch",
            )
