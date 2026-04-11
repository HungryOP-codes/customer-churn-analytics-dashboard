import plotly.express as px
import streamlit as st


def app(df, segmented_df, model_package, model_metrics):
    st.title('Customer Segmentation')
    segmented = segmented_df.copy()
    segmented['Risk_Group'] = segmented['Risk_Group'].astype(str)

    st.subheader('Risk Group Distribution')
    risk_counts = segmented['Risk_Group'].value_counts().reset_index()
    risk_counts.columns = ['Risk Group', 'Count']
    fig_risk = px.bar(risk_counts, x='Risk Group', y='Count', title='Risk Group Distribution', color='Risk Group')
    st.plotly_chart(fig_risk, width="stretch")

    st.subheader('Cluster Visualization')
    fig_cluster = px.scatter(
        segmented,
        x='tenure',
        y='MonthlyCharges',
        color='Risk_Group',
        symbol='Cluster',
        title='Customer Clusters by Tenure and Monthly Charges',
        labels={'tenure': 'Tenure (months)', 'MonthlyCharges': 'Monthly Charges'},
    )
    st.plotly_chart(fig_cluster, width="stretch")

    st.subheader('Customer Count by Segment')
    segment_counts = segmented.groupby(['Cluster', 'Risk_Group']).size().reset_index(name='Count')
    fig_segment = px.bar(segment_counts, x='Cluster', y='Count', color='Risk_Group', title='Customer Count by Segment')
    st.plotly_chart(fig_segment, width="stretch")

    st.markdown('---')
    st.write('Customer segmentation helps identify groups with similar churn risk and revenue potential.')
