import pandas as pd
import streamlit as st


def app(df, segmented_df, model_package, model_metrics: dict):
    st.markdown("## 🔍 Interactive Data Explorer & Warehouse")
    st.markdown(
        "<p style='color: #6b7280; font-size: 0.95rem;'>Search, filter, inspect customer records, and export sanitized datasets for offline modeling and business reporting.</p>",
        unsafe_allow_html=True,
    )

    display_df = df.copy()

    # Filters and Search
    with st.expander("⚡ Search & Filter Data Records", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            search_query = st.text_input("🔎 Search by Customer ID", placeholder="e.g. 7590-VHVEG")
        with f2:
            status_filter = st.selectbox("Churn Status", ["All Records", "Retained Accounts Only (0)", "Churned Accounts Only (1)"])
        with f3:
            contract_filter = st.selectbox("Contract Type", ["All Contracts"] + sorted(display_df['Contract'].dropna().unique().tolist()))

    if search_query:
        if 'customerID' in display_df.columns:
            display_df = display_df[display_df['customerID'].astype(str).str.contains(search_query, case=False, na=False)]

    if status_filter == "Retained Accounts Only (0)":
        display_df = display_df[display_df['Churn'] == 0]
    elif status_filter == "Churned Accounts Only (1)":
        display_df = display_df[display_df['Churn'] == 1]

    if contract_filter != "All Contracts":
        display_df = display_df[display_df['Contract'] == contract_filter]

    # Metrics Bar
    total_matches = len(display_df)
    churned_matches = int(display_df['Churn'].sum()) if total_matches > 0 else 0
    pct_churn = churned_matches / total_matches if total_matches > 0 else 0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Records Matching Filters", f"{total_matches:,}", delta=f"{total_matches / len(df):.1%} of Dataset")
    with m2:
        st.metric("Churn Rate in Results", f"{pct_churn:.1%}")
    with m3:
        avg_charge = display_df['MonthlyCharges'].mean() if total_matches > 0 else 0
        st.metric("Avg Monthly Charges in Results", f"${avg_charge:,.2f}")

    st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

    # Render Table
    st.dataframe(
        display_df,
        column_config={
            "customerID": st.column_config.TextColumn("Customer ID"),
            "tenure": st.column_config.ProgressColumn("Tenure (Months)", min_value=0, max_value=72, format="%d mos"),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly Charges", format="$%.2f"),
            "TotalCharges": st.column_config.NumberColumn("Total Charges", format="$%.2f"),
            "Churn": st.column_config.CheckboxColumn("Churned?"),
        },
        width="stretch",
        height=450,
    )

    # Download Button
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Filtered Dataset to CSV",
        data=csv,
        file_name='filtered_customer_data.csv',
        mime='text/csv',
        width="stretch",
    )
