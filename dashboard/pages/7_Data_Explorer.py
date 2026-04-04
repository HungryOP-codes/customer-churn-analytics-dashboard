import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder


def app(df, segmented_df, model_package, model_metrics: dict):
    st.title('Data Explorer')
    st.write('Search, sort, filter, and download cleaned customer data.')

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(filter=True, sortable=True, resizable=True)
    gb.configure_selection('multiple', use_checkbox=True)
    grid_options = gb.build()

    AgGrid(
        df,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        height=500,
    )

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Download CSV', csv, 'cleaned_customer_data.csv', 'text/csv')

    st.markdown('---')
    st.write('Use this explorer to validate data quality, review feature values, and export the processed dataset for further analysis.')
