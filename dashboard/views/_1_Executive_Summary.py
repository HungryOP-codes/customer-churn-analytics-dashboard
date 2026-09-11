import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def app(df: pd.DataFrame, segmented_df: pd.DataFrame, model_package, model_metrics: dict):
    st.markdown("## 📊 Executive Summary & Business KPIs")
    st.markdown(
        "<p style='color: #6b7280; font-size: 0.95rem;'>High-level performance indicators, revenue health, and customer retention metrics across the organization.</p>",
        unsafe_allow_html=True,
    )

    df = df.copy()
    if 'Customer_Lifetime_Value' not in df.columns:
        df['Customer_Lifetime_Value'] = df['MonthlyCharges'] * df['tenure']
    df['Active'] = df['Churn'].apply(lambda x: 0 if x == 1 else 1)

    total_customers = len(df)
    churned_customers = int(df['Churn'].sum())
    active_customers = int(df['Active'].sum())
    churn_rate = churned_customers / total_customers if total_customers else 0
    total_revenue = float(df['Customer_Lifetime_Value'].sum())
    average_monthly = float(df['MonthlyCharges'].mean())
    average_tenure = float(df['tenure'].mean())
    average_ltv = float(df['Customer_Lifetime_Value'].mean())

    # Row 1: Retention & Volume KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Customers", f"{total_customers:,}", delta="Active Base")
    with c2:
        st.metric("Active Accounts", f"{active_customers:,}", delta=f"{(1 - churn_rate):.1%} Retention")
    with c3:
        st.metric("Churned Accounts", f"{churned_customers:,}", delta=f"-{churn_rate:.1%}", delta_color="inverse")
    with c4:
        st.metric("Overall Churn Rate", f"{churn_rate:.1%}", delta="Annualized Benchmark", delta_color="inverse")

    st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

    # Row 2: Financial & Tenure KPIs
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.metric("Total Realized Revenue", f"${total_revenue:,.0f}")
    with f2:
        st.metric("Avg Monthly Charges", f"${average_monthly:,.2f}")
    with f3:
        st.metric("Avg Tenure", f"{average_tenure:.1f} mos")
    with f4:
        st.metric("Avg Customer Lifetime Value", f"${average_ltv:,.0f}")

    st.markdown("<div style='margin: 1.5rem 0 0.5rem 0;'></div>", unsafe_allow_html=True)

    # Row 3: Charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("<div class='section-header'>Contract Portfolio Distribution</div>", unsafe_allow_html=True)
        distribution = df.groupby('Contract').size().reset_index(name='Count')
        fig_donut = px.pie(
            distribution,
            names='Contract',
            values='Count',
            hole=0.55,
            color='Contract',
            color_discrete_map={
                'Month-to-month': '#ef4444',
                'One year': '#3b82f6',
                'Two year': '#10b981',
            },
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#ffffff', width=2)))
        fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, showlegend=False)
        st.plotly_chart(fig_donut, width="stretch")

    with col_right:
        st.markdown("<div class='section-header'>Churn Rate by Contract Type</div>", unsafe_allow_html=True)
        churn_trend = df.groupby('Contract')['Churn'].mean().reset_index()
        fig_churn = px.bar(
            churn_trend,
            x='Contract',
            y='Churn',
            color='Contract',
            color_discrete_map={
                'Month-to-month': '#ef4444',
                'One year': '#f59e0b',
                'Two year': '#10b981',
            },
            text=churn_trend['Churn'].apply(lambda x: f"{x:.1%}"),
        )
        fig_churn.update_yaxes(tickformat='.0%', title='Churn Rate')
        fig_churn.update_xaxes(title=None)
        fig_churn.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, showlegend=False)
        st.plotly_chart(fig_churn, width="stretch")

    # Row 4: Revenue & Retention Trajectory
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.markdown("<div class='section-header'>Cumulative Revenue by Tenure Cohort</div>", unsafe_allow_html=True)
        df['Tenure_Group'] = pd.cut(
            df['tenure'],
            bins=[0, 6, 12, 24, 48, df['tenure'].max() + 1],
            labels=['0-6 mos', '7-12 mos', '13-24 mos', '25-48 mos', '49+ mos'],
            include_lowest=True,
        )
        revenue_trend = df.groupby('Tenure_Group', observed=False)['Customer_Lifetime_Value'].sum().reset_index()
        fig_rev = px.area(
            revenue_trend,
            x='Tenure_Group',
            y='Customer_Lifetime_Value',
            markers=True,
            color_discrete_sequence=['#4f46e5'],
            labels={'Customer_Lifetime_Value': 'Total Revenue ($)', 'Tenure_Group': 'Tenure Cohort'},
        )
        fig_rev.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_rev, width="stretch")

    with r_col2:
        st.markdown("<div class='section-header'>Monthly Charges: Churned vs. Retained</div>", unsafe_allow_html=True)
        df['Churn_Label'] = df['Churn'].map({0: 'Retained', 1: 'Churned'})
        fig_hist = px.histogram(
            df,
            x='MonthlyCharges',
            color='Churn_Label',
            barmode='overlay',
            opacity=0.7,
            nbins=35,
            color_discrete_map={'Retained': '#10b981', 'Churned': '#ef4444'},
            labels={'MonthlyCharges': 'Monthly Charges ($)', 'Churn_Label': 'Status'},
        )
        fig_hist.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_hist, width="stretch")

    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    m2m_churn = df[df['Contract'] == 'Month-to-month']['Churn'].mean() if len(df[df['Contract'] == 'Month-to-month']) > 0 else 0
    mature_retention = 1 - (df[df['tenure'] > 24]['Churn'].mean() if len(df[df['tenure'] > 24]) > 0 else 0)

    with st.expander("💡 Executive Retention Insights & Strategy", expanded=True):
        st.markdown(
            f"""
            * **Month-to-month contracts** present the single largest churn vulnerability (**{m2m_churn:.1%}** churn rate). Prioritize promotional 1-year contract migration incentives.
            * **High Monthly Charges (>$70/mo)** combined with short tenure (<6 months) exhibits the highest attrition density. Introduce a 90-day onboarding engagement sequence.
            * **Tenure stability threshold**: Customers retained beyond **24 months** demonstrate an **{mature_retention:.1%}** lifetime retention rate.
            """
        )
