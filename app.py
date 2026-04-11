import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
import bokeh.palettes as bp
import zipfile

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Data Loading (Backend Datasets)
# ==========================================
# We use @st.cache_data so the data isn't reloaded every time the user interacts with the app.
@st.cache_data
def load_data():
    """
    Load the 4 main datasets from a single ZIP file in the repository.
    Make sure 'data.zip' is in the same folder as app.py, and contains
    'hourly_usage.csv', 'infrastructure_costs.csv', 'research_requests.csv', and 'users.csv'.
    """
    try:
        # Open the single zip file
        with zipfile.ZipFile('data.zip', 'r') as z:
            # Dataset 1: Hourly Usage Data
            df_hourly = pd.read_csv(z.open('hourly_usage.csv'))
            if 'hour' in df_hourly.columns:
                df_hourly['hour'] = pd.to_datetime(df_hourly['hour'])
            
            # Dataset 2: Infrastructure Costs Data
            df_costs = pd.read_csv(z.open('infrastructure_costs.csv'))
            if 'hour' in df_costs.columns:
                df_costs['hour'] = pd.to_datetime(df_costs['hour'])
            
            # Dataset 3: Research Requests Data
            df_research = pd.read_csv(z.open('research_requests.csv'))
            if 'timestamp' in df_research.columns:
                df_research['timestamp'] = pd.to_datetime(df_research['timestamp'])

            # Dataset 4: Users Data
            df_users = pd.read_csv(z.open('users.csv'))
            
        return df_hourly, df_costs, df_research, df_users
        
    except FileNotFoundError:
        st.error("Missing 'data.zip' file. Please ensure it is uploaded to your repository.")
        st.stop()
    except KeyError as e:
        st.error(f"Missing file inside the zip: {e}. Ensure all CSVs are named exactly as requested inside 'data.zip'.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading the data: {e}. Please check your CSV column names and formats.")
        st.stop()

# Load the datasets
df_hourly, df_costs, df_research, df_users = load_data()

# ==========================================
# 3. Sidebar Navigation & Filters
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview", 
    "Part 1: Product Analysis", 
    "Part 2: Infrastructure & Cost"
])

st.sidebar.markdown("---")
st.sidebar.subheader("Dashboard Info")
st.sidebar.info("Tavily Data Analyst Home Assignment by Dan Benbenisti")

# ==========================================
# 4. Main Dashboard Area
# ==========================================

if page == "Overview":
    st.title("📊 Executive Overview")
    st.markdown("High-level metrics across usage, costs, and user growth.")
    
    # --- Important Stats (Metrics) ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Summing all cost columns if they exist based on the PDF schema
        cost_cols = [c for c in df_costs.columns if c.startswith('infra_') or c.startswith('model_')]
        total_cost = df_costs[cost_cols].sum().sum() if cost_cols else 0
        st.metric(label="Total Infrastructure Cost", value=f"${total_cost:,.2f}")
        
    with col2:
        total_calls = df_hourly['request_count'].sum() if 'request_count' in df_hourly.columns else len(df_hourly)
        st.metric(label="Total API Calls", value=f"{total_calls:,}")
        
    with col3:
        total_users = len(df_users)
        st.metric(label="Total Users", value=f"{total_users:,}")

    with col4:
        total_requests = len(df_research)
        st.metric(label="Total Research Requests", value=f"{total_requests:,}")

    st.markdown("---")
    st.info("👈 Select **Part 1** or **Part 2** from the sidebar to dive into the assignment requirements.")

elif page == "Part 1: Product Analysis":
    st.title("📦 Part 1: Product Analysis")
    st.write("Analysis of the Research API health, growth, and future direction.")
    
    st.markdown("### 🎯 Critical Business Questions")
    st.markdown("""
    1. **[Question 1 Placeholder]** - Justification...
    2. **[Question 2 Placeholder]** - Justification...
    3. **[Question 3 Placeholder]** - Justification...
    """)
    
    st.markdown("---")
    
    # Using tabs to organize the different datasets for Part 1 cleanly
    tab1, tab2, tab3 = st.tabs(["Research Requests", "Hourly Usage", "User Metrics"])
    
    with tab1:
        st.subheader("Research Requests Analysis")
        if 'status' in df_research.columns:
            status_counts = df_research['status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            statuses = status_counts['Status'].tolist()
            
            p_research = figure(x_range=statuses, title="Request Outcomes", height=400)
            p_research.vbar(x='Status', top='Count', width=0.5, source=ColumnDataSource(status_counts), color="purple", alpha=0.7)
            st.bokeh_chart(p_research, use_container_width=True)
        else:
            st.warning("Missing 'status' column in research_requests.csv")

    with tab2:
        st.subheader("Hourly Usage Patterns")
        if 'hour' in df_hourly.columns and 'request_count' in df_hourly.columns:
            df_hourly['Date_Only'] = df_hourly['hour'].dt.date
            daily_usage = df_hourly.groupby('Date_Only')['request_count'].sum().reset_index()
            daily_usage['Date_Only'] = pd.to_datetime(daily_usage['Date_Only'])

            p_usage = figure(title="Daily Request Volume", x_axis_type='datetime', height=400)
            p_usage.vbar(x='Date_Only', top='request_count', width=86400000 * 0.8, source=ColumnDataSource(daily_usage), color="green", alpha=0.7)
            st.bokeh_chart(p_usage, use_container_width=True)
        else:
            st.warning("Missing 'hour' or 'request_count' in hourly_usage.csv")

    with tab3:
        st.subheader("User Plan Distribution")
        if 'plan' in df_users.columns:
            plan_counts = df_users['plan'].value_counts().reset_index()
            plan_counts.columns = ['Plan', 'Users']
            plans = plan_counts['Plan'].astype(str).tolist()

            p_users = figure(x_range=plans, title="Users by Subscription Plan", height=400)
            p_users.vbar(x='Plan', top='Users', width=0.5, source=ColumnDataSource(plan_counts), color="orange", alpha=0.8)
            st.bokeh_chart(p_users, use_container_width=True)
        else:
            st.warning("Missing 'plan' column in users.csv")

elif page == "Part 2: Infrastructure & Cost":
    st.title("💸 Part 2: Infrastructure & Cost Analysis")
    st.write("Understanding how infrastructure costs behave and what drives them.")
    
    st.markdown("### 🔑 Key Cost Drivers Analysis")
    st.markdown("**Hypothesis:** [Write your cost hypothesis here]")
    
    # Calculate costs based on the specific columns from the PDF
    cost_cols = [c for c in df_costs.columns if c.startswith('infra_') or c.startswith('model_')]
    
    if cost_cols and 'hour' in df_costs.columns:
        # 1. Total Cost Over Time
        df_costs['Total_Cost'] = df_costs[cost_cols].sum(axis=1)
        costs_daily = df_costs.groupby(df_costs['hour'].dt.date)['Total_Cost'].sum().reset_index()
        costs_daily['hour'] = pd.to_datetime(costs_daily['hour'])
        
        p_overview = figure(title="Daily Total Infrastructure Cost", x_axis_type='datetime', height=400)
        p_overview.line(costs_daily['hour'], costs_daily['Total_Cost'], line_width=2, color="navy")
        p_overview.varea(x=costs_daily['hour'], y1=0, y2=costs_daily['Total_Cost'], color="lightblue", alpha=0.5)
        st.bokeh_chart(p_overview, use_container_width=True)
        
        # 2. Cost Breakdown by Category
        st.subheader("Cost Breakdown by Component")
        sum_costs = df_costs[cost_cols].sum().reset_index()
        sum_costs.columns = ['Component', 'Cost']
        sum_costs = sum_costs.sort_values('Cost', ascending=False).head(10) # Top 10 costs
        
        components = sum_costs['Component'].tolist()
        p_breakdown = figure(y_range=components[::-1], title="Top 10 Cost Drivers", height=400)
        p_breakdown.hbar(y='Component', right='Cost', height=0.6, source=ColumnDataSource(sum_costs), color="red", alpha=0.6)
        st.bokeh_chart(p_breakdown, use_container_width=True)
        
    else:
        st.warning("Cost components (infra_ / model_) or 'hour' column not found in infrastructure_costs.csv")
