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
            df_hourly = pd.read_csv(z.open('hourly_usage.csv'))
            df_costs = pd.read_csv(z.open('infrastructure_costs.csv'))
            df_research = pd.read_csv(z.open('research_requests.csv'))
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
    "Product Analysis", 
    "Infrastructure & Cost Analysis", 
])

# st.sidebar.markdown("---")
# st.sidebar.subheader("Dashboard Info")
# st.sidebar.info("Tavily Data Analyst Home Assignment by Dan Benbenisti")

# ==========================================
# 4. Main Dashboard Area
# ==========================================

if page == "Overview":
    st.title("📊 Executive Overview")
    st.markdown("High-level metrics across usage, costs, and user growth.")
    
    # --- Important Stats (Metrics) ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Assuming 'Cost' column exists
        total_cost = df_costs['Cost'].sum() if 'Cost' in df_costs.columns else 0
        st.metric(label="Total Infrastructure Cost", value=f"${total_cost:,.2f}")
        
    with col2:
        # Assuming 'API_Calls' column exists
        total_calls = df_hourly['API_Calls'].sum() if 'API_Calls' in df_hourly.columns else len(df_hourly)
        st.metric(label="Total API Calls", value=f"{total_calls:,}")
        
    with col3:
        total_users = len(df_users)
        st.metric(label="Total Users", value=f"{total_users:,}")

    with col4:
        total_requests = len(df_research)
        st.metric(label="Total Research Requests", value=f"{total_requests:,}")

    st.markdown("---")
    
    # --- Figures ---
    st.subheader("Infrastructure Cost Over Time")
    if 'Date' in df_costs.columns and 'Cost' in df_costs.columns:
        costs_daily = df_costs.groupby('Date')['Cost'].sum().reset_index()
        p_overview = figure(title="Daily Infrastructure Cost", x_axis_type='datetime', height=400)
        p_overview.line(costs_daily['Date'], costs_daily['Cost'], line_width=2, color="navy")
        p_overview.varea(x=costs_daily['Date'], y1=0, y2=costs_daily['Cost'], color="lightblue", alpha=0.5)
        st.bokeh_chart(p_overview, use_container_width=True)
    else:
        st.warning("Cost or Date columns not found. Please adjust column names in the script.")


elif page == "Hourly Usage Analytics":
    st.title("📈 Hourly Usage Analytics")
    st.write("Detailed breakdown of API usage over time.")
    
    if 'Timestamp' in df_hourly.columns and 'API_Calls' in df_hourly.columns:
        # Aggregate by day for a clearer overall chart, or keep hourly if preferred
        df_hourly['Date_Only'] = df_hourly['Timestamp'].dt.date
        daily_usage = df_hourly.groupby('Date_Only')['API_Calls'].sum().reset_index()
        daily_usage['Date_Only'] = pd.to_datetime(daily_usage['Date_Only'])

        p_usage = figure(title="Daily API Calls Volume", x_axis_type='datetime', height=400)
        p_usage.vbar(x='Date_Only', top='API_Calls', width=86400000 * 0.8, source=ColumnDataSource(daily_usage), color="green", alpha=0.7)
        st.bokeh_chart(p_usage, use_container_width=True)
        
        with st.expander("View Raw Hourly Data"):
            st.dataframe(df_hourly.head(100))
    else:
        st.warning("Missing required columns ('Timestamp' or 'API_Calls') in hourly_usage.csv")


elif page == "Infrastructure Costs":
    st.title("💸 Infrastructure Costs")
    st.write("Breakdown of costs by service and time.")
    
    if 'Service' in df_costs.columns and 'Cost' in df_costs.columns:
        service_costs = df_costs.groupby('Service')['Cost'].sum().reset_index()
        services = service_costs['Service'].tolist()
        
        p_costs = figure(x_range=services, title="Total Cost by Service", height=400, toolbar_location=None)
        cmap = factor_cmap('Service', palette=bp.Category10[max(3, len(services))], factors=services)
        p_costs.vbar(x='Service', top='Cost', width=0.8, source=ColumnDataSource(service_costs), color=cmap, alpha=0.8)
        p_costs.xgrid.grid_line_color = None
        st.bokeh_chart(p_costs, use_container_width=True)
    else:
        st.warning("Missing required columns ('Service' or 'Cost') in infrastructure_costs.csv")


elif page == "Research Requests":
    st.title("🔎 Research Requests")
    st.write("Analysis of research request topics and processing times.")
    
    if 'Topic' in df_research.columns:
        topic_counts = df_research['Topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        topics = topic_counts['Topic'].tolist()
        
        p_research = figure(y_range=topics[::-1], title="Volume of Requests by Topic", height=400)
        p_research.hbar(y='Topic', right='Count', height=0.8, source=ColumnDataSource(topic_counts), color="purple", alpha=0.6)
        st.bokeh_chart(p_research, use_container_width=True)
        
        if 'Processing_Time_sec' in df_research.columns:
            avg_time = df_research['Processing_Time_sec'].mean()
            st.metric("Average Processing Time", f"{avg_time:.2f} seconds")
    else:
        st.warning("Missing 'Topic' column in research_requests.csv")


elif page == "User Metrics":
    st.title("👥 User Metrics")
    st.write("Overview of user base and subscription tiers.")
    
    if 'Subscription_Tier' in df_users.columns:
        tier_counts = df_users['Subscription_Tier'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Users']
        tiers = tier_counts['Tier'].astype(str).tolist()
        tier_counts['Tier'] = tier_counts['Tier'].astype(str)

        p_users = figure(x_range=tiers, title="Users by Subscription Tier", height=400)
        p_users.vbar(x='Tier', top='Users', width=0.5, source=ColumnDataSource(tier_counts), color="orange", alpha=0.8)
        st.bokeh_chart(p_users, use_container_width=True)
    else:
        st.warning("Missing 'Subscription_Tier' column in users.csv")

    with st.expander("View User Directory"):
        st.dataframe(df_users.head(100))
