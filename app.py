"""
Tavily Data Analyst assignment — Streamlit dashboard.
Loads CSVs from data.zip (deploy) or from the same folder / parent folder (local dev).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tavily Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent


@st.cache_data
def load_data():
    """Load four assignment tables from zip or loose CSVs."""
    names = (
        "hourly_usage.csv",
        "infrastructure_costs.csv",
        "research_requests.csv",
        "users.csv",
    )

    def read_zip() -> dict[str, pd.DataFrame]:
        zpath = BASE_DIR / "data.zip"
        if not zpath.is_file():
            raise FileNotFoundError("data.zip")
        out = {}
        with zipfile.ZipFile(zpath, "r") as z:
            for n in names:
                with z.open(n) as f:
                    out[n] = pd.read_csv(f)
        return out

    def read_loose(root: Path) -> dict[str, pd.DataFrame]:
        out = {}
        for n in names:
            p = root / n
            if not p.is_file():
                raise FileNotFoundError(str(p))
            out[n] = pd.read_csv(p)
        return out

    try:
        frames = read_zip()
    except (FileNotFoundError, KeyError):
        try:
            frames = read_loose(BASE_DIR)
        except FileNotFoundError:
            frames = read_loose(PARENT_DIR)

    df_hourly = frames["hourly_usage.csv"].copy()
    df_costs = frames["infrastructure_costs.csv"].copy()
    df_research = frames["research_requests.csv"].copy()
    df_users = frames["users.csv"].copy()

    df_hourly.columns = df_hourly.columns.str.lower()
    df_costs.columns = df_costs.columns.str.lower()
    df_research.columns = df_research.columns.str.lower()
    df_users.columns = df_users.columns.str.lower()

    if "hour" in df_hourly.columns:
        df_hourly["hour"] = pd.to_datetime(df_hourly["hour"], utc=True, errors="coerce")
    if "hour" in df_costs.columns:
        df_costs["hour"] = pd.to_datetime(df_costs["hour"], utc=True, errors="coerce")
    if "timestamp" in df_research.columns:
        df_research["timestamp"] = pd.to_datetime(
            df_research["timestamp"].astype(str).str.replace(" Z", "Z", regex=False),
            utc=True,
            errors="coerce",
        )
    if "created_at" in df_users.columns:
        df_users["created_at"] = pd.to_datetime(df_users["created_at"], errors="coerce")

    for c in ("request_cost", "credits_used", "response_time_seconds"):
        if c in df_research.columns:
            df_research[c] = pd.to_numeric(df_research[c], errors="coerce")
    for c in ("request_count", "total_credits_used", "paygo_credits_used"):
        if c in df_hourly.columns:
            df_hourly[c] = pd.to_numeric(df_hourly[c], errors="coerce")

    for c in ("has_output_schema", "stream"):
        if c in df_research.columns:
            df_research[c] = df_research[c].astype(str).str.upper().eq("TRUE")

    if "has_paygo" in df_users.columns:
        df_users["has_paygo"] = df_users["has_paygo"].astype(str).str.lower().eq("true")

    return df_hourly, df_costs, df_research, df_users


try:
    df_hourly, df_costs, df_research, df_users = load_data()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# One row per user_id (for consistent user counts)
_u = df_users.dropna(subset=["user_id"]).copy()
_u["user_id"] = _u["user_id"].astype(int)
_agg: dict = {}
for _col in ("plan", "plan_limit"):
    if _col in _u.columns:
        _agg[_col] = "first"
if "has_paygo" in _u.columns:
    _agg["has_paygo"] = "max"
if "created_at" in _u.columns:
    _agg["created_at"] = "min"
if _agg:
    df_users_unique = _u.groupby("user_id", as_index=False).agg(_agg)
else:
    df_users_unique = _u.drop_duplicates(subset=["user_id"])

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Product Analysis", "Infrastructure & Cost Analysis"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Tavily Data Analyst home assignment — Dan Benbenisti")

def _fmt_millions(n: float) -> str:
    return f"{n / 1e6:.2f}M"


def _is_paying_user_row(u: pd.DataFrame) -> pd.Series:
    """Paying = Pay-as-you-go and/or non-freemium plan. Adjust FREEMIUM_PLANS as needed."""
    pl = u["plan"].fillna("").astype(str).str.lower().str.strip() if "plan" in u.columns else pd.Series("", index=u.index)
    freemium = pl.eq("researcher") | pl.eq("(unknown)") | pl.eq("unknown")
    paygo = u["has_paygo"].astype(bool) if "has_paygo" in u.columns else pd.Series(False, index=u.index)
    on_paid_plan = ~freemium & pl.ne("")
    return paygo | on_paid_plan


# -----------------------------------------------------------------------------
# Overview — Executive Summary
# -----------------------------------------------------------------------------
if page == "Overview":
    # --- Swap these blocks for dummy data vs. live data ---------------------------------
    total_users = int(df_users_unique["user_id"].nunique())
    paying_mask = _is_paying_user_row(df_users_unique)
    paying_users_count = int(paying_mask.sum())
    free_users_count = int(total_users - paying_users_count)

    if "request_count" in df_hourly.columns:
        total_requests_platform = int(df_hourly["request_count"].sum())
    else:
        total_requests_platform = len(df_hourly)

    if "total_credits_used" in df_hourly.columns:
        total_credits_platform = int(df_hourly["total_credits_used"].sum())
    else:
        total_credits_platform = 0

    if "request_type" in df_hourly.columns and "request_count" in df_hourly.columns:
        endpoint_dist = df_hourly.groupby(df_hourly["request_type"].fillna("(unknown)").astype(str))["request_count"].sum()
    else:
        endpoint_dist = pd.Series(dtype=float)

    # -------------------------------------------------------------------------------------

    st.title("Platform Overview: Executive Summary")
    st.markdown(
        "*A high-level view of user monetization and platform traffic distribution.*"
    )

    paying_pct = (100.0 * paying_users_count / total_users) if total_users else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Users", f"{total_users:,}")
    m2.metric("Total Requests", _fmt_millions(float(total_requests_platform)))
    m3.metric("Total Credits Consumed", _fmt_millions(float(total_credits_platform)))
    m4.metric("Paying Users (%)", f"{paying_pct:.1f}%")

    free_pct_users = (100.0 * free_users_count / total_users) if total_users else 0.0

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            f"### Freemium heavy: **{free_pct_users:.0f}%** of user base is non-monetized"
        )
        donut_df = pd.DataFrame(
            {
                "segment": ["Free users", "Paying users"],
                "users": [free_users_count, paying_users_count],
            }
        )
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=donut_df["segment"],
                    values=donut_df["users"],
                    hole=0.52,
                    marker=dict(colors=["#cbd5e1", "#2563eb"]),
                    textinfo="label+percent",
                    textposition="outside",
                    hovertemplate="<b>%{label}</b><br>Users: %{value:,}<br>Share: %{percent}<extra></extra>",
                )
            ]
        )
        fig_donut.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(t=30, b=80, l=40, r=40),
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown("### Query is king, Research is a niche")
        if endpoint_dist.empty:
            st.info("No endpoint / request data available for chart.")
        else:
            s = endpoint_dist.sort_values(ascending=True)
            bar_df = s.reset_index()
            bar_df.columns = ["endpoint", "requests"]
            colors = [
                "#ea580c" if str(e).lower() == "research" else "#64748b"
                for e in bar_df["endpoint"]
            ]
            fig_bar = go.Figure(
                go.Bar(
                    x=bar_df["requests"],
                    y=bar_df["endpoint"],
                    orientation="h",
                    marker_color=colors,
                    hovertemplate="<b>%{y}</b><br>Requests: %{x:,}<extra></extra>",
                )
            )
            fig_bar.update_layout(
                xaxis_title="Requests (sum of hourly counts)",
                yaxis_title=None,
                height=max(360, 40 + 32 * len(bar_df)),
                margin=dict(t=30, b=40, l=40, r=24),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    research_req = float(endpoint_dist.get("research", 0)) if not endpoint_dist.empty else 0.0
    research_share = (100.0 * research_req / total_requests_platform) if total_requests_platform else 0.0

    st.info(
        f"**Strategic insight:** While the platform handles large volume (mostly **query**), monetization "
        f"relies on about **{paying_pct:.1f}%** paying users. The **research** endpoint is only "
        f"**~{research_share:.2f}%** of total traffic in this sample, yet its high resource use "
        f"warrants careful unit economics — see **Product Analysis**."
    )

    ts_left, ts_right = st.columns(2)

    with ts_left:
        st.markdown("### Platform Traffic (7-Day Moving Average)")
        if "hour" not in df_hourly.columns or "request_count" not in df_hourly.columns:
            st.info("Missing `hour` or `request_count` in hourly usage data.")
        else:
            _hu = df_hourly.dropna(subset=["hour"]).copy()
            _hu["day"] = pd.to_datetime(_hu["hour"], utc=True).dt.normalize()
            _daily = (
                _hu.groupby("day", as_index=False)["request_count"]
                .sum()
                .sort_values("day")
                .reset_index(drop=True)
            )
            # min_periods=1: early dates use 1..6 days of history; from day 7 onward uses full 7-day window
            _daily["requests_ma7"] = _daily["request_count"].rolling(window=7, min_periods=1).mean()
            if _daily.empty:
                st.info("Not enough data for a 7-day moving average.")
            else:
                fig_ma = px.line(
                    _daily,
                    x="day",
                    y="requests_ma7",
                    title=None,
                )
                fig_ma.update_traces(line=dict(color="#2563eb", width=2))
                fig_ma.update_layout(
                    showlegend=False,
                    xaxis_title="Date (UTC)",
                    yaxis_title="Requests (7d MA)",
                    height=380,
                    margin=dict(t=20, b=48, l=48, r=24),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_ma, use_container_width=True)
                st.caption(
                    "Moving average uses `rolling(window=7, min_periods=1)`: the first days average over "
                    "however many daily points exist up to seven."
                )

    with ts_right:
        st.markdown("### Cumulative User Growth")
        if "created_at" not in df_users_unique.columns:
            st.info("Missing `created_at` in users data.")
        else:
            _uu = df_users_unique.dropna(subset=["created_at"]).copy()
            _uu["day"] = pd.to_datetime(_uu["created_at"], utc=True, errors="coerce").dt.normalize()
            _uu = _uu.dropna(subset=["day"])
            if _uu.empty:
                st.info("No valid signup dates for cumulative growth.")
            else:
                _growth = (
                    _uu.groupby("day").size().reset_index(name="new_users").sort_values("day").reset_index(drop=True)
                )
                _growth["cumulative_users"] = _growth["new_users"].cumsum()
                fig_cum = px.area(
                    _growth,
                    x="day",
                    y="cumulative_users",
                    title=None,
                )
                fig_cum.update_traces(
                    line=dict(color="#059669", width=1),
                    fillcolor="rgba(5, 150, 105, 0.25)",
                )
                fig_cum.update_layout(
                    showlegend=False,
                    xaxis_title="Date (UTC)",
                    yaxis_title="Cumulative users",
                    height=380,
                    margin=dict(t=20, b=48, l=48, r=24),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_cum, use_container_width=True)

# -----------------------------------------------------------------------------
# Product Analysis
# -----------------------------------------------------------------------------
elif page == "Product Analysis":
    st.title("Product analysis")
    st.caption("Charts and tables removed — add content step by step.")

# -----------------------------------------------------------------------------
# Infrastructure & Cost Analysis
# -----------------------------------------------------------------------------
elif page == "Infrastructure & Cost Analysis":
    st.title("Infrastructure & cost analysis")
    st.caption("Charts and tables removed — add content step by step.")
