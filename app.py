"""
Tavily Data Analyst assignment — Streamlit dashboard.
Loads CSVs from data.zip (deploy) or from the same folder / parent folder (local dev).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

    for c in df_costs.columns:
        if c != "hour":
            df_costs[c] = pd.to_numeric(df_costs[c], errors="coerce")

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


def _is_paying_user_row(u: pd.DataFrame) -> pd.Series:
    """Paying = Pay-as-you-go and/or non-freemium plan. Adjust FREEMIUM_PLANS as needed."""
    pl = u["plan"].fillna("").astype(str).str.lower().str.strip() if "plan" in u.columns else pd.Series("", index=u.index)
    freemium = pl.eq("researcher") | pl.eq("(unknown)") | pl.eq("unknown")
    paygo = u["has_paygo"].astype(bool) if "has_paygo" in u.columns else pd.Series(False, index=u.index)
    on_paid_plan = ~freemium & pl.ne("")
    return paygo | on_paid_plan


# Research API go-live (UTC calendar date); align created_at / hour comparisons to UTC.
RESEARCH_API_LAUNCH_UTC = pd.Timestamp("2025-11-23", tz="UTC")


def _count_post_launch_first_research_users(users_unique: pd.DataFrame, hourly: pd.DataFrame) -> int:
    """Users with created_at >= launch whose earliest hourly_usage row (by `hour`) is Research.

    Matches notebook logic: sort by time, then ``groupby(user_id).first()``. If several rows share
    the same earliest hour, the first row after sorting is used. Case-insensitive ``request_type``.
    Post-launch users with no hourly rows are excluded (no first request in this extract).
    """
    if not {"user_id", "created_at"}.issubset(users_unique.columns):
        return 0
    if not {"user_id", "hour", "request_type"}.issubset(hourly.columns):
        return 0
    u = users_unique.dropna(subset=["user_id", "created_at"]).copy()
    if u.empty:
        return 0
    u["user_id"] = u["user_id"].astype(int)
    created = pd.to_datetime(u["created_at"], utc=True, errors="coerce")
    post_ids = u.loc[created >= RESEARCH_API_LAUNCH_UTC, "user_id"].unique()
    if len(post_ids) == 0:
        return 0
    h = hourly.dropna(subset=["hour", "user_id"]).copy()
    h["user_id"] = h["user_id"].astype(int)
    h = h[h["user_id"].isin(post_ids)]
    if h.empty:
        return 0
    h = h.sort_values("hour")
    first = h.groupby("user_id", sort=False).first()
    rt = first["request_type"].fillna("").astype(str).str.lower().str.strip()
    return int((rt == "research").sum())


def _count_research_api_users(hourly: pd.DataFrame) -> int:
    """Distinct ``user_id`` with positive total ``request_count`` on ``request_type`` research."""
    need = {"user_id", "request_type", "request_count"}
    if hourly.empty or not need.issubset(hourly.columns):
        return 0
    h = hourly.dropna(subset=["user_id"]).copy()
    h["request_type"] = h["request_type"].fillna("").astype(str).str.lower().str.strip()
    rh = h[h["request_type"] == "research"]
    if rh.empty:
        return 0
    rh = rh.copy()
    rh["request_count"] = pd.to_numeric(rh["request_count"], errors="coerce").fillna(0.0)
    by_u = rh.groupby("user_id")["request_count"].sum()
    return int((by_u > 0).sum())


def _research_requests_pareto_df(hourly: pd.DataFrame, n_tiers: int = 10) -> pd.DataFrame | None:
    """Pareto table: users ranked by total research ``request_count``, then bucketed for display.

    Returns columns ``label``, ``research_requests``, ``cum_pct_requests`` (0–100), ``users_in_bucket``.
    """
    need = {"user_id", "request_type", "request_count"}
    if hourly.empty or not need.issubset(hourly.columns):
        return None
    h = hourly.dropna(subset=["user_id"]).copy()
    h["request_type"] = h["request_type"].fillna("").astype(str).str.lower().str.strip()
    rh = h[h["request_type"] == "research"]
    if rh.empty:
        return None
    rh["request_count"] = pd.to_numeric(rh["request_count"], errors="coerce").fillna(0.0)
    by_u = rh.groupby("user_id", as_index=False)["request_count"].sum()
    by_u = (
        by_u[by_u["request_count"] > 0]
        .sort_values("request_count", ascending=False)
        .reset_index(drop=True)
    )
    n = len(by_u)
    if n == 0:
        return None

    total_req = float(by_u["request_count"].sum())
    rows: list[dict[str, object]] = []

    if n <= 25:
        cum = 0.0
        for i, (_, r) in enumerate(by_u.iterrows(), start=1):
            v = float(r["request_count"])
            cum += v
            rows.append(
                {
                    "label": f"Rank {i}",
                    "research_requests": v,
                    "cum_pct_requests": 100.0 * cum / total_req,
                    "users_in_bucket": 1,
                }
            )
    else:
        q = min(n_tiers, n)
        by_u["_tier"] = pd.qcut(np.arange(n), q=q, labels=False, duplicates="drop")
        for tid in sorted(by_u["_tier"].dropna().unique()):
            chunk = by_u[by_u["_tier"] == tid]
            uc = int(len(chunk))
            req_sum = float(chunk["request_count"].sum())
            i_lo = int(chunk.index.min()) + 1
            i_hi = int(chunk.index.max()) + 1
            rows.append(
                {
                    "label": f"Ranks {i_lo}–{i_hi} ({uc} users)",
                    "research_requests": req_sum,
                    "cum_pct_requests": 0.0,
                    "users_in_bucket": uc,
                }
            )
        cum = 0.0
        for row in rows:
            cum += float(row["research_requests"])
            row["cum_pct_requests"] = 100.0 * cum / total_req

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Overview — placeholder (full-population story TBD)
# -----------------------------------------------------------------------------
if page == "Overview":
    st.title("Overview")
    st.info(
        "Nothing here yet. This extract is a **sample** (not full production scale), so the previous "
        "Overview metrics and time series were misleading as a platform-wide summary. "
        "You can refine this page once the narrative matches complete data."
    )

# -----------------------------------------------------------------------------
# Product Analysis
# -----------------------------------------------------------------------------
elif page == "Product Analysis":
    st.title("Product analysis")
    st.caption(
        "All figures below are limited to **users and hourly usage in this assignment extract** "
        "(e.g. Research-related sample), not the full Tavily user base or total traffic."
    )

    total_users = int(df_users_unique["user_id"].nunique())
    n_research_api_users = _count_research_api_users(df_hourly)
    paying_mask = _is_paying_user_row(df_users_unique)
    paying_users_count = int(paying_mask.sum())
    free_users_count = int(total_users - paying_users_count)

    if "request_type" in df_hourly.columns and "request_count" in df_hourly.columns:
        endpoint_dist = df_hourly.groupby(df_hourly["request_type"].fillna("(unknown)").astype(str))[
            "request_count"
        ].sum()
    else:
        endpoint_dist = pd.Series(dtype=float)

    research_first_users = _count_post_launch_first_research_users(df_users_unique, df_hourly)
    if n_research_api_users == total_users:
        metric_label = "Total Users"
        metric_users = total_users
        metric_help_extra = ""
    else:
        metric_label = "Research API users"
        metric_users = n_research_api_users
        d = n_research_api_users - total_users
        if d > 0:
            metric_help_extra = (
                f" From **hourly_usage**: distinct users with research requests. "
                f"**{d}** more than distinct `user_id` in `users.csv` ({total_users:,})."
            )
        else:
            metric_help_extra = (
                f" From **hourly_usage**: distinct users with research requests. "
                f"`users.csv` has **{-d}** users without positive research volume here."
            )
    pct_research_first = (
        (100.0 * research_first_users / metric_users) if metric_users else 0.0
    )
    st.metric(
        metric_label,
        f"{metric_users:,}",
        delta=f"+{pct_research_first:.2f}%",
        delta_color="normal",
        help=(
            f"Users signed up after {RESEARCH_API_LAUNCH_UTC.day}/{RESEARCH_API_LAUNCH_UTC.month}/"
            f"{RESEARCH_API_LAUNCH_UTC.year} (Research API launched) who used Research as their "
            f"first request in hourly usage (UTC). Delta = that count as % of **{metric_label.lower()}**."
            + metric_help_extra
        ),
    )

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

    _pareto = _research_requests_pareto_df(df_hourly)
    st.markdown("### Research usage concentration (Pareto)")
    if _pareto is None or _pareto.empty:
        st.info("No research `request_type` rows with positive `request_count` for a Pareto chart.")
    else:
        n_research_users = _count_research_api_users(df_hourly)
        st.caption(
            f"Users with at least one research request in this extract: **{n_research_users:,}**. "
            "Bars = research request volume per bucket (users sorted heaviest-first). "
            "Line = cumulative share of all research requests (right axis)."
        )
        fig_p = make_subplots(specs=[[{"secondary_y": True}]])
        fig_p.add_trace(
            go.Bar(
                x=_pareto["label"],
                y=_pareto["research_requests"],
                name="Research requests",
                marker_color="#ea580c",
                hovertemplate=(
                    "<b>%{x}</b><br>Requests: %{y:,.0f}<br>Users in bucket: %{customdata}<extra></extra>"
                ),
                customdata=_pareto["users_in_bucket"],
            ),
            secondary_y=False,
        )
        fig_p.add_trace(
            go.Scatter(
                x=_pareto["label"],
                y=_pareto["cum_pct_requests"],
                name="Cumulative % of research requests",
                mode="lines+markers",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=8),
                hovertemplate="Cumulative: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )
        fig_p.update_layout(
            height=max(420, 48 * len(_pareto)),
            margin=dict(t=40, b=120, l=56, r=56),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(tickangle=-35),
        )
        fig_p.update_yaxes(title_text="Research requests (sum of hourly counts)", secondary_y=False)
        fig_p.update_yaxes(
            title_text="Cumulative % of research requests",
            range=[0, 100],
            secondary_y=True,
        )
        st.plotly_chart(fig_p, use_container_width=True)

# -----------------------------------------------------------------------------
# Infrastructure & Cost Analysis
# -----------------------------------------------------------------------------
elif page == "Infrastructure & Cost Analysis":
    st.title("Infrastructure & cost analysis")
    st.caption("Charts and tables removed — add content step by step.")
