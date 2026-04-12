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


def _monthly_infra_model_spend_usd(costs: pd.DataFrame) -> pd.DataFrame:
    """Calendar-month totals: sum of all non-`hour` USD columns in infrastructure_costs."""
    if costs.empty or "hour" not in costs.columns:
        return pd.DataFrame(columns=["month", "total_usd"])
    d = costs.dropna(subset=["hour"]).copy()
    value_cols = [c for c in d.columns if c != "hour"]
    if not value_cols:
        return pd.DataFrame(columns=["month", "total_usd"])
    d["total_usd"] = d[value_cols].fillna(0).sum(axis=1)
    out = (
        d.groupby(pd.Grouper(key="hour", freq="MS"), as_index=False)["total_usd"]
        .sum()
        .sort_values("hour")
        .rename(columns={"hour": "month"})
    )
    return out


PAYGO_USD_PER_CREDIT = 0.008
# Fixed monthly subscription by plan (USD). Enterprise is custom — not priced here (treated as 0).
SUBSCRIPTION_PLAN_USD_MONTH: dict[str, float] = {
    "project": 30.0,
    "bootstrap": 100.0,
    "startup": 220.0,
    "growth": 500.0,
}
REVENUE_MEAN_SINCE_UTC = pd.Timestamp("2025-11-01", tz="UTC")


def _subscription_monthly_charge_usd(plan_norm: str) -> float:
    return SUBSCRIPTION_PLAN_USD_MONTH.get(plan_norm, 0.0)


def _utc_month_starts_inclusive_range(
    hourly: pd.DataFrame, users_unique: pd.DataFrame, costs: pd.DataFrame
) -> pd.DatetimeIndex:
    bounds: list[pd.Timestamp] = []
    if "hour" in hourly.columns:
        s = pd.to_datetime(hourly["hour"], utc=True, errors="coerce").dropna()
        if not s.empty:
            bounds.extend([s.min(), s.max()])
    if "hour" in costs.columns:
        s = pd.to_datetime(costs["hour"], utc=True, errors="coerce").dropna()
        if not s.empty:
            bounds.extend([s.min(), s.max()])
    if "created_at" in users_unique.columns:
        s = pd.to_datetime(users_unique["created_at"], utc=True, errors="coerce").dropna()
        if not s.empty:
            bounds.extend([s.min(), s.max()])
    if not bounds:
        return pd.DatetimeIndex([], tz="UTC")
    lo = min(bounds).tz_convert("UTC").replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    hi = max(bounds).tz_convert("UTC").replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return pd.date_range(lo, hi, freq="MS", inclusive="both", tz="UTC")


def _monthly_paygo_revenue_usd(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty or "hour" not in hourly.columns or "paygo_credits_used" not in hourly.columns:
        return pd.DataFrame(columns=["month", "paygo_revenue_usd"])
    d = hourly.dropna(subset=["hour"]).copy()
    d["paygo_credits_used"] = pd.to_numeric(d["paygo_credits_used"], errors="coerce").fillna(0.0)
    g = d.groupby(pd.Grouper(key="hour", freq="MS"), as_index=False)["paygo_credits_used"].sum()
    g = g.rename(columns={"hour": "month"})
    g["paygo_revenue_usd"] = g["paygo_credits_used"] * PAYGO_USD_PER_CREDIT
    return g[["month", "paygo_revenue_usd"]]


def _monthly_subscription_revenue_usd(
    users_unique: pd.DataFrame, month_starts: pd.DatetimeIndex
) -> pd.DataFrame:
    if (
        users_unique.empty
        or "created_at" not in users_unique.columns
        or "plan" not in users_unique.columns
        or len(month_starts) == 0
    ):
        return pd.DataFrame(columns=["month", "subscription_revenue_usd"])
    uu = users_unique.dropna(subset=["user_id", "created_at"]).copy()
    uu["created_at"] = pd.to_datetime(uu["created_at"], utc=True, errors="coerce")
    uu = uu.dropna(subset=["created_at"])
    uu["plan_norm"] = uu["plan"].fillna("").astype(str).str.lower().str.strip()
    rows: list[dict] = []
    for ms in month_starts:
        nxt = ms + pd.DateOffset(months=1)
        elig = uu[uu["created_at"] < nxt]
        sub = float(elig["plan_norm"].map(_subscription_monthly_charge_usd).sum())
        rows.append({"month": ms, "subscription_revenue_usd": sub})
    return pd.DataFrame(rows)


def _monthly_revenue_and_spend_table(
    hourly: pd.DataFrame, users_unique: pd.DataFrame, costs: pd.DataFrame
) -> pd.DataFrame:
    months = _utc_month_starts_inclusive_range(hourly, users_unique, costs)
    if len(months) == 0:
        return pd.DataFrame(
            columns=[
                "month",
                "paygo_revenue_usd",
                "subscription_revenue_usd",
                "revenue_usd",
                "spend_usd",
            ]
        )
    base = pd.DataFrame({"month": months})
    pay = _monthly_paygo_revenue_usd(hourly)
    sub = _monthly_subscription_revenue_usd(users_unique, months)
    sp = _monthly_infra_model_spend_usd(costs).rename(columns={"total_usd": "spend_usd"})
    out = base.merge(pay, on="month", how="left").merge(sub, on="month", how="left").merge(sp, on="month", how="left")
    out["paygo_revenue_usd"] = out["paygo_revenue_usd"].fillna(0.0)
    out["subscription_revenue_usd"] = out["subscription_revenue_usd"].fillna(0.0)
    out["spend_usd"] = out["spend_usd"].fillna(0.0)
    out["revenue_usd"] = out["paygo_revenue_usd"] + out["subscription_revenue_usd"]
    return out


# -----------------------------------------------------------------------------
# Overview — Executive Summary
# -----------------------------------------------------------------------------
if page == "Overview":
    # --- Swap these blocks for dummy data vs. live data ---------------------------------
    total_users = int(df_users_unique["user_id"].nunique())
    paying_mask = _is_paying_user_row(df_users_unique)
    paying_users_count = int(paying_mask.sum())
    free_users_count = int(total_users - paying_users_count)

    if "request_type" in df_hourly.columns and "request_count" in df_hourly.columns:
        endpoint_dist = df_hourly.groupby(df_hourly["request_type"].fillna("(unknown)").astype(str))["request_count"].sum()
    else:
        endpoint_dist = pd.Series(dtype=float)

    # -------------------------------------------------------------------------------------

    st.title("Platform Overview: Executive Summary")
    st.markdown(
        "*A high-level view of user monetization and platform traffic distribution.*"
    )

    research_first_users = _count_post_launch_first_research_users(df_users_unique, df_hourly)
    pct_research_first = (
        (100.0 * research_first_users / total_users) if total_users else 0.0
    )
    _rev_spend = _monthly_revenue_and_spend_table(df_hourly, df_users_unique, df_costs)
    _spend_from_costs_only = _monthly_infra_model_spend_usd(df_costs)
    if not _spend_from_costs_only.empty and _spend_from_costs_only["total_usd"].notna().any():
        mean_monthly_infra_usd = float(_spend_from_costs_only["total_usd"].mean())
    else:
        mean_monthly_infra_usd = None
    if not _rev_spend.empty:
        _rs_from_nov = _rev_spend[_rev_spend["month"] >= REVENUE_MEAN_SINCE_UTC]
        mean_monthly_revenue_usd = (
            float(_rs_from_nov["revenue_usd"].mean()) if not _rs_from_nov.empty else None
        )
    else:
        mean_monthly_revenue_usd = None

    m_users, m_rev, m_infra = st.columns(3)
    with m_users:
        st.metric(
            "Total Users",
            f"{total_users:,}",
            delta=f"+{pct_research_first:.2f}%",
            delta_color="normal",
            help=(
                f"Users signed up after {RESEARCH_API_LAUNCH_UTC.day}/{RESEARCH_API_LAUNCH_UTC.month}/"
                f"{RESEARCH_API_LAUNCH_UTC.year} (Research API launched) who used Research as their "
                "first request in hourly usage (UTC)."
            ),
        )
    with m_rev:
        if mean_monthly_revenue_usd is not None:
            st.metric(
                "Mean monthly revenue (from Nov 2025, USD)",
                f"${mean_monthly_revenue_usd:,.0f}",
                help=(
                    "Average monthly **modeled** revenue for months from **Nov 2025** onward: "
                    f"PayGo (`paygo_credits_used` × ${PAYGO_USD_PER_CREDIT}/credit) plus fixed plan fees "
                    f"(Project ${SUBSCRIPTION_PLAN_USD_MONTH['project']:.0f}, Bootstrap "
                    f"${SUBSCRIPTION_PLAN_USD_MONTH['bootstrap']:.0f}, Startup "
                    f"${SUBSCRIPTION_PLAN_USD_MONTH['startup']:.0f}, Growth "
                    f"${SUBSCRIPTION_PLAN_USD_MONTH['growth']:.0f}/mo; Enterprise/custom not priced). "
                    "Subscription uses each user’s current **plan** from the users extract applied to every "
                    "month after signup (no plan history)."
                ),
            )
        else:
            st.metric(
                "Mean monthly revenue (from Nov 2025, USD)",
                "—",
                help="Could not compute monthly revenue (missing data).",
            )
    with m_infra:
        if mean_monthly_infra_usd is not None:
            st.metric(
                "Mean monthly infrastructure spend (USD)",
                f"${mean_monthly_infra_usd:,.0f}",
                help=(
                    "Mean of monthly spend **only for months that appear in `infrastructure_costs.csv`** "
                    "(same totals as the Spend bars for those months). The chart can include extra months "
                    "with $0 spend where usage/users exist but there is no cost row—those are excluded here."
                ),
            )
        else:
            st.metric(
                "Mean monthly infrastructure spend (USD)",
                "—",
                help="No usable monthly totals from `infrastructure_costs.csv`.",
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

    # Daily traffic (shared: left chart + start date to align user growth)
    _daily = pd.DataFrame()
    traffic_start = None
    if "hour" in df_hourly.columns and "request_count" in df_hourly.columns:
        _hu_pre = df_hourly.dropna(subset=["hour"]).copy()
        _hu_pre["day"] = pd.to_datetime(_hu_pre["hour"], utc=True).dt.normalize()
        _daily = (
            _hu_pre.groupby("day", as_index=False)["request_count"]
            .sum()
            .sort_values("day")
            .reset_index(drop=True)
        )
        if not _daily.empty:
            traffic_start = pd.Timestamp(_daily["day"].iloc[0])
            if traffic_start.tzinfo is None:
                traffic_start = traffic_start.tz_localize("UTC")

    ts_left, ts_right = st.columns(2)

    with ts_left:
        st.markdown("### Platform Traffic (7-Day Moving Average)")
        if _daily.empty:
            st.info("Missing `hour` or `request_count` in hourly usage data, or no rows after aggregation.")
        else:
            # min_periods=1: early dates use 1..6 days of history; from day 7 onward uses full 7-day window
            _daily = _daily.copy()
            _daily["requests_ma7"] = _daily["request_count"].rolling(window=7, min_periods=1).mean()
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
        if traffic_start is None:
            st.info("Need traffic data first to align the start date.")
        elif "created_at" not in df_users_unique.columns:
            st.info("Missing `created_at` in users data.")
        else:
            _uu = df_users_unique.dropna(subset=["created_at"]).copy()
            _uu["day"] = pd.to_datetime(_uu["created_at"], utc=True, errors="coerce").dt.normalize()
            _uu = _uu.dropna(subset=["day"])
            if _uu["day"].dt.tz is None:
                _uu["day"] = _uu["day"].dt.tz_localize("UTC")
            if _uu.empty:
                st.info("No valid signup dates for cumulative growth.")
            else:
                traffic_end = pd.Timestamp(_daily["day"].iloc[-1])
                if traffic_end.tzinfo is None:
                    traffic_end = traffic_end.tz_localize("UTC")
                user_last = pd.Timestamp(_uu["day"].max())
                if user_last.tzinfo is None:
                    user_last = user_last.tz_localize("UTC")
                day_end = max(traffic_end, user_last)

                users_before_traffic = int((_uu["day"] < traffic_start).sum())
                all_days = pd.date_range(traffic_start, day_end, freq="D", tz="UTC")
                new_by_day = _uu.groupby("day").size().reindex(all_days, fill_value=0)
                new_by_day.index.name = "day"
                _growth = new_by_day.reset_index(name="new_users")
                _growth["cumulative_users"] = users_before_traffic + _growth["new_users"].cumsum()

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
                st.caption(
                    "Series starts on the **first day with traffic**; the level includes users who signed up **before** that day."
                )

    if not _rev_spend.empty:
        st.markdown("### Monthly revenue vs platform spend (USD)")
        fig_rs = go.Figure(
            data=[
                go.Bar(
                    name="Revenue",
                    x=_rev_spend["month"],
                    y=_rev_spend["revenue_usd"],
                    marker_color="#059669",
                    hovertemplate="Revenue: $%{y:,.0f}<extra></extra>",
                ),
                go.Bar(
                    name="Spend",
                    x=_rev_spend["month"],
                    y=_rev_spend["spend_usd"],
                    marker_color="#6366f1",
                    hovertemplate="Spend: $%{y:,.0f}<extra></extra>",
                ),
            ]
        )
        fig_rs.update_layout(
            barmode="group",
            bargap=0.2,
            bargroupgap=0.1,
            height=400,
            margin=dict(t=24, b=48, l=56, r=24),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Month (UTC)",
            yaxis_title="USD",
        )
        fig_rs.update_yaxes(tickformat=",.0f")
        fig_rs.update_xaxes(tickformat="%b %Y")
        st.plotly_chart(fig_rs, use_container_width=True)
        st.caption(
            "Revenue = PayGo (`paygo_credits_used` × $0.008/credit) plus fixed monthly plan fees for every user "
            "who had signed up before that month (Project $30, Bootstrap $100, Startup $220, Growth $500; "
            "researcher/freemium $0; Enterprise/custom not priced). Each row uses the user’s **current** plan from "
            "the extract for all months (no historical plan changes). Spend = hourly infra + model cost columns summed by month."
        )

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
