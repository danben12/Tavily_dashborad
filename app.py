"""
Tavily Data Analyst assignment — Streamlit dashboard.
Loads CSVs from data.zip (deploy) or from the same folder / parent folder (local dev).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
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

# Match Streamlit dark UI: light axes/labels, dark plot surface (see .streamlit/config.toml).
pio.templates.default = "plotly_dark"

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


def _research_first_after_launch_metrics(
    users_unique: pd.DataFrame,
    hourly: pd.DataFrame,
    research: pd.DataFrame,
) -> tuple[int, int]:
    """Return (count whose first hourly row is Research, count joined after first research timestamp).

    Launch anchor = earliest ``timestamp`` in ``research_requests``. Cohort = ``created_at`` strictly after that.
    """
    if research.empty or "timestamp" not in research.columns:
        return (0, 0)
    if not {"user_id", "created_at"}.issubset(users_unique.columns):
        return (0, 0)
    if not {"user_id", "hour", "request_type"}.issubset(hourly.columns):
        return (0, 0)

    first_rs = research["timestamp"].dropna().min()
    if pd.isna(first_rs):
        return (0, 0)
    first_rs = pd.Timestamp(first_rs)
    if first_rs.tzinfo is None:
        first_rs = first_rs.tz_localize("UTC")
    else:
        first_rs = first_rs.tz_convert("UTC")

    u = users_unique.dropna(subset=["user_id", "created_at"]).copy()
    u["user_id"] = u["user_id"].astype(int)
    created = pd.to_datetime(u["created_at"], utc=True, errors="coerce")
    cohort = u.loc[created.notna() & (created > first_rs), "user_id"].unique()
    n_joined_after = int(len(cohort))
    if n_joined_after == 0:
        return (0, 0)

    h = hourly.dropna(subset=["hour", "user_id"]).copy()
    h["user_id"] = h["user_id"].astype(int)
    h = h[h["user_id"].isin(cohort)]
    if h.empty:
        return (0, n_joined_after)
    h = h.sort_values("hour")
    first = h.groupby("user_id", sort=False).first()
    rt = first["request_type"].fillna("").astype(str).str.lower().str.strip()
    return (int((rt == "research").sum()), n_joined_after)


def _research_pareto_pct_curve(research: pd.DataFrame, max_points: int = 2500) -> pd.DataFrame | None:
    """Cumulative % users (by descending request volume) vs cumulative % of research requests.

    One row per request in ``research_requests`` is counted toward that user's volume.
    """
    if research.empty or "user_id" not in research.columns:
        return None
    r = research.dropna(subset=["user_id"]).copy()
    r["user_id"] = r["user_id"].astype(int)
    by_u = r.groupby("user_id", observed=True).size().sort_values(ascending=False)
    if by_u.empty:
        return None
    total_req = int(by_u.sum())
    n = len(by_u)
    cum = by_u.cumsum().values.astype(float)
    pct_users = 100.0 * np.arange(1, n + 1) / n
    pct_req = 100.0 * cum / total_req
    out = pd.DataFrame({"pct_users": np.concatenate([[0.0], pct_users]), "pct_requests": np.concatenate([[0.0], pct_req])})
    if len(out) > max_points:
        idx = np.unique(np.linspace(0, len(out) - 1, max_points, dtype=int))
        out = out.iloc[idx].reset_index(drop=True)
    return out


def _research_cohort_user_ids(research: pd.DataFrame) -> np.ndarray:
    if research.empty or "user_id" not in research.columns:
        return np.array([], dtype=int)
    return research.dropna(subset=["user_id"])["user_id"].astype(int).unique()


def _user_monetized_row(plan: object, has_paygo: object) -> bool:
    """True if paid tier (non-researcher) or Pay-as-you-go."""
    try:
        paygo = bool(has_paygo)
    except (TypeError, ValueError):
        paygo = False
    p = str(plan if plan is not None else "").lower().strip()
    if paygo:
        return True
    return bool(p) and p != "researcher"


def _format_compact_amount(value: float) -> str:
    """One-decimal USD display with M / K suffix (e.g. $9.7 M, $12.4 K)."""
    if value is None or not np.isfinite(value):
        return "—"
    x = float(value)
    ax = abs(x)
    sign = "-" if x < 0 else ""
    if ax >= 1_000_000:
        return f"{sign}${ax / 1_000_000:.1f} M"
    if ax >= 1_000:
        return f"{sign}${ax / 1_000:.1f} K"
    return f"{sign}${ax:,.0f}"


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Product Analysis",
        "Infrastructure & Cost Analysis",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Tavily Data Analyst home assignment — Dan Benbenisti")


# -----------------------------------------------------------------------------
# Product Analysis — Part 1 (rebuild)
# -----------------------------------------------------------------------------
if page == "Product Analysis":
    st.title("Product analysis")
    if "user_id" in df_research.columns:
        n_research_users_rq = int(
            df_research.dropna(subset=["user_id"])["user_id"].astype(int).nunique()
        )
    else:
        n_research_users_rq = 0
    st.info(
        "**Methodological Note:** This dashboard presents an analysis of the 16,333 users who interacted "
        "with the Research API."
    )
    n_research_first, n_joined_after_research = _research_first_after_launch_metrics(
        df_users_unique, df_hourly, df_research
    )
    pct_of_post_launch = (
        (100.0 * n_research_first / n_joined_after_research) if n_joined_after_research else 0.0
    )
    n_rel = len(df_research)
    n_success = 0
    if n_rel and "status" in df_research.columns:
        _st_r = df_research["status"].fillna("").astype(str).str.lower().str.strip()
        n_success = int((_st_r == "success").sum())
    pct_reliability = (100.0 * n_success / n_rel) if n_rel else 0.0

    total_api_cost = 0.0
    if "request_cost" in df_research.columns:
        total_api_cost = float(pd.to_numeric(df_research["request_cost"], errors="coerce").fillna(0).sum())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Unique users (Research API)",
            f"{n_research_users_rq:,}",
            help="Distinct user_id in research_requests.csv.",
        )
    with m2:
        st.markdown(
            """
            <style>
            div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-child(2)
            div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
                color: #4ade80 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        _delta = f"+{pct_of_post_launch:.1f}%" if n_joined_after_research else None
        st.metric(
            "Research-first signups",
            f"{n_research_first:,}",
            delta=_delta,
            delta_color="normal",
            help=(
                "Joined after Research API launch; first hourly request is Research. "
                "Delta: % of users who signed up after Research appeared in data (earliest research_requests row)."
            ),
        )
    with m3:
        st.metric(
            "Overall research reliability",
            f"{pct_reliability:.1f}%",
            help="Share of research_requests rows where status is `success` (empty/missing counts as non-success).",
        )
    with m4:
        st.metric(
            "Total API cost",
            _format_compact_amount(total_api_cost),
            help="Sum of `request_cost` across all rows in research_requests (this extract).",
        )

    _cohort_set_prod = set(
        int(x) for x in _research_cohort_user_ids(df_research).tolist()
    )
    _h_prod = df_hourly.dropna(subset=["hour", "user_id"]).copy()
    _h_prod["user_id"] = _h_prod["user_id"].astype(int)
    _h_prod = _h_prod[_h_prod["user_id"].isin(_cohort_set_prod)]

    _sec_chart_h = 380

    st.markdown("---")
    st.subheader("1. Cohort Profile & Ecosystem Footprint")
    _s1_l, _s1_r = st.columns(2)
    with _s1_l:
        st.markdown("### Research API: unique users per day")
        st.caption(
            "**Blue:** distinct `user_id` with ≥1 **research** request that day (`research_requests`, UTC). "
            "**Purple:** distinct `user_id` with ≥1 **hourly_usage** row that day (any request type, UTC)—sample-wide DAU."
        )
        if not {"user_id", "timestamp"}.issubset(df_research.columns):
            st.info("research_requests is missing `user_id` or `timestamp` for this chart.")
        else:
            _rdu = df_research.dropna(subset=["user_id", "timestamp"]).copy()
            _rdu["user_id"] = _rdu["user_id"].astype(int)
            _rdu["_day"] = pd.to_datetime(_rdu["timestamp"], utc=True, errors="coerce").dt.normalize()
            _rdu["_day"] = _rdu["_day"].dt.tz_localize(None)
            _rdu = _rdu.dropna(subset=["_day"])
            if _rdu.empty:
                st.info("No valid timestamps in research_requests for this chart.")
            else:
                _daily_research_u = _rdu.groupby("_day", observed=True)["user_id"].nunique().sort_index()
                _daily_hourly_u = pd.Series(dtype=np.int64)
                if {"hour", "user_id"}.issubset(df_hourly.columns):
                    _hdu = df_hourly.dropna(subset=["hour", "user_id"]).copy()
                    _hdu["user_id"] = _hdu["user_id"].astype(int)
                    _hdu["_day"] = (
                        pd.to_datetime(_hdu["hour"], utc=True, errors="coerce")
                        .dt.normalize()
                        .dt.tz_localize(None)
                    )
                    _hdu = _hdu.dropna(subset=["_day"])
                    if not _hdu.empty:
                        _daily_hourly_u = (
                            _hdu.groupby("_day", observed=True)["user_id"].nunique().sort_index()
                        )

                _dau_merged = (
                    pd.DataFrame({"research": _daily_research_u})
                    .join(_daily_hourly_u.rename("hourly"), how="outer")
                    .sort_index()
                )

                fig_dau = go.Figure()
                fig_dau.add_trace(
                    go.Scatter(
                        x=_dau_merged.index,
                        y=_dau_merged["research"],
                        mode="lines",
                        line=dict(color="#38bdf8", width=2.2),
                        fill="tozeroy",
                        fillcolor="rgba(56, 189, 248, 0.18)",
                        name="Research API",
                        connectgaps=False,
                        hovertemplate="Research API<br>%{x|%Y-%m-%d}<br>%{y:,} users<extra></extra>",
                    )
                )
                if _daily_hourly_u.size > 0:
                    fig_dau.add_trace(
                        go.Scatter(
                            x=_dau_merged.index,
                            y=_dau_merged["hourly"],
                            mode="lines",
                            line=dict(color="#a78bfa", width=2, dash="solid"),
                            name="Hourly usage (all types)",
                            connectgaps=False,
                            hovertemplate="Hourly (any product)<br>%{x|%Y-%m-%d}<br>%{y:,} users<extra></extra>",
                        )
                    )
                fig_dau.update_layout(
                    xaxis_title="Date (UTC)",
                    yaxis_title="Unique users",
                    height=_sec_chart_h,
                    margin=dict(t=16, b=48, l=52, r=16),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0,
                        font=dict(size=11),
                    ),
                    xaxis=dict(gridcolor="rgba(148,163,184,0.25)", tickangle=-35),
                    yaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
                )
                st.plotly_chart(fig_dau, use_container_width=True)

    with _s1_r:
        st.markdown("### Query is still the Backbone for Research Users")
        st.caption("Total request volume in hourly_usage for the Research API cohort, by request type.")
        if _h_prod.empty or "request_type" not in _h_prod.columns:
            st.info("No hourly usage for cohort users.")
        else:
            _vol = (
                _h_prod.groupby("request_type", observed=True)["request_count"]
                .sum()
                .sort_values(ascending=True)
            )
            _lbls = _vol.index.astype(str)
            _fill = []
            _line = []
            for _lb in _lbls:
                _is_res = _lb.lower().strip() == "research"
                _fill.append("#ea580c" if _is_res else "#ffffff")
                _line.append("#c2410c" if _is_res else "#94a3b8")
            fig_bar = go.Figure(
                go.Bar(
                    x=_vol.values,
                    y=_lbls,
                    orientation="h",
                    marker=dict(color=_fill, line=dict(color=_line, width=1)),
                    hovertemplate="%{y}<br>%{x:,.0f} requests<extra></extra>",
                )
            )
            fig_bar.update_layout(
                xaxis_title="Total requests (sum of request_count)",
                yaxis_title="",
                height=_sec_chart_h,
                margin=dict(t=16, b=48, l=100, r=16),
                xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.28)"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Profitability vs. Resource Distribution")
    _s2_l, _s2_r = st.columns(2)
    with _s2_l:
        st.markdown("### Pro request: cost vs. estimated revenue")
        if "model" not in df_research.columns or "request_cost" not in df_research.columns:
            st.info("research_requests needs `model` and `request_cost` for this chart.")
        else:
            _pro_mask = df_research["model"].astype(str).str.strip().str.lower() == "pro"
            _pro_rc = pd.to_numeric(df_research.loc[_pro_mask, "request_cost"], errors="coerce").dropna()
            if _pro_rc.empty:
                st.info("No rows with `model == 'pro'` and valid `request_cost` in this extract.")
            else:
                _avg_cost_usd = float(_pro_rc.mean() / 1000.0)
                _rev_pro = 0.28
                fig_ue = go.Figure(
                    go.Bar(
                        x=["Average cost (Pro)", "Estimated revenue (Pro)"],
                        y=[_avg_cost_usd, _rev_pro],
                        marker_color=["#94a3b8", "#22c55e"],
                        text=[f"${_avg_cost_usd:.3f}", f"${_rev_pro:.2f}"],
                        textposition="outside",
                        hovertemplate="%{x}<br>$%{y:.4f}<extra></extra>",
                    )
                )
                fig_ue.update_layout(
                    yaxis_title="USD",
                    height=_sec_chart_h,
                    margin=dict(t=24, b=48, l=52, r=16),
                    showlegend=False,
                    xaxis=dict(tickangle=-12),
                    yaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
                )
                st.plotly_chart(fig_ue, use_container_width=True)
                st.caption(
                    "Insight: The base unit economics are profitable (positive margin), but overall profitability "
                    "is eroded by free-tier distribution."
                )

    with _s2_r:
        st.markdown("### Research requests vs users (Pareto)")
        _pareto_pct = _research_pareto_pct_curve(df_research)
        if _pareto_pct is None or len(_pareto_pct) < 2:
            st.info("Not enough data in research_requests to plot cumulative % users vs % requests.")
        else:
            st.caption(
                "Users sorted by **number of research requests** (highest first). "
                "Each point: include the top *x*% of those users → they account for *y*% of all requests in this extract."
            )
            fig_par = go.Figure()
            fig_par.add_trace(
                go.Scatter(
                    x=_pareto_pct["pct_users"],
                    y=_pareto_pct["pct_requests"],
                    mode="lines",
                    line=dict(color="#ea580c", width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(234, 88, 12, 0.12)",
                    name="Actual",
                    hovertemplate="Users: %{x:.1f}%<br>Requests: %{y:.1f}%<extra></extra>",
                )
            )
            fig_par.add_trace(
                go.Scatter(
                    x=[0, 100],
                    y=[0, 100],
                    mode="lines",
                    line=dict(color="#94a3b8", width=1, dash="dash"),
                    name="Equal share",
                    hoverinfo="skip",
                )
            )
            fig_par.update_layout(
                xaxis_title="Cumulative % of users (by request volume rank)",
                yaxis_title="Cumulative % of research requests",
                xaxis=dict(
                    range=[0, 100],
                    dtick=10,
                    ticksuffix="%",
                    gridcolor="rgba(148,163,184,0.25)",
                ),
                yaxis=dict(
                    range=[0, 100],
                    dtick=10,
                    ticksuffix="%",
                    gridcolor="rgba(148,163,184,0.25)",
                ),
                height=_sec_chart_h,
                margin=dict(t=24, b=48, l=56, r=24),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_par, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Friction & Drop-off (User Experience)")
    _s3_l, _s3_r = st.columns(2)
    with _s3_l:
        _bin_lbls = ["0-30s", "30-60s", "60-90s", "90-120s", "120s+"]
        if not {"response_time_seconds", "status"}.issubset(df_research.columns):
            st.info("research_requests needs `response_time_seconds` and `status` for this chart.")
        else:
            _dz = df_research.dropna(subset=["response_time_seconds"]).copy()
            _dz["response_time_seconds"] = pd.to_numeric(_dz["response_time_seconds"], errors="coerce")
            _dz = _dz.dropna(subset=["response_time_seconds"])
            if _dz.empty:
                st.info("No valid `response_time_seconds` in research_requests.")
            else:
                _dz["_st"] = _dz["status"].fillna("").astype(str).str.lower().str.strip()
                _dz["time_bin"] = pd.cut(
                    _dz["response_time_seconds"],
                    bins=[0, 30, 60, 90, 120, np.inf],
                    labels=_bin_lbls,
                    right=False,
                    include_lowest=True,
                )
                _dz = _dz.dropna(subset=["time_bin"])
                _can_rates = (
                    _dz.groupby("time_bin", observed=False)["_st"]
                    .agg(lambda s: 100.0 * float((s == "cancelled").sum()) / max(len(s), 1))
                    .reindex(_bin_lbls)
                    .fillna(0.0)
                )
                _bar_colors = ["#ef4444" if b == "90-120s" else "#3b82f6" for b in _can_rates.index.astype(str)]
                fig_dz = go.Figure(
                    go.Bar(
                        x=_can_rates.index.astype(str),
                        y=_can_rates.values,
                        marker_color=_bar_colors,
                        hovertemplate="%{x}<br>Cancellation rate: %{y:.2f}%<extra></extra>",
                    )
                )
                fig_dz.update_layout(
                    title=dict(
                        text="The Danger Zone: Cancellation Rate by Latency",
                        x=0.5,
                        xanchor="center",
                    ),
                    xaxis_title="Response time bucket",
                    yaxis_title="Cancellation rate (%)",
                    height=_sec_chart_h,
                    margin=dict(t=48, b=48, l=52, r=16),
                    showlegend=False,
                    yaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
                    xaxis=dict(type="category", categoryorder="array", categoryarray=_bin_lbls),
                )
                st.plotly_chart(fig_dz, use_container_width=True)

    with _s3_r:
        st.markdown("### Product stickiness: active days per user")
        if not {"user_id", "timestamp"}.issubset(df_research.columns):
            st.info("research_requests is missing `user_id` or `timestamp` for this chart.")
        else:
            _stk = df_research.dropna(subset=["user_id", "timestamp"]).copy()
            _stk["user_id"] = _stk["user_id"].astype(int)
            _stk["_date"] = pd.to_datetime(_stk["timestamp"], utc=True, errors="coerce").dt.date
            _stk = _stk.dropna(subset=["_date"])
            if _stk.empty:
                st.info("No valid timestamps in research_requests for this chart.")
            else:
                _active_days_per_user = (
                    _stk.groupby("user_id", observed=True)["_date"].nunique().reset_index()
                )
                _active_days_per_user.columns = ["user_id", "days_active"]
                _user_distribution = (
                    _active_days_per_user.groupby("days_active", observed=True)["user_id"]
                    .count()
                    .reset_index()
                )
                _user_distribution.columns = ["days_active", "user_count"]
                _user_distribution = _user_distribution.sort_values("days_active")
                fig_stick = px.bar(
                    _user_distribution,
                    x="days_active",
                    y="user_count",
                    labels={
                        "days_active": "Total days used",
                        "user_count": "Number of users",
                    },
                    color_discrete_sequence=["#38bdf8"],
                )
                fig_stick.update_layout(
                    title=dict(
                        text="Product Stickiness: Distribution of Active Days per User",
                        x=0.5,
                        xanchor="center",
                    ),
                    xaxis_title="Total Days Used",
                    yaxis_title="Number of Users",
                    showlegend=False,
                    height=_sec_chart_h,
                    margin=dict(t=48, b=48, l=56, r=16),
                    xaxis=dict(type="category", gridcolor="rgba(148,163,184,0.25)"),
                    yaxis=dict(
                        type="log",
                        gridcolor="rgba(148,163,184,0.25)",
                        tickmode="array",
                        tickvals=[1, 10, 100, 1000, 10000],
                        ticktext=["1", "10", "100", "1,000", "10,000"],
                    ),
                )
                fig_stick.update_traces(
                    hovertemplate="Days active: %{x}<br>Users: %{y:,}<extra></extra>",
                )
                st.plotly_chart(fig_stick, use_container_width=True)
                st.caption(
                    "Insight: This distribution highlights product stickiness. A heavy concentration on **1 day** "
                    "indicates users trying the feature once, while the tail represents highly engaged, retained users."
                )

    st.info(
        "Additional Part 1 sections (questions, hypotheses, KPIs, visuals) will be added here. "
        "Hourly usage and user metadata remain available from `load_data()`."
    )

# -----------------------------------------------------------------------------
# Infrastructure & Cost Analysis
# -----------------------------------------------------------------------------
elif page == "Infrastructure & Cost Analysis":
    st.title("Infrastructure & cost analysis")
    st.caption("Charts and tables removed — add content step by step.")
