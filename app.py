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
import plotly.io as pio
import streamlit as st

pio.templates.default = "plotly_dark"

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


# Monthly subscription USD (assignment assumptions; Enterprise = $1k/mo).
_PLAN_MONTHLY_USD: dict[str, float] = {
    "researcher": 0.0,
    "project": 30.0,
    "bootstrap": 99.0,
    "startup": 199.0,
    "growth": 499.0,
    "enterprise": 1000.0,
}
_CREDIT_USD = 0.008
_FREE_INCLUDED_CREDITS = 1000
_PROJECT_FLAT_USD = 30.0


def _plan_monthly_usd(plan: object) -> float:
    p = str(plan if plan is not None else "").lower().strip()
    return float(_PLAN_MONTHLY_USD.get(p, 0.0))


def _to_naive_utc_series(ser: pd.Series) -> pd.Series:
    """UTC-aware datetimes → timezone-naive UTC wall time."""
    t = pd.to_datetime(ser, utc=True, errors="coerce")
    if getattr(t.dt, "tz", None) is not None:
        return t.dt.tz_convert("UTC").dt.tz_localize(None)
    return t


def _research_cost_usd_from_credits(research: pd.DataFrame) -> float:
    """REQUEST_COST treated as platform credits; retail USD = credits × $0.008."""
    if research.empty or "request_cost" not in research.columns:
        return 0.0
    c = pd.to_numeric(research["request_cost"], errors="coerce").fillna(0.0)
    return float(c.sum() * _CREDIT_USD)


def _first_hourly_action_research_rate(hourly: pd.DataFrame) -> tuple[float, int, int]:
    """Return (pct, n_research_first, n_with_hourly)."""
    if not {"user_id", "hour", "request_type"}.issubset(hourly.columns):
        return (0.0, 0, 0)
    h = hourly.dropna(subset=["hour", "user_id"]).copy()
    h["user_id"] = h["user_id"].astype(int)
    h["_hs"] = _to_naive_utc_series(h["hour"])
    h = h.dropna(subset=["_hs"])
    h = h.sort_values(["user_id", "_hs"])
    first = h.groupby("user_id", sort=False).first()
    n = len(first)
    if n == 0:
        return (0.0, 0, 0)
    rt = first["request_type"].fillna("").astype(str).str.lower().str.strip()
    n_r = int((rt == "research").sum())
    return (100.0 * n_r / n, n_r, n)


def _user_first_hourly_ts_and_type(hourly: pd.DataFrame) -> pd.DataFrame | None:
    """Per user_id: first hour (naive UTC) and first request_type."""
    if not {"user_id", "hour", "request_type"}.issubset(hourly.columns):
        return None
    h = hourly.dropna(subset=["hour", "user_id"]).copy()
    h["user_id"] = h["user_id"].astype(int)
    h["_hn"] = _to_naive_utc_series(h["hour"])
    h = h.dropna(subset=["_hn"])
    h = h.sort_values(["user_id", "_hn"])
    first = h.groupby("user_id", sort=False).first().reset_index()
    return first[["user_id", "_hn", "request_type"]].rename(columns={"_hn": "first_hour"})


_ACQ_PCT_GLOBAL = _first_hourly_action_research_rate(df_hourly)[0]

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
if page == "Product Analysis":
    st.sidebar.info(
        "**Strategic recommendations**\n\n"
        "• **Gate the Pro model:** Move the **pro** Research model to **paid tiers only** to protect unit economics.\n\n"
        "• **Fix the arbitrage:** Lower **Project** to **$20/mo** or increase included credits to **~5,000** so PAYGO "
        "does not undercut the upgrade path past ~4,750 credits/mo.\n\n"
        f"• **Loss-leader rationale:** ~**{_ACQ_PCT_GLOBAL:.0f}%** of users with hourly data start on **research**—a strong "
        "acquisition lever worth keeping alongside the pricing fixes above."
    )
st.sidebar.caption("Tavily Data Analyst home assignment — Dan Benbenisti")


# -----------------------------------------------------------------------------
# Product Analysis
# -----------------------------------------------------------------------------
if page == "Product Analysis":
    st.title("Product Analysis: Research API Strategic Deep-Dive")

    # --- Global KPIs (USD; research cost = REQUEST_COST credits × $0.008) ---
    _sub_mrr = 0.0
    if not df_users_unique.empty and "plan" in df_users_unique.columns:
        _sub_mrr = float(df_users_unique["plan"].map(_plan_monthly_usd).fillna(0).sum())

    _paygo_credits_total = 0.0
    if "paygo_credits_used" in df_hourly.columns:
        _paygo_credits_total = float(pd.to_numeric(df_hourly["paygo_credits_used"], errors="coerce").fillna(0).sum())
    _paygo_usd_total = _paygo_credits_total * _CREDIT_USD

    _total_revenue_cohort = _sub_mrr + _paygo_usd_total
    _research_cost_usd = _research_cost_usd_from_credits(df_research)
    _acq_pct, _acq_n, _acq_denom = _first_hourly_action_research_rate(df_hourly)
    _net_margin = _total_revenue_cohort - _research_cost_usd

    st.subheader("1. Cohort economics & acquisition")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            "Total revenue (cohort)",
            _format_compact_amount(_total_revenue_cohort),
            help=(
                "Sum of **monthly subscription run-rate** (each user’s tier price) plus **all-time PAYGO** in this "
                f"extract (`paygo_credits_used` × ${_CREDIT_USD}). Mixed window—use charts for monthly view."
            ),
        )
    with k2:
        st.metric(
            "Total research cost (retail)",
            _format_compact_amount(_research_cost_usd),
            help="Sum of `request_cost` as **platform credits** × $0.008 (retail USD) over all research_requests.",
        )
    with k3:
        st.metric(
            "Acquisition power",
            f"{_acq_pct:.1f}%",
            help="Share of users with hourly data whose **first** recorded request type is `research`.",
        )
    with k4:
        st.metric(
            "Net margin",
            _format_compact_amount(_net_margin),
            help="Total revenue (cohort) minus total research cost (retail USD).",
        )

    st.markdown("---")

    # --- Chart 2: Pricing arbitrage ---
    st.subheader("2. The pricing arbitrage (Free + PAYGO vs. Project)")
    _credits_x = np.arange(1000, 15001, 100)
    _free_line = np.maximum(0.0, (_credits_x - _FREE_INCLUDED_CREDITS) * _CREDIT_USD)
    _project_line = np.full_like(_credits_x, _PROJECT_FLAT_USD, dtype=float)
    _arb_mask = _free_line < _PROJECT_FLAT_USD
    _arb_x0 = float(_credits_x[_arb_mask][0]) if _arb_mask.any() else 1000.0
    _arb_x1 = float(_credits_x[_arb_mask][-1]) if _arb_mask.any() else 1000.0

    fig_arb = go.Figure()
    fig_arb.add_trace(
        go.Scatter(
            x=_credits_x,
            y=_free_line,
            mode="lines",
            name="Free + PAYGO",
            line=dict(color="#38bdf8", width=2.5),
            hovertemplate="Credits: %{x:,}<br>Cost: $%{y:.2f}<extra></extra>",
        )
    )
    fig_arb.add_trace(
        go.Scatter(
            x=_credits_x,
            y=_project_line,
            mode="lines",
            name="Project ($30/mo)",
            line=dict(color="#fbbf24", width=2, dash="dash"),
            hovertemplate="Credits: %{x:,}<br>Project: $%{y:.2f}<extra></extra>",
        )
    )
    fig_arb.add_vrect(
        x0=_arb_x0,
        x1=_arb_x1,
        fillcolor="rgba(56, 189, 248, 0.12)",
        layer="below",
        line_width=0,
    )
    fig_arb.update_layout(
        xaxis_title="Monthly credits needed",
        yaxis_title="Monthly out-of-pocket (USD)",
        height=420,
        margin=dict(t=24, b=48, l=56, r=24),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
    )
    st.plotly_chart(fig_arb, use_container_width=True)
    st.caption(
        "Shaded **arbitrage zone**: Free + PAYGO is cheaper than Project until about **4,750** credits/month "
        "(break-even: 1,000 included + 3,750 × $0.008 = $30). "
        "Mathematically, users have little incentive to upgrade to Project until they exceed that level."
    )

    st.markdown("---")

    # --- Chart 3: Growth vs burn (monthly) ---
    st.subheader("3. Growth vs. burn (revenue vs. research cost)")
    _months = pd.period_range("2025-11", "2026-03", freq="M")

    _rs_ts = _to_naive_utc_series(df_research["timestamp"]) if "timestamp" in df_research.columns else pd.Series(dtype="datetime64[ns]")
    _rs_cost = pd.to_numeric(df_research["request_cost"], errors="coerce").fillna(0.0) * _CREDIT_USD if "request_cost" in df_research.columns else pd.Series(0.0, index=df_research.index)
    _rdf = pd.DataFrame({"_ts": _rs_ts, "_usd": _rs_cost}).dropna(subset=["_ts"])
    _rdf["_ym"] = _rdf["_ts"].dt.to_period("M")
    _cost_by_m = _rdf.groupby("_ym", observed=True)["_usd"].sum()

    _hh = _to_naive_utc_series(df_hourly["hour"]) if "hour" in df_hourly.columns else pd.Series(dtype="datetime64[ns]")
    _hp = pd.to_numeric(df_hourly["paygo_credits_used"], errors="coerce").fillna(0.0) if "paygo_credits_used" in df_hourly.columns else pd.Series(0.0, index=df_hourly.index)
    _hdf = pd.DataFrame({"_hn": _hh, "_pg": _hp}).dropna(subset=["_hn"])
    _hdf["_ym"] = _hdf["_hn"].dt.to_period("M")
    _paygo_by_m = _hdf.groupby("_ym", observed=True)["_pg"].sum() * _CREDIT_USD

    _sub_by_m: list[float] = []
    if not df_users_unique.empty and {"created_at", "plan"}.issubset(df_users_unique.columns):
        _uu = df_users_unique.copy()
        _uu["_c"] = _to_naive_utc_series(_uu["created_at"])
        _uu = _uu.dropna(subset=["_c"])
        for _per in _months:
            _end = _per.to_timestamp(how="end").normalize()
            _mask = _uu["_c"] <= _end
            _sub_by_m.append(float(_uu.loc[_mask, "plan"].map(_plan_monthly_usd).fillna(0).sum()))
    else:
        _sub_by_m = [0.0] * len(_months)

    _m_labels = [str(p) for p in _months]
    _rev_bars = [
        _sub_by_m[i] + float(_paygo_by_m.get(_months[i], 0.0)) for i in range(len(_months))
    ]
    _cost_line = [float(_cost_by_m.get(_months[i], 0.0)) for i in range(len(_months))]

    fig_gb = go.Figure()
    fig_gb.add_trace(
        go.Bar(
            x=_m_labels,
            y=_rev_bars,
            name="Revenue (sub run-rate EOM + PAYGO)",
            marker_color="#22c55e",
            hovertemplate="%{x}<br>Revenue: $%{y:,.0f}<extra></extra>",
        )
    )
    fig_gb.add_trace(
        go.Scatter(
            x=_m_labels,
            y=_cost_line,
            name="Research cost (retail)",
            mode="lines+markers",
            line=dict(color="#ef4444", width=2.5),
            marker=dict(size=8),
            yaxis="y2",
            hovertemplate="%{x}<br>Research cost: $%{y:,.0f}<extra></extra>",
        )
    )
    fig_gb.update_layout(
        height=460,
        margin=dict(t=24, b=48, l=56, r=56),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
        yaxis=dict(title="Revenue (USD)", gridcolor="rgba(148,163,184,0.2)", side="left"),
        yaxis2=dict(
            title="Research cost (USD)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        xaxis=dict(title="Month (UTC)", gridcolor="rgba(148,163,184,0.2)"),
    )
    st.plotly_chart(fig_gb, use_container_width=True)
    st.caption(
        "Bars: **end-of-month subscription run-rate** (users signed up by month-end × tier price) plus **PAYGO** "
        "credits used that month. Line: **Research** `request_cost` retail (credits × $0.008) in the month."
    )

    st.markdown("---")

    # --- Chart 4: CAC-style efficiency ---
    st.subheader("4. CAC efficiency: cost per research-first user (by month)")
    _fe = _user_first_hourly_ts_and_type(df_hourly)
    if _fe is None or _fe.empty:
        st.info("Not enough hourly data to attribute research-first users by month.")
    else:
        _fe = _fe.copy()
        _fe["_rt"] = _fe["request_type"].fillna("").astype(str).str.lower().str.strip()
        _fe = _fe[_fe["_rt"] == "research"]
        _fe["_ym"] = _fe["first_hour"].dt.to_period("M")
        _acq_by_m = _fe.groupby("_ym", observed=True).size()
        _eff_y: list[float | None] = []
        _eff_x: list[str] = []
        for _per in _months:
            _c = float(_cost_by_m.get(_per, 0.0))
            _n = int(_acq_by_m.get(_per, 0))
            _eff_x.append(str(_per))
            if _n > 0:
                _eff_y.append(_c / _n)
            else:
                _eff_y.append(float("nan"))

        fig_cac = go.Figure(
            go.Bar(
                x=_eff_x,
                y=_eff_y,
                marker_color="#6366f1",
                hovertemplate="%{x}<br>Cost / research-first user: $%{y:.2f}<extra></extra>",
            )
        )
        fig_cac.update_layout(
            yaxis_title="Research retail USD ÷ research-first users",
            xaxis_title="Month of first activity (UTC)",
            height=420,
            margin=dict(t=24, b=48, l=56, r=24),
            yaxis=dict(gridcolor="rgba(148,163,184,0.2)"),
        )
        st.plotly_chart(fig_cac, use_container_width=True)
        st.caption(
            "**Research-first users**: distinct `user_id` whose **first** hourly row is `research`; month = "
            "that first timestamp (UTC). Efficiency = monthly **research retail cost** ÷ count of those users in the month."
        )

# -----------------------------------------------------------------------------
# Infrastructure & Cost Analysis
# -----------------------------------------------------------------------------
elif page == "Infrastructure & Cost Analysis":
    st.title("Infrastructure & cost analysis")
    st.caption("Charts and tables removed — add content step by step.")
