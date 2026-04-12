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


def _format_currency_usd(value: float) -> str:
    """USD for UI metrics: ``$140K``, ``$9.7M``, ``-$9.6M`` (no long digit strings)."""
    if value is None or not np.isfinite(value):
        return "—"
    v = float(value)
    neg = v < 0
    a = abs(v)
    if a >= 1_000_000:
        body = f"${a / 1_000_000:.1f}M"
    elif a >= 1_000:
        body = f"${a / 1_000:,.0f}K"
    else:
        body = f"${a:,.0f}"
    return f"-{body}" if neg else body


def _format_currency_compact(value: float) -> str:
    """Same scale as ``_format_currency_usd`` (for chart text overlays)."""
    return _format_currency_usd(value)


def _log_yaxis_money_ticks(positive_values: np.ndarray) -> tuple[list[float], list[str]]:
    """Powers-of-ten tick positions and K/M labels for a log-scaled currency axis."""
    v = positive_values[np.isfinite(positive_values) & (positive_values > 0)]
    if v.size == 0:
        ticks = [1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0, 1_000_000.0]
        return ticks, [_format_currency_usd(float(t)) for t in ticks]
    lo = int(np.floor(np.log10(float(v.min()))))
    hi = int(np.ceil(np.log10(float(v.max()))))
    lo = max(lo, -3)
    hi = min(hi, 12)
    ticks = [10**i for i in range(lo, hi + 1)]
    return ticks, [_format_currency_usd(float(t)) for t in ticks]


def _plotly_log_y_range(y_lo: float, y_hi: float, tick_vals: list[float] | None = None) -> list[float]:
    """Plotly ``layout.yaxis.range`` for ``type='log'`` must be **[log10(min), log10(max)]**, not raw USD."""
    lo_lin = max(float(y_lo) * 0.25, 1e-15)
    hi_lin = max(float(y_hi) * 4.0, lo_lin * 10.0)
    if tick_vals:
        tv = np.asarray(tick_vals, dtype=float)
        lo_lin = min(lo_lin, float(tv.min()) * 0.4)
        hi_lin = max(hi_lin, float(tv.max()) * 2.5)
    return [float(np.log10(lo_lin)), float(np.log10(hi_lin))]


def _log_bar_base(y_lo: float) -> float:
    """Bars on a log y-axis cannot start at 0; use a positive base strictly below the smallest value."""
    y_lo = max(float(y_lo), 1e-15)
    return float(10 ** (np.floor(np.log10(y_lo)) - 1))


# Pricing assumptions (monthly subscription list price, USD). No hardcoded revenue totals.
PAYGO_USD_PER_CREDIT = 0.008
PLAN_MONTHLY_USD: dict[str, float] = {
    "researcher": 0.0,
    "project": 30.0,
    "bootstrap": 100.0,
    "startup": 220.0,
    "growth": 500.0,
    "enterprise": 1000.0,
}


def _plan_norm_value(v: object) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    if isinstance(v, (int, np.integer)) and int(v) == 0:
        return "unknown"
    if isinstance(v, (float, np.floating)) and float(v) == 0.0:
        return "unknown"
    s = str(v).strip().lower()
    if s in ("", "nan", "none", "<na>"):
        return "unknown"
    try:
        if float(s) == 0.0:
            return "unknown"
    except ValueError:
        pass
    return s


def _plan_norm(series: pd.Series) -> pd.Series:
    return series.map(_plan_norm_value)


# Split ``researcher`` into free vs PayGo for plan-level economics charts.
PLAN_BUCKET_FREE = "Simple free tier"
PLAN_BUCKET_RESEARCHER_PAYGO = "Researcher with PayGo"


def _plan_bucket_series(plan_norm: pd.Series, has_paygo: pd.Series) -> pd.Series:
    pn = plan_norm.astype(str).str.lower().str.strip()
    pg = has_paygo.fillna(False).astype(bool)
    b = np.where(
        pn.eq("researcher").to_numpy() & ~pg.to_numpy(),
        PLAN_BUCKET_FREE,
        np.where(
            pn.eq("researcher").to_numpy() & pg.to_numpy(),
            PLAN_BUCKET_RESEARCHER_PAYGO,
            pn.astype(str).to_numpy(),
        ),
    )
    return pd.Series(b, index=plan_norm.index, dtype=object)


def _period_end_utc(period: pd.Period) -> pd.Timestamp:
    ts = period.to_timestamp(how="end")
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _period_start_utc(period: pd.Period) -> pd.Timestamp:
    ts = period.to_timestamp(how="start")
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@st.cache_data
def compute_monthly_financials(
    req_df: pd.DataFrame,
    users_unique: pd.DataFrame,
) -> pd.DataFrame:
    """Per calendar month from request timestamps: cost, PayGo revenue, subscription MRR, total revenue."""
    need_r = {"timestamp", "request_cost", "credits_used", "user_id"}
    need_u = {"user_id", "plan", "has_paygo", "created_at"}
    if req_df.empty or not need_r.issubset(req_df.columns):
        return pd.DataFrame()
    if users_unique.empty or not need_u.issubset(users_unique.columns):
        return pd.DataFrame()

    r = req_df.dropna(subset=["timestamp", "user_id"]).copy()
    r["user_id"] = pd.to_numeric(r["user_id"], errors="coerce")
    r = r.dropna(subset=["user_id"])
    r["user_id"] = r["user_id"].astype(int)
    r["request_cost"] = pd.to_numeric(r["request_cost"], errors="coerce").fillna(0.0)
    r["credits_used"] = pd.to_numeric(r["credits_used"], errors="coerce").fillna(0.0)
    r["period"] = r["timestamp"].dt.to_period("M")

    u = users_unique.dropna(subset=["user_id"]).copy()
    u["user_id"] = u["user_id"].astype(int)
    u["plan_norm"] = _plan_norm(u["plan"])
    u["created_at"] = pd.to_datetime(u["created_at"], utc=True, errors="coerce")

    periods = sorted(r["period"].dropna().unique())
    rows: list[dict] = []
    for per in periods:
        m_start = _period_start_utc(per)
        m_end = _period_end_utc(per)
        slice_r = r[(r["timestamp"] >= m_start) & (r["timestamp"] <= m_end)]
        month_cost = float(slice_r["request_cost"].sum())

        merged = slice_r.merge(u, on="user_id", how="left", suffixes=("", "_usr"))
        paygo_mask = merged["has_paygo"].fillna(False).astype(bool)
        paygo_rev = float(merged.loc[paygo_mask, "credits_used"].sum() * PAYGO_USD_PER_CREDIT)

        active = u[u["created_at"].notna() & (u["created_at"] <= m_end)]
        active = active.assign(
            _price=active["plan_norm"].map(PLAN_MONTHLY_USD).fillna(0.0).astype(float)
        )
        sub_rev = float(active["_price"].sum())

        rows.append(
            {
                "period": per,
                "period_label": per.strftime("%b %Y"),
                "month_cost": month_cost,
                "paygo_revenue": paygo_rev,
                "subscription_revenue": sub_rev,
                "total_revenue": sub_rev + paygo_rev,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data
def compute_plan_cost_and_revenue(
    req_df: pd.DataFrame,
    users_unique: pd.DataFrame,
) -> pd.DataFrame:
    """All-time request cost and revenue (subscription MRR over months + PayGo) by user plan."""
    monthly = compute_monthly_financials(req_df, users_unique)
    need_r = {"user_id", "request_cost", "credits_used"}
    need_u = {"user_id", "plan", "has_paygo", "created_at"}
    if req_df.empty or not need_r.issubset(req_df.columns):
        return pd.DataFrame()
    if users_unique.empty or not need_u.issubset(users_unique.columns):
        return pd.DataFrame()

    r = req_df.dropna(subset=["user_id"]).copy()
    r["user_id"] = pd.to_numeric(r["user_id"], errors="coerce")
    r = r.dropna(subset=["user_id"])
    r["user_id"] = r["user_id"].astype(int)
    r["request_cost"] = pd.to_numeric(r["request_cost"], errors="coerce").fillna(0.0)
    r["credits_used"] = pd.to_numeric(r["credits_used"], errors="coerce").fillna(0.0)

    u = users_unique.dropna(subset=["user_id"]).copy()
    u["user_id"] = u["user_id"].astype(int)
    u["plan_norm"] = _plan_norm(u["plan"])
    u["created_at"] = pd.to_datetime(u["created_at"], utc=True, errors="coerce")

    merged_all = r.merge(u, on="user_id", how="inner", suffixes=("", "_usr"))
    merged_all["plan_bucket"] = _plan_bucket_series(merged_all["plan_norm"], merged_all["has_paygo"])
    cost_by = merged_all.groupby("plan_bucket", observed=True)["request_cost"].sum()

    paygo_m = merged_all.loc[merged_all["has_paygo"].fillna(False).astype(bool)]
    paygo_by = paygo_m.groupby("plan_bucket", observed=True)["credits_used"].sum() * PAYGO_USD_PER_CREDIT

    u = u.assign(plan_bucket=_plan_bucket_series(u["plan_norm"], u["has_paygo"]))

    sub_by = pd.Series(0.0, dtype=float)
    if not monthly.empty:
        for per in monthly["period"].tolist():
            m_end = _period_end_utc(per)
            active = u[u["created_at"].notna() & (u["created_at"] <= m_end)]
            active = active.assign(
                _price=active["plan_norm"].map(PLAN_MONTHLY_USD).fillna(0.0).astype(float)
            )
            part = active.groupby("plan_bucket", observed=True)["_price"].sum()
            sub_by = sub_by.add(part, fill_value=0.0)

    _tier_order = [
        PLAN_BUCKET_FREE,
        PLAN_BUCKET_RESEARCHER_PAYGO,
        "project",
        "bootstrap",
        "startup",
        "growth",
        "enterprise",
        "unknown",
    ]
    raw_plans = set(cost_by.index.astype(str)) | set(sub_by.index.astype(str)) | set(paygo_by.index.astype(str))
    plans = [p for p in _tier_order if p in raw_plans] + sorted(p for p in raw_plans if p not in _tier_order)
    out = pd.DataFrame(
        {
            "plan": plans,
            "total_cost": [float(cost_by.get(p, 0.0)) for p in plans],
            "subscription_revenue": [float(sub_by.get(p, 0.0)) for p in plans],
            "paygo_revenue": [float(paygo_by.get(p, 0.0)) for p in plans],
        }
    )
    out = out.groupby("plan", as_index=False, sort=False).agg(
        total_cost=("total_cost", "sum"),
        subscription_revenue=("subscription_revenue", "sum"),
        paygo_revenue=("paygo_revenue", "sum"),
    )
    out["total_revenue"] = out["subscription_revenue"] + out["paygo_revenue"]
    tier_rank = {p: i for i, p in enumerate(_tier_order)}
    out["_rk"] = out["plan"].map(lambda x: tier_rank.get(x, 999))
    out = out.sort_values(["_rk", "plan"], kind="stable").drop(columns="_rk")
    return out


@st.cache_data
def compute_strictly_free_leakage(req_df: pd.DataFrame, users_unique: pd.DataFrame) -> dict:
    """Strictly free cohort: actual cost, hypothetical all-mini cost, savings, pro-only spend."""
    out: dict = {
        "n_users": 0,
        "n_requests": 0,
        "actual_cost": 0.0,
        "avg_mini_unit_cost": np.nan,
        "projected_all_mini_cost": np.nan,
        "potential_savings": np.nan,
        "wasted_pro_cost": 0.0,
    }
    need = {"user_id", "request_cost", "model", "plan", "has_paygo"}
    if req_df.empty or users_unique.empty:
        return out
    if not {"user_id", "request_cost", "model"}.issubset(req_df.columns):
        return out
    if not {"user_id", "plan", "has_paygo"}.issubset(users_unique.columns):
        return out

    r = req_df.dropna(subset=["user_id"]).copy()
    r["user_id"] = pd.to_numeric(r["user_id"], errors="coerce")
    r = r.dropna(subset=["user_id"])
    r["user_id"] = r["user_id"].astype(int)

    u = users_unique.dropna(subset=["user_id"]).copy()
    u["user_id"] = u["user_id"].astype(int)
    u = u.drop_duplicates(subset=["user_id"], keep="first")

    m = r.merge(u, on="user_id", how="inner", suffixes=("", "_usr"))
    plan_norm = _plan_norm(m["plan"])
    paygo = m["has_paygo"].fillna(False).astype(bool)
    cohort = m.loc[plan_norm.eq("researcher") & ~paygo].copy()
    if cohort.empty:
        return out

    cohort["_cost"] = pd.to_numeric(cohort["request_cost"], errors="coerce").fillna(0.0)
    mod = cohort["model"].astype(str).str.lower().str.strip()
    mini = mod.eq("mini")

    n_req = int(len(cohort))
    n_mini = int(mini.sum())
    cost_mini_sum = float(cohort.loc[mini, "_cost"].sum())
    avg_mini = cost_mini_sum / n_mini if n_mini > 0 else np.nan
    actual = float(cohort["_cost"].sum())
    projected = float(n_req * avg_mini) if np.isfinite(avg_mini) else np.nan
    savings = float(actual - projected) if np.isfinite(projected) else np.nan
    wasted_pro = float(cohort.loc[mod.eq("pro"), "_cost"].sum())

    out.update(
        {
            "n_users": int(cohort["user_id"].nunique()),
            "n_requests": n_req,
            "actual_cost": actual,
            "avg_mini_unit_cost": float(avg_mini) if np.isfinite(avg_mini) else np.nan,
            "projected_all_mini_cost": projected if np.isfinite(projected) else np.nan,
            "potential_savings": savings if np.isfinite(savings) else np.nan,
            "wasted_pro_cost": wasted_pro,
        }
    )
    return out


def render_pricing_model_reference_expander() -> None:
    with st.expander("Pricing model reference"):
        st.markdown(
            """
| Plan | Monthly list price (USD) |
|------|-------------------------|
| Researcher | $0 |
| Project | $30 |
| Bootstrap | $100 |
| Startup | $220 |
| Growth | $500 |
| Enterprise | Custom (we assume **$1,000** / user / month for modeling — very few enterprise seats, so this barely moves totals) |

**Pay-as-you-go:** **$0.008** per credit used (`credits_used` on each request).

Subscription revenue is modeled as the sum of each active subscriber’s monthly list price for every month in which their `created_at` is on or before that month-end (MRR-style snapshot). PayGo revenue is request-level credits × **$0.008** for users with `has_paygo == True`. **Request cost** is the sum of `request_cost` (nominal units as provided in the extract).
            """
        )


def render_product_analytics_dashboard(req_df: pd.DataFrame, users_unique: pd.DataFrame) -> None:
    """Cohesive product report: pricing context, KPIs, trends, plan economics, free-tier leakage."""
    st.caption(
        "Financial performance, unit economics, and operational efficiency for the Research API. "
        "All currency figures are derived from the loaded tables and the pricing rules below — "
        "no hardcoded revenue totals."
    )

    render_pricing_model_reference_expander()

    monthly = compute_monthly_financials(req_df, users_unique)
    if monthly.empty:
        st.warning("Insufficient data to build monthly financials (need timestamps and user attributes).")
        return

    total_rev = float(monthly["total_revenue"].sum())
    total_cost = float(monthly["month_cost"].sum())
    net = total_rev - total_cost

    st.subheader("Executive snapshot")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric(
            "All-time revenue",
            _format_currency_usd(total_rev),
            help="Sum of modeled subscription MRR (per month) plus PayGo revenue over months with research traffic.",
        )
    with k2:
        st.metric(
            "All-time request cost",
            _format_currency_usd(total_cost),
            help="Sum of `request_cost` over the same months.",
        )
    with k3:
        st.metric(
            "Net position (profit / burn)",
            _format_currency_usd(net),
            help="Revenue minus request cost in nominal `request_cost` units.",
        )

    st.subheader("Monthly revenue vs. request cost")
    rev_y = monthly["total_revenue"].astype(float).to_numpy()
    cost_y = monthly["month_cost"].astype(float).to_numpy()
    rev_plot = np.where(rev_y > 0, rev_y, np.nan)
    cost_plot = np.where(cost_y > 0, cost_y, np.nan)
    _pos = np.concatenate([rev_y[rev_y > 0], cost_y[cost_y > 0]])
    tick_vals, tick_text = _log_yaxis_money_ticks(_pos if _pos.size else np.array([1.0], dtype=float))

    _stack = np.concatenate([rev_plot, cost_plot])
    _finite = _stack[np.isfinite(_stack) & (_stack > 0)]
    if _finite.size:
        y_lo, y_hi = float(_finite.min()), float(_finite.max())
    else:
        y_lo, y_hi = 1.0, 10.0
    _bar_base = _log_bar_base(y_lo)

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Bar(
            x=monthly["period_label"],
            y=rev_plot,
            base=_bar_base,
            name="Total revenue",
            marker_color="#42A5F5",
        )
    )
    fig_trend.add_trace(
        go.Bar(
            x=monthly["period_label"],
            y=cost_plot,
            base=_bar_base,
            name="Total request cost",
            marker_color="#EF5350",
        )
    )
    fig_trend.update_layout(
        template="plotly_dark",
        barmode="group",
        height=440,
        margin=dict(t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(type="category", title="Month", categoryorder="array", categoryarray=monthly["period_label"].tolist()),
        yaxis_title="USD (cost/revenue)",
    )
    fig_trend.update_yaxes(
        type="log",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_text,
        range=_plotly_log_y_range(y_lo, y_hi, tick_vals),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Resource allocation and cost controls")
    st.markdown(
        "Compare cumulative **request cost** to modeled revenue **by billing segment**. "
        "**Simple free tier** is `plan == researcher` with `has_paygo == False`; **Researcher with PayGo** is the same plan with PayGo enabled. "
        "Subscription revenue is the sum of monthly list prices for users active through each month; PayGo is allocated to the user’s segment."
    )

    plan_df = compute_plan_cost_and_revenue(req_df, users_unique)
    if not plan_df.empty:
        plan_df = plan_df[
            ~plan_df["plan"].astype(str).str.fullmatch(r"0(\.0+)?", case=False, na=False)
        ].copy()
        plan_df = plan_df[(plan_df["total_cost"] + plan_df["total_revenue"]) > 0].copy()
    if not plan_df.empty:
        x_plans = plan_df["plan"].astype(str)
        cost_y = plan_df["total_cost"].astype(float).to_numpy()
        rev_y = plan_df["total_revenue"].astype(float).to_numpy()
        cost_plot = np.where(cost_y > 0, cost_y, np.nan)
        rev_plot = np.where(rev_y > 0, rev_y, np.nan)
        _pos_p = np.concatenate([cost_y[cost_y > 0], rev_y[rev_y > 0]])
        tick_p, text_p = _log_yaxis_money_ticks(_pos_p if _pos_p.size else np.array([1.0], dtype=float))

        _sp = np.concatenate([cost_plot, rev_plot])
        _fin_p = _sp[np.isfinite(_sp) & (_sp > 0)]
        if _fin_p.size:
            p_lo, p_hi = float(_fin_p.min()), float(_fin_p.max())
        else:
            p_lo, p_hi = 1.0, 10.0
        _plan_bar_base = _log_bar_base(p_lo)

        fig_plan = go.Figure(
            data=[
                go.Bar(
                    name="Total revenue",
                    x=x_plans,
                    y=rev_plot,
                    base=_plan_bar_base,
                    marker_color="#42A5F5",
                ),
                go.Bar(
                    name="Total request cost",
                    x=x_plans,
                    y=cost_plot,
                    base=_plan_bar_base,
                    marker_color="#EF5350",
                ),
            ]
        )
        fig_plan.update_layout(
            barmode="group",
            template="plotly_dark",
            height=460,
            margin=dict(t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                type="category",
                title="User plan",
                categoryorder="array",
                categoryarray=x_plans.tolist(),
            ),
            yaxis_title="USD (cost/revenue)",
        )
        fig_plan.update_yaxes(
            type="log",
            tickmode="array",
            tickvals=tick_p,
            ticktext=text_p,
            range=_plotly_log_y_range(p_lo, p_hi, tick_p),
        )
        st.plotly_chart(fig_plan, use_container_width=True)

    leak = compute_strictly_free_leakage(req_df, users_unique)
    savings_disp = leak["potential_savings"]
    wasted_disp = leak["wasted_pro_cost"]

    st.markdown("**Strictly free cohort** — `plan == researcher` and `has_paygo == False`.")
    k_u, k_r, k_c, k_w, k_s = st.columns(5)
    with k_u:
        st.metric("Users in cohort", f"{leak['n_users']:,}")
    with k_r:
        st.metric("Requests in cohort", f"{leak['n_requests']:,}")
    with k_c:
        st.metric(
            "Cohort request cost",
            _format_currency_usd(leak["actual_cost"]),
            help="Sum of `request_cost` for this cohort.",
        )
    with k_w:
        st.metric(
            "Wasted “pro” cost (cohort)",
            _format_currency_usd(wasted_disp),
            help="Subset of cohort cost attributed to `model == pro`.",
        )
    with k_s:
        st.metric(
            "Projected savings (all requests @ mini unit cost)",
            _format_currency_usd(savings_disp),
            help="Actual cohort cost minus (cohort request count × average `mini` request cost).",
        )

    if not np.isfinite(leak.get("potential_savings", np.nan)):
        st.caption(
            "Projected savings unavailable: the cohort has no **mini** traffic to estimate unit cost."
        )


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
# Product Analysis
# -----------------------------------------------------------------------------
if page == "Product Analysis":
    st.title("Research API — product analytics")
    render_product_analytics_dashboard(df_research, df_users_unique)

# -----------------------------------------------------------------------------
# Infrastructure & Cost Analysis
# -----------------------------------------------------------------------------
elif page == "Infrastructure & Cost Analysis":
    st.title("Infrastructure & cost analysis")
    st.caption("Charts and tables removed — add content step by step.")
