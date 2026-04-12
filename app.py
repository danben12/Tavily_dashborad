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
        "Overview: Research User Profile",
        "Product Analysis",
        "Infrastructure & Cost Analysis",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Tavily Data Analyst home assignment — Dan Benbenisti")


# -----------------------------------------------------------------------------
# Overview — Research user cohort (sample; not platform-wide)
# -----------------------------------------------------------------------------
if page == "Overview: Research User Profile":
    st.title("Research User Cohort: Profile & Adoption")
    cohort_ids = _research_cohort_user_ids(df_research)
    cohort_set = set(int(x) for x in cohort_ids.tolist())
    n_cohort = int(len(cohort_ids))

    # --- KPIs ---
    # 1) Cohort size (above as n_cohort)
    # 2) Research-led acquisition: first chronological hourly row is research
    h_all = df_hourly.dropna(subset=["hour", "user_id"]).copy()
    h_all["user_id"] = h_all["user_id"].astype(int)
    h_cohort = h_all[h_all["user_id"].isin(cohort_set)].sort_values(["user_id", "hour"])
    first_usage = h_cohort.groupby("user_id", sort=False).first() if not h_cohort.empty else pd.DataFrame()
    n_with_hourly = len(first_usage)
    n_research_led = 0
    if n_with_hourly and "request_type" in first_usage.columns:
        rt0 = first_usage["request_type"].fillna("").astype(str).str.lower().str.strip()
        n_research_led = int((rt0 == "research").sum())
    pct_research_led = (100.0 * n_research_led / n_with_hourly) if n_with_hourly else 0.0

    # 3) Monetization (cohort users present in users table)
    uc = df_users_unique[df_users_unique["user_id"].isin(cohort_set)].copy()
    n_users_joined = len(uc)
    n_monetized = 0
    if n_users_joined and {"plan", "has_paygo"}.issubset(uc.columns):
        n_monetized = int(
            uc.apply(lambda r: _user_monetized_row(r["plan"], r["has_paygo"]), axis=1).sum()
        )
    pct_monetized = (100.0 * n_monetized / n_users_joined) if n_users_joined else 0.0

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Research cohort size", f"{n_cohort:,}", help="Distinct user_id in research_requests.")
    with k2:
        st.metric(
            "Research-led acquisition",
            f"{pct_research_led:.1f}%",
            help="Share of cohort users with hourly usage whose **first** recorded request (by time) is `research`.",
        )
    with k3:
        st.metric(
            "Cohort monetization",
            f"{pct_monetized:.1f}%",
            help="Share of cohort users in users.csv on a non-researcher plan or with Pay-as-you-go.",
        )

    if n_with_hourly < n_cohort:
        st.caption(
            f"Research-led rate uses **{n_with_hourly:,} / {n_cohort:,}** cohort users who appear in hourly_usage "
            "(users with no hourly rows are excluded from this percentage)."
        )
    if n_users_joined < n_cohort:
        st.caption(
            f"Monetization uses **{n_users_joined:,} / {n_cohort:,}** cohort users matched in users.csv."
        )

    st.markdown("---")

    # Row 1: 7-day MA traffic + cumulative signups
    r2a, r2b = st.columns(2)
    with r2a:
        st.markdown("### Traffic trend (7-day moving average)")
        st.caption("Daily total requests from hourly_usage (cohort); line is a 7-day moving average.")
        if h_cohort.empty:
            st.info("No hourly data for cohort.")
        else:
            hc = h_cohort.copy()
            hc["_day"] = pd.to_datetime(hc["hour"], utc=True, errors="coerce").dt.normalize()
            hc["_day"] = hc["_day"].dt.tz_localize(None)
            daily = hc.groupby(hc["_day"], observed=True)["request_count"].sum().sort_index()
            ma7 = daily.rolling(window=7, min_periods=1).mean()
            fig_ma = go.Figure()
            fig_ma.add_trace(
                go.Scatter(
                    x=daily.index,
                    y=daily.values,
                    mode="lines",
                    name="Daily total",
                    line=dict(color="#94a3b8", width=1),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} requests<extra></extra>",
                )
            )
            fig_ma.add_trace(
                go.Scatter(
                    x=ma7.index,
                    y=ma7.values,
                    mode="lines",
                    name="7-day MA",
                    line=dict(color="#2563eb", width=2.5),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} requests (MA)<extra></extra>",
                )
            )
            fig_ma.update_layout(
                xaxis_title="Date (UTC, naive)",
                yaxis_title="Requests",
                height=420,
                margin=dict(t=24, b=48, l=56, r=24),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
            )
            st.plotly_chart(fig_ma, use_container_width=True)

    with r2b:
        st.markdown("### Cohort growth (by signup date)")
        st.caption("Cumulative distinct cohort users by `created_at` (users.csv).")
        if uc.empty or "created_at" not in uc.columns:
            st.info("No `created_at` for matched cohort users.")
        else:
            cr = pd.to_datetime(uc["created_at"], utc=True, errors="coerce").dropna()
            cr = cr.dt.normalize().dt.tz_localize(None)
            daily_u = cr.value_counts().sort_index()
            cum_u = daily_u.cumsum()
            fig_area = go.Figure(
                go.Scatter(
                    x=cum_u.index,
                    y=cum_u.values,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color="#059669", width=2),
                    fillcolor="rgba(5, 150, 105, 0.2)",
                    name="Cumulative users",
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,} users<extra></extra>",
                )
            )
            fig_area.update_layout(
                xaxis_title="Signup date (UTC, naive)",
                yaxis_title="Cumulative users",
                height=420,
                margin=dict(t=24, b=48, l=56, r=24),
                showlegend=False,
                xaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
            )
            st.plotly_chart(fig_area, use_container_width=True)

    st.success(
        "**Cohort Insight:** The Research API acts as a strong acquisition magnet, with over 18% of this cohort "
        "joining Tavily specifically to use this feature. While these users are highly active across the ecosystem "
        "(especially in **Query**), they represent a premium segment where **reliability** and **cost-efficiency** are "
        "paramount."
    )

# -----------------------------------------------------------------------------
# Product Analysis — Part 1 (rebuild)
# -----------------------------------------------------------------------------
elif page == "Product Analysis":
    st.title("Product analysis")
    if "user_id" in df_research.columns:
        n_research_users_rq = int(
            df_research.dropna(subset=["user_id"])["user_id"].astype(int).nunique()
        )
    else:
        n_research_users_rq = 0
    st.info(
        f"**About this page:** Everything below is scoped to **{n_research_users_rq:,}** distinct users who appear "
        "in **research_requests** in this sample. The goal is to understand **Research API** adoption, mix, and "
        "economics for that group—not to represent all Tavily traffic or accounts."
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
    _uc_prod = df_users_unique[df_users_unique["user_id"].isin(_cohort_set_prod)].copy()
    _n_users_prod = len(_uc_prod)
    _n_mon_prod = 0
    if _n_users_prod and {"plan", "has_paygo"}.issubset(_uc_prod.columns):
        _n_mon_prod = int(
            _uc_prod.apply(lambda r: _user_monetized_row(r["plan"], r["has_paygo"]), axis=1).sum()
        )
    _n_free_prod = max(0, _n_users_prod - _n_mon_prod) if _n_users_prod else 0

    st.markdown("### Monetization (Research API cohort)")
    st.caption(
        "Users who appear in research_requests, matched to users.csv: **Monetized** (non-researcher plan or PayGo) "
        "vs **Fully free**."
    )
    if _n_users_prod == 0:
        st.info("No cohort users matched in users.csv for this chart.")
    else:
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Monetized", "Fully free"],
                    values=[_n_mon_prod, _n_free_prod],
                    hole=0.55,
                    marker=dict(colors=["#0ea5e9", "#cbd5e1"]),
                    textinfo="label+percent",
                    hovertemplate="%{label}<br>%{value:,} users<br>%{percent}<extra></extra>",
                )
            ]
        )
        fig_donut.update_layout(
            showlegend=True,
            height=400,
            margin=dict(t=24, b=24, l=24, r=24),
            legend=dict(font=dict(color="#e2e8f0")),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("### Query is still the Backbone for Research Users")
    st.caption("Total request volume in hourly_usage for the Research API cohort, by request type.")
    _h_prod = df_hourly.dropna(subset=["hour", "user_id"]).copy()
    _h_prod["user_id"] = _h_prod["user_id"].astype(int)
    _h_prod = _h_prod[_h_prod["user_id"].isin(_cohort_set_prod)]
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
            height=400,
            margin=dict(t=24, b=48, l=120, r=24),
            xaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.28)"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    _pareto_pct = _research_pareto_pct_curve(df_research)
    st.markdown("### Research requests vs users (Pareto)")
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
            height=460,
            margin=dict(t=32, b=48, l=56, r=24),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_par, use_container_width=True)

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
