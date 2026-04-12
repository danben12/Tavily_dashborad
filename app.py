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
        "**Methodological Note:** This dashboard presents a Cohort Analysis of the 16,333 users who interacted "
        "with the Research API. The data—including traffic volumes in other products—reflects the profile and "
        "behavior of **Research Users** specifically, and does not represent a macro-view of the entire Tavily "
        "platform."
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

    st.markdown("### Research API: unique users per day")
    st.caption(
        "Count of **distinct user_id** with at least one research request on each calendar day (UTC). "
        "Multiple requests the same day still count as one user."
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
            _daily_u = _rdu.groupby("_day", observed=True)["user_id"].nunique().sort_index()
            fig_dau = go.Figure(
                go.Scatter(
                    x=_daily_u.index,
                    y=_daily_u.values,
                    mode="lines",
                    line=dict(color="#38bdf8", width=2.2),
                    fill="tozeroy",
                    fillcolor="rgba(56, 189, 248, 0.18)",
                    name="Unique users",
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:,} users<extra></extra>",
                )
            )
            fig_dau.update_layout(
                xaxis_title="Date (UTC)",
                yaxis_title="Unique users",
                height=380,
                margin=dict(t=24, b=48, l=56, r=24),
                showlegend=False,
                xaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.25)"),
            )
            st.plotly_chart(fig_dau, use_container_width=True)

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
