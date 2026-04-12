"""
Tavily Data Analyst assignment — Streamlit dashboard.
Loads CSVs from data.zip (deploy) or from the same folder / parent folder (local dev).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
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
    st.title("Product analysis")

# -----------------------------------------------------------------------------
# Infrastructure & Cost Analysis
# -----------------------------------------------------------------------------
elif page == "Infrastructure & Cost Analysis":
    st.title("Infrastructure & cost analysis")
    st.caption("Charts and tables removed — add content step by step.")
