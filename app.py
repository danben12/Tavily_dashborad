"""
Tavily Data Analyst assignment — Streamlit dashboard.
Loads CSVs from data.zip (deploy) or from the same folder / parent folder (local dev).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
import bokeh.palettes as bp

# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tavily Research — Analytics",
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
        source = "data.zip"
    except (FileNotFoundError, KeyError):
        try:
            frames = read_loose(BASE_DIR)
            source = str(BASE_DIR)
        except FileNotFoundError:
            frames = read_loose(PARENT_DIR)
            source = str(PARENT_DIR)

    df_hourly = frames["hourly_usage.csv"].copy()
    df_costs = frames["infrastructure_costs.csv"].copy()
    df_research = frames["research_requests.csv"].copy()
    df_users = frames["users.csv"].copy()

    # Normalize to snake_case for simpler code paths
    df_hourly.columns = df_hourly.columns.str.lower()
    df_costs.columns = df_costs.columns.str.lower()
    df_research.columns = df_research.columns.str.lower()
    df_users.columns = df_users.columns.str.lower()

    # Parse dates
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

    # Numeric cleanup
    for c in ("request_cost", "credits_used", "response_time_seconds"):
        if c in df_research.columns:
            df_research[c] = pd.to_numeric(df_research[c], errors="coerce")
    for c in ("request_count", "total_credits_used", "paygo_credits_used"):
        if c in df_hourly.columns:
            df_hourly[c] = pd.to_numeric(df_hourly[c], errors="coerce")

    bool_cols = ["has_output_schema", "stream"]
    for c in bool_cols:
        if c in df_research.columns:
            df_research[c] = df_research[c].astype(str).str.upper().eq("TRUE")

    if "has_paygo" in df_users.columns:
        df_users["has_paygo"] = df_users["has_paygo"].astype(str).str.lower().eq("true")

    return df_hourly, df_costs, df_research, df_users, source


try:
    df_hourly, df_costs, df_research, df_users, _data_source = load_data()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

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

# -----------------------------------------------------------------------------
# Derived helpers (not cached — cheap vs load)
# -----------------------------------------------------------------------------
cost_value_cols = [
    c
    for c in df_costs.columns
    if c != "hour" and pd.api.types.is_numeric_dtype(df_costs[c])
]
if not cost_value_cols:
    cost_value_cols = [c for c in df_costs.columns if c != "hour"]

df_costs_long = df_costs.melt(
    id_vars=["hour"],
    value_vars=[c for c in cost_value_cols if c in df_costs.columns],
    var_name="component",
    value_name="usd",
)
df_costs_long["usd"] = pd.to_numeric(df_costs_long["usd"], errors="coerce").fillna(0)

research_users = set(df_research["user_id"].dropna().astype(int))

# One row per user_id: user counts and segmentation use unique IDs, not raw row count
_u = df_users.dropna(subset=["user_id"]).copy()
_u["user_id"] = _u["user_id"].astype(int)
_agg: dict = {}
for _col in ("plan", "plan_limit"):
    if _col in _u.columns:
        _agg[_col] = "first"
if "has_paygo" in _u.columns:
    _agg["has_paygo"] = "max"  # True if any duplicate row had PAYGO on
if "created_at" in _u.columns:
    _agg["created_at"] = "min"  # earliest = signup
if _agg:
    df_users_unique = _u.groupby("user_id", as_index=False).agg(_agg)
else:
    df_users_unique = _u.drop_duplicates(subset=["user_id"])

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
if page == "Overview":
    st.title("Overview")
    n_users = int(df_users_unique["user_id"].nunique())
    st.metric("Users (unique user_id)", f"{n_users:,}")

    st.subheader("Plan segmentation")
    if "plan" not in df_users_unique.columns:
        st.info("No `plan` column in users data.")
    elif n_users == 0:
        st.info("No users to segment.")
    else:
        u = df_users_unique.copy()
        u["plan"] = u["plan"].fillna("(unknown)").astype(str)
        seg = u.groupby("plan", as_index=False).agg(users=("user_id", "count"))
        seg["% of all users"] = (100.0 * seg["users"] / n_users).map(lambda x: f"{x:.1f}%")
        if "has_paygo" in u.columns:
            paygo = u.groupby("plan")["has_paygo"].mean().reset_index()
            paygo.columns = ["plan", "_paygo_rate"]
            seg = seg.merge(paygo, on="plan", how="left")
            seg["% with PAYGO (within plan)"] = (100.0 * seg["_paygo_rate"]).map(lambda x: f"{x:.1f}%")
            seg = seg.drop(columns=["_paygo_rate"])
        seg = seg.sort_values("users", ascending=False).reset_index(drop=True)
        seg_display = seg.drop(columns=["users"])
        st.dataframe(seg_display, use_container_width=True, hide_index=True)
        st.caption(
            "**% of all users** — share of unique users in each plan (sums to 100%). "
            "**% with PAYGO (within plan)** — among users on that plan only, share with PAYGO enabled."
        )

        if "has_paygo" in u.columns:
            plans_ordered = seg["plan"].tolist()
            # String segments avoid unstack issues if booleans differ by environment (e.g. numpy vs Python bool)
            u["_paygo_seg"] = np.where(u["has_paygo"], "PAYGO on", "Non-PAYGO")
            cnt = u.groupby(["plan", "_paygo_seg"])["user_id"].count().unstack(fill_value=0)
            cnt = cnt.reindex(plans_ordered).fillna(0).astype(int)
            paygo_off = cnt.get("Non-PAYGO", pd.Series(0, index=plans_ordered))
            paygo_on = cnt.get("PAYGO on", pd.Series(0, index=plans_ordered))
            paygo_off = paygo_off.reindex(plans_ordered).fillna(0).astype(int)
            paygo_on = paygo_on.reindex(plans_ordered).fillna(0).astype(int)

            rows: list[dict] = []
            for p in plans_ordered:
                off = int(paygo_off.loc[p])
                on = int(paygo_on.loc[p])
                for segment_name, n_u, o in (
                    ("Non-PAYGO", off, 0),
                    ("PAYGO on", on, 1),
                ):
                    share = (n_u / n_users) if n_users else 0.0
                    pct_all = 100.0 * share
                    rows.append(
                        {
                            "plan": p,
                            "segment": segment_name,
                            "users": n_u,
                            "share_of_all": share,
                            "pct_of_all_users": pct_all,
                            "_ord": o,
                        }
                    )
            df_long = pd.DataFrame(rows)

            color_enc = alt.Color(
                "segment:N",
                title="",
                scale=alt.Scale(
                    domain=["Non-PAYGO", "PAYGO on"],
                    range=["#94a3b8", "#1d4ed8"],
                ),
            )

            # Small multiples: each plan gets a full-height bar so PAYGO mix is readable for tiny plans too
            n_plans = len(plans_ordered)
            facet_cols = min(5, max(1, n_plans))
            mix_chart = (
                alt.Chart(df_long)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "segment:N",
                        title=None,
                        sort=["Non-PAYGO", "PAYGO on"],
                        axis=alt.Axis(labelLimit=200),
                    ),
                    y=alt.Y(
                        "users:Q",
                        title="% within plan",
                        stack="normalize",
                        axis=alt.Axis(format=".0%"),
                    ),
                    color=color_enc,
                    order=alt.Order("_ord:O"),
                    tooltip=[
                        alt.Tooltip("plan:N", title="Plan"),
                        alt.Tooltip("segment:N", title="Segment"),
                        alt.Tooltip("users:Q", title="Users", format=",.0f"),
                        alt.Tooltip("pct_of_all_users:Q", title="% of all users", format=".2f"),
                    ],
                )
                .properties(width=72, height=200)
                .facet(
                    alt.Facet(
                        "plan:N",
                        title=None,
                        sort=plans_ordered,
                        header=alt.Header(labelOrient="bottom", labelPadding=4),
                    ),
                    columns=facet_cols,
                )
            )

            st.markdown("**PAYGO vs non-PAYGO** — one panel per plan (each bar is **100%** of users on that plan).")
            st.altair_chart(mix_chart, use_container_width=True)

            # Log-scaled totals: small plans visible next to dominant researcher plan
            st.markdown("**Users per plan** — horizontal bars, **log** x-axis.")
            seg_size = seg[["plan", "users"]].copy()
            seg_size["users_plot"] = seg_size["users"].clip(lower=1)
            size_chart = (
                alt.Chart(seg_size)
                .mark_bar()
                .encode(
                    y=alt.Y("plan:N", sort="-x", title=None),
                    x=alt.X(
                        "users_plot:Q",
                        title="Users (log scale)",
                        scale=alt.Scale(type="log", nice=False),
                    ),
                    tooltip=[
                        alt.Tooltip("plan:N", title="Plan"),
                        alt.Tooltip("users:Q", title="Users", format=",.0f"),
                    ],
                )
                .properties(height=max(100, min(520, 28 * len(seg_size))))
            )
            st.altair_chart(size_chart, use_container_width=True)

            st.caption(
                "Use the **faceted** chart to compare PAYGO mix when one plan dominates counts. "
                "Use the **log** bar chart to compare absolute plan sizes."
            )

# -----------------------------------------------------------------------------
# Product Analysis (users + research_requests + hourly_usage)
# -----------------------------------------------------------------------------
elif page == "Product Analysis":
    st.title("Product analysis — Research API & platform usage")
    st.markdown(
        "Joins **users**, **research_requests**, and **hourly_usage** (same `user_id` universe in the sample)."
    )

    tab1, tab2, tab3 = st.tabs(["Research requests", "Users & adoption", "Hourly portfolio"])

    with tab1:
        st.subheader("Model tier & client surface")
        c1, c2 = st.columns(2)
        with c1:
            if "model" in df_research.columns:
                st.bar_chart(df_research["model"].fillna("(empty)").value_counts())
        with c2:
            if "client_source" in df_research.columns:
                top = df_research["client_source"].fillna("unknown").value_counts().head(12)
                st.bar_chart(top)

        st.subheader("Latency & intensity (successful vs other)")
        df_r = df_research.copy()
        df_r["outcome_group"] = np.where(
            df_r["status"].fillna("").astype(str).str.lower().eq("success"),
            "success",
            "other / empty",
        )
        agg = df_r.groupby("outcome_group").agg(
            n=("request_id", "count"),
            median_latency_s=("response_time_seconds", "median"),
            median_tokens=("total_tokens", "median"),
            median_searches=("search_calls", "median"),
        )
        show = agg.copy()
        show["median_latency_s"] = show["median_latency_s"].round(1)
        show["median_tokens"] = show["median_tokens"].round(0)
        show["median_searches"] = show["median_searches"].round(1)
        st.dataframe(show)

        st.subheader("Structured output & streaming")
        if "has_output_schema" in df_research.columns and "stream" in df_research.columns:
            feat = df_research.groupby(["has_output_schema", "stream"]).size().reset_index(name="n")
            st.dataframe(feat)

    with tab2:
        st.subheader("Pay-as-you-go vs research engagement")
        u = df_users_unique.copy()
        u["did_research"] = u["user_id"].isin(research_users)
        paygo = u.groupby("has_paygo").agg(users=("user_id", "count"), research_adopters=("did_research", "sum"))
        paygo["share_adopted"] = paygo["research_adopters"] / paygo["users"]
        paygo_show = paygo.copy()
        paygo_show["share_adopted"] = (paygo_show["share_adopted"] * 100).round(1).astype(str) + "%"
        st.dataframe(paygo_show)

        st.subheader("Account age (days) at data extract — research adopters vs not")
        if u["created_at"].notna().any():
            anchor = u["created_at"].max()
            u["account_age_days"] = (anchor - u["created_at"]).dt.days
            box_df = u.assign(segment=np.where(u["did_research"], "research user", "no research row"))
            summary = box_df.groupby("segment")["account_age_days"].describe()[["mean", "50%", "min", "max"]]
            st.dataframe(summary)

    with tab3:
        st.subheader("Request mix in hourly_usage (sampled users)")
        if "request_type" in df_hourly.columns:
            st.bar_chart(df_hourly.groupby("request_type")["request_count"].sum().sort_values(ascending=False))
        if "request_type" in df_hourly.columns and "depth" in df_hourly.columns:
            st.subheader("Depth within type (top combinations)")
            combo = (
                df_hourly.groupby(["request_type", "depth"])["request_count"]
                .sum()
                .reset_index()
                .sort_values("request_count", ascending=False)
                .head(20)
            )
            st.dataframe(combo, use_container_width=True)

# -----------------------------------------------------------------------------
# Infrastructure & Cost Analysis
# -----------------------------------------------------------------------------
elif page == "Infrastructure & Cost Analysis":
    st.title("Infrastructure & model cost analysis")
    st.markdown("From **infrastructure_costs.csv**: hourly USD by component (EKS, data, models, …).")

    component_totals = df_costs_long.groupby("component")["usd"].sum().sort_values(ascending=False)
    top_n = st.slider("Top N components (chart)", 5, 25, 12)
    top_components = component_totals.head(top_n).index.tolist()

    sub = df_costs_long[df_costs_long["component"].isin(top_components)]
    pivot = sub.pivot_table(index="hour", columns="component", values="usd", aggfunc="sum").fillna(0)
    pivot = pivot.sort_index()

    st.subheader(f"Stacked area — top {top_n} components")
    st.caption("Use the legend in the chart menu to show/hide series (Streamlit native chart).")
    st.area_chart(pivot, height=450)

    st.subheader("Total spend by component")
    bar_df = component_totals.head(20).reset_index()
    bar_df.columns = ["component", "usd"]
    factors = bar_df["component"].tolist()
    src = ColumnDataSource(bar_df)
    p2 = figure(
        y_range=factors[::-1],
        height=min(600, 24 * len(factors) + 120),
        sizing_mode="stretch_width",
        title="Top components by total USD",
    )
    cmap = factor_cmap("component", palette=bp.Category20[max(3, len(factors))], factors=factors)
    p2.hbar(y="component", right="usd", height=0.7, source=src, color=cmap, alpha=0.85)
    st.bokeh_chart(p2, use_container_width=True)

    with st.expander("Raw long table (sample)"):
        st.dataframe(df_costs_long.sort_values("hour", ascending=False).head(500))
