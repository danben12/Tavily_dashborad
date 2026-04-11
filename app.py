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
status_series = df_research["status"].fillna("").astype(str).str.strip()
status_series = status_series.replace("", "(empty)")

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
if page == "Overview":
    st.title("Overview")
    st.caption("High-level view of the sampled user base: scale, plans, and monetization levers.")

    user_ids_in_table = set(df_users["user_id"].dropna().astype(int))
    n_users = len(df_users)
    paygo_users = int(df_users["has_paygo"].sum()) if "has_paygo" in df_users.columns else 0
    paygo_pct = (100.0 * paygo_users / n_users) if n_users else 0.0
    research_distinct = len(research_users)
    research_in_sample = len(research_users & user_ids_in_table)
    research_pct = (100.0 * research_in_sample / n_users) if n_users else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Users in sample", f"{n_users:,}")
    m2.metric(
        "Pay-as-you-go enabled",
        f"{paygo_users:,}",
        f"{paygo_pct:.1f}% of users",
    )
    m3.metric(
        "Users with ≥1 research request",
        f"{research_in_sample:,}",
        f"{research_pct:.1f}% of sample ({research_distinct:,} distinct in research log)",
    )

    st.subheader("Plan segmentation")
    if "plan" in df_users.columns:
        plan_counts = df_users["plan"].fillna("(unknown)").value_counts()
        left, right = st.columns((1, 1))
        with left:
            st.bar_chart(plan_counts)
        with right:
            seg = plan_counts.reset_index()
            seg.columns = ["Plan", "Users"]
            total_p = int(seg["Users"].sum())
            seg["Share"] = ((seg["Users"] / total_p) * 100).round(1).astype(str) + "%"
            st.dataframe(seg, hide_index=True, use_container_width=True)
    else:
        st.info("No `plan` column in users data.")

    st.subheader("Plan limits & PAYGO mix")
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown("**Monthly credit limit (`plan_limit`)**")
        if "plan_limit" in df_users.columns:
            lim = pd.to_numeric(df_users["plan_limit"], errors="coerce").fillna(0).astype(int)
            lim_counts = lim.value_counts().sort_index()
            st.bar_chart(lim_counts)
        else:
            st.caption("Column not present.")
    with c_b:
        st.markdown("**PAYGO flag**")
        if "has_paygo" in df_users.columns:
            paygo_label = np.where(df_users["has_paygo"], "PAYGO on", "PAYGO off")
            st.bar_chart(pd.Series(paygo_label).value_counts())
        else:
            st.caption("Column not present.")

    if "created_at" in df_users.columns and df_users["created_at"].notna().any():
        st.subheader("Account age (snapshot)")
        anchor = df_users["created_at"].max()
        valid_created = df_users["created_at"].notna()
        age_days = (anchor - df_users.loc[valid_created, "created_at"]).dt.days.clip(lower=0)
        st.caption(f"Days since signup, relative to newest account in sample ({anchor.date()}).")
        age_binned = pd.cut(
            age_days,
            bins=[0, 30, 90, 180, 365, float("inf")],
            labels=["0–30d", "31–90d", "91–180d", "181–365d", "365d+"],
            right=True,
            include_lowest=True,
        )
        bucket_order = ["0–30d", "31–90d", "91–180d", "181–365d", "365d+"]
        age_counts = age_binned.astype(str).value_counts().reindex(bucket_order).fillna(0).astype(int)
        st.bar_chart(age_counts)

    with st.expander("Operational snapshot (usage volume, cost, research outcomes)"):
        total_infra_usd = float(df_costs_long["usd"].sum())
        total_hourly_requests = (
            int(df_hourly["request_count"].sum()) if "request_count" in df_hourly.columns else len(df_hourly)
        )
        n_research_req = len(df_research)
        e1, e2, e3 = st.columns(3)
        e1.metric("Infra + model spend (period)", f"${total_infra_usd:,.0f}")
        e2.metric("Request events (hourly log)", f"{total_hourly_requests:,}")
        e3.metric("Research API requests", f"{n_research_req:,}")
        st.subheader("Total cost per hour (all components)")
        hourly_total = df_costs_long.groupby("hour", as_index=False)["usd"].sum()
        hourly_total = hourly_total.sort_values("hour")
        p = figure(
            title="Sum of infrastructure + model columns by hour",
            x_axis_type="datetime",
            height=320,
            sizing_mode="stretch_width",
        )
        p.line(hourly_total["hour"], hourly_total["usd"], line_width=2, color="#2563eb")
        p.varea(x=hourly_total["hour"], y1=0, y2=hourly_total["usd"], fill_color="#93c5fd", fill_alpha=0.4)
        st.bokeh_chart(p, use_container_width=True)
        st.subheader("Research request outcomes")
        st.bar_chart(status_series.value_counts())

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
        u = df_users.copy()
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
