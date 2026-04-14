import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

MODEL_COLORS = {"mini": "#72B7B2", "pro": "#E45756"}
MODEL_COLORS_UPPER = {"MINI": "#72B7B2", "PRO": "#E45756"}
USER_COLORS = {"Free Users": "#F58518", "Paying Users": "#4C78A8"}
COOLWARM_SCALE = [
    [0.0, "#3B4CC0"],
    [0.2, "#6F92F3"],
    [0.4, "#AFC7FD"],
    [0.5, "#DDDCDC"],
    [0.6, "#F7B89C"],
    [0.8, "#E7745B"],
    [1.0, "#B40426"],
]
FIRST_REQUEST_TYPE_COLORS = {
    "query": "#4C78A8",
    "research": "#F58518",
    "extract": "#54A24B",
    "crawl": "#E45756",
    "map": "#B279A2",
}


@st.cache_data
def load_datasets_from_zip() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required datasets, preferring data.zip."""
    app_dir = Path(__file__).resolve().parent
    parent_dir = app_dir.parent
    zip_path = app_dir / "data.zip"
    required_files = (
        "hourly_usage.csv",
        "infrastructure_costs.csv",
        "research_requests.csv",
        "users.csv",
    )

    loaded_frames: dict[str, pd.DataFrame] = {}

    if zip_path.is_file():
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = set(zf.namelist())
                missing = [name for name in required_files if name not in members]
                if not missing:
                    for name in required_files:
                        with zf.open(name) as dataset_file:
                            loaded_frames[name] = pd.read_csv(dataset_file)
        except zipfile.BadZipFile:
            loaded_frames = {}

    if not loaded_frames:
        for name in required_files:
            candidate_paths = (app_dir / name, parent_dir / name)
            csv_path = next((path for path in candidate_paths if path.is_file()), None)
            if csv_path is None:
                raise FileNotFoundError(
                    f"Could not load '{name}' from data.zip or CSV fallbacks."
                )
            loaded_frames[name] = pd.read_csv(csv_path)

    return (
        loaded_frames["hourly_usage.csv"],
        loaded_frames["infrastructure_costs.csv"],
        loaded_frames["research_requests.csv"],
        loaded_frames["users.csv"],
    )


def _lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.str.lower()
    return out


def _build_hourly_lifecycle(users: pd.DataFrame, hourly_usage: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    users_l = _lowercase_columns(users)
    hourly_l = _lowercase_columns(hourly_usage)
    required_users = {"user_id", "created_at"}
    required_hourly = {"user_id", "hour", "request_type"}
    if not required_users.issubset(users_l.columns) or not required_hourly.issubset(hourly_l.columns):
        return pd.DataFrame(), 0

    users_l = users_l[["user_id", "created_at"]].copy()
    users_l["created_at"] = pd.to_datetime(users_l["created_at"], errors="coerce", utc=True)
    users_l["user_id"] = pd.to_numeric(users_l["user_id"], errors="coerce")
    users_l = users_l.dropna(subset=["user_id", "created_at"]).copy()
    users_l["user_id"] = users_l["user_id"].astype(int)

    nov_start = pd.Timestamp("2025-11-01", tz="UTC")
    new_users = users_l.loc[users_l["created_at"] >= nov_start, ["user_id", "created_at"]].drop_duplicates(
        subset=["user_id"]
    )

    hourly_l = hourly_l[["user_id", "hour", "request_type"]].copy()
    hourly_l["hour"] = pd.to_datetime(hourly_l["hour"], errors="coerce", utc=True)
    hourly_l["user_id"] = pd.to_numeric(hourly_l["user_id"], errors="coerce")
    hourly_l = hourly_l.dropna(subset=["user_id", "hour"]).copy()
    hourly_l["user_id"] = hourly_l["user_id"].astype(int)

    valid_events = new_users.merge(hourly_l, on="user_id", how="inner")
    valid_events = valid_events.loc[valid_events["hour"] >= valid_events["created_at"]].copy()
    valid_events = valid_events.sort_values(["user_id", "hour"]).reset_index(drop=True)
    if valid_events.empty:
        return pd.DataFrame(), int(new_users["user_id"].nunique())

    first_actions = (
        valid_events.groupby("user_id", as_index=False)
        .first()[["user_id", "hour", "request_type"]]
        .rename(columns={"hour": "first_event_ts", "request_type": "first_source"})
    )
    last_actions = (
        valid_events.groupby("user_id", as_index=False)
        .last()[["user_id", "hour"]]
        .rename(columns={"hour": "last_event_ts"})
    )
    lifecycle = first_actions.merge(last_actions, on="user_id", how="inner")
    lifecycle["first_source"] = lifecycle["first_source"].astype(str).str.strip().str.lower()
    usage_row_counts = valid_events.groupby("user_id").size().reset_index(name="usage_row_count")
    lifecycle = lifecycle.merge(usage_row_counts, on="user_id", how="left")
    lifecycle["usage_row_count"] = lifecycle["usage_row_count"].fillna(0).astype(int)
    lifecycle["single_row_only"] = lifecycle["usage_row_count"].eq(1)
    return lifecycle, int(new_users["user_id"].nunique())


def _single_row_no_return_by_first_request(lifecycle: pd.DataFrame) -> pd.DataFrame:
    """Share of users with exactly one hourly_usage row after signup, by first request_type."""
    if lifecycle.empty:
        return pd.DataFrame(
            columns=[
                "first_source",
                "first_request_label",
                "user_count",
                "single_row_count",
                "pct_single_row",
            ]
        )
    tmp = lifecycle.assign(_single=lifecycle["single_row_only"].astype(bool))
    out = (
        tmp.groupby("first_source", as_index=False)
        .agg(user_count=("user_id", "count"), single_row_count=("_single", "sum"))
        .sort_values("user_count", ascending=False)
        .reset_index(drop=True)
    )
    out["single_row_count"] = out["single_row_count"].astype(int)
    out["pct_single_row"] = 100.0 * out["single_row_count"] / out["user_count"]
    out["first_request_label"] = out["first_source"].astype(str).str.title()
    return out


def _prepare_q2_economics(
    users: pd.DataFrame, research_requests: pd.DataFrame
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    users_l = _lowercase_columns(users)
    rr = _lowercase_columns(research_requests)
    required_users = {"user_id", "has_paygo", "plan"}
    required_rr = {"user_id", "model", "request_cost"}
    if not required_users.issubset(users_l.columns) or not required_rr.issubset(rr.columns):
        return None

    users_l = users_l.copy()
    users_l["user_id"] = pd.to_numeric(users_l["user_id"], errors="coerce")
    users_l = users_l.dropna(subset=["user_id"]).copy()
    users_l["user_id"] = users_l["user_id"].astype(int)
    users_l["has_paygo_bool"] = (
        users_l["has_paygo"].astype(str).str.strip().str.lower().eq("true")
    )
    users_l["plan_norm"] = users_l["plan"].astype(str).str.strip().str.lower()
    users_l["is_paying_user"] = users_l["has_paygo_bool"] | (~users_l["plan_norm"].eq("researcher"))
    users_l = users_l.drop_duplicates(subset=["user_id"], keep="first")
    users_l["user_type"] = users_l["is_paying_user"].map(
        {True: "Paying Users", False: "Free Users"}
    )

    rr = rr.copy()
    rr["user_id"] = pd.to_numeric(rr["user_id"], errors="coerce")
    rr["request_cost"] = pd.to_numeric(rr["request_cost"], errors="coerce")
    rr = rr.dropna(subset=["user_id", "request_cost"]).copy()
    rr["user_id"] = rr["user_id"].astype(int)
    rr["model"] = rr["model"].astype(str).str.strip().str.lower()

    merged = rr.merge(
        users_l[["user_id", "is_paying_user", "user_type"]], on="user_id", how="left"
    )
    merged["is_paying_user"] = merged["is_paying_user"].fillna(False).astype(bool)
    merged["user_type"] = merged["user_type"].fillna("Free Users")

    free_pro = merged[
        (~merged["is_paying_user"]) & (merged["model"] == "pro")
    ].copy()
    total_pro_cost_free = float(free_pro["request_cost"].sum())
    pro_requests_free_count = int(len(free_pro))
    mini_avg_cost = float(
        rr.loc[rr["model"] == "mini", "request_cost"].mean()
    ) if (rr["model"] == "mini").any() else 0.0
    hypothetical_mini_cost = float(pro_requests_free_count * mini_avg_cost)
    potential_savings = float(total_pro_cost_free - hypothetical_mini_cost)

    user_dist = (
        users_l.drop_duplicates(subset=["user_id"])
        .groupby("user_type", as_index=False)["user_id"]
        .nunique()
        .rename(columns={"user_id": "users"})
    )
    request_cost_dist = rr[rr["model"].isin(["mini", "pro"])][["model", "request_cost"]].copy()
    request_cost_dist["model"] = request_cost_dist["model"].str.upper()
    cost_by_model_user = (
        merged.groupby(["model", "user_type"], as_index=False)["request_cost"].sum()
    )
    cost_by_model_user = cost_by_model_user[cost_by_model_user["model"].isin(["mini", "pro"])]

    metrics = {
        "total_pro_cost_free": total_pro_cost_free,
        "potential_savings": potential_savings,
    }
    return metrics, user_dist, request_cost_dist, cost_by_model_user


def _prepare_latency_points(research_requests: pd.DataFrame) -> pd.DataFrame | None:
    rr = _lowercase_columns(research_requests)
    if not {"model", "response_time_seconds"}.issubset(rr.columns):
        return None
    rr["model"] = rr["model"].astype(str).str.lower().str.strip()
    rr["response_time_seconds"] = pd.to_numeric(rr["response_time_seconds"], errors="coerce")
    return rr[rr["model"].isin(["mini", "pro"])].dropna(subset=["response_time_seconds"])


def _prepare_pareto(research_requests: pd.DataFrame) -> tuple[pd.DataFrame, float] | None:
    rr = _lowercase_columns(research_requests)
    if "user_id" not in rr.columns:
        return None
    rr["user_id"] = pd.to_numeric(rr["user_id"], errors="coerce")
    rr = rr.dropna(subset=["user_id"]).copy()
    rr["user_id"] = rr["user_id"].astype(int)
    counts = rr.groupby("user_id").size().sort_values(ascending=False)
    if counts.empty:
        return None

    pareto = pd.DataFrame({"requests": counts.values})
    pareto["cum_requests_pct"] = 100.0 * pareto["requests"].cumsum() / pareto["requests"].sum()
    pareto["cum_users_pct"] = 100.0 * (pareto.index + 1) / len(pareto)
    pareto = pd.concat(
        [
            pd.DataFrame({"cum_users_pct": [0.0], "cum_requests_pct": [0.0]}),
            pareto[["cum_users_pct", "cum_requests_pct"]],
        ],
        ignore_index=True,
    )
    y_at_5 = float(
        pareto.loc[pareto["cum_users_pct"] >= 5.0, "cum_requests_pct"].head(1).fillna(0.0).iloc[0]
    )
    return pareto, y_at_5


def _is_true_stream(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _is_cancelled_status(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.str.contains("cancel", na=False)


def _format_compact_cost(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:.1f}"


def _prepare_q3_top_metrics(research_requests: pd.DataFrame) -> tuple[float, float] | None:
    rr = _lowercase_columns(research_requests)
    required_cols = {
        "status",
        "stream",
        "response_time_seconds",
        "credits_used",
        "request_cost",
    }
    if not required_cols.issubset(rr.columns):
        return None

    rr = rr.copy()
    rr["response_time_seconds"] = pd.to_numeric(rr["response_time_seconds"], errors="coerce")
    rr["credits_used"] = pd.to_numeric(rr["credits_used"], errors="coerce")
    rr["request_cost"] = pd.to_numeric(rr["request_cost"], errors="coerce")
    rr["is_cancelled"] = _is_cancelled_status(rr["status"])
    human_ui = rr[_is_true_stream(rr["stream"])].copy()
    human_ui = human_ui.dropna(subset=["response_time_seconds"])
    human_ui["duration_group"] = human_ui["response_time_seconds"].apply(
        lambda x: "< 90 seconds" if x < 90 else ">= 90 seconds"
    )

    cancel_rate_gt90 = human_ui.loc[
        human_ui["duration_group"] == ">= 90 seconds", "is_cancelled"
    ].mean()
    if pd.isna(cancel_rate_gt90):
        cancel_rate_gt90 = 0.0

    unbilled_cancelled_cost = rr.loc[
        rr["is_cancelled"] & (rr["credits_used"].fillna(0).eq(0)),
        "request_cost",
    ].sum()
    return float(cancel_rate_gt90), float(unbilled_cancelled_cost)


def _render_q3_cancellation_section(research_requests: pd.DataFrame) -> None:
    rr = _lowercase_columns(research_requests)
    required_cols = {
        "status",
        "stream",
        "response_time_seconds",
        "llm_calls",
        "num_sources",
        "credits_used",
        "request_cost",
    }
    if not required_cols.issubset(rr.columns):
        st.warning("Missing required columns for Q3 cancellation analysis.")
        return

    rr = rr.copy()
    rr["response_time_seconds"] = pd.to_numeric(rr["response_time_seconds"], errors="coerce")
    rr["llm_calls"] = pd.to_numeric(rr["llm_calls"], errors="coerce")
    rr["num_sources"] = pd.to_numeric(rr["num_sources"], errors="coerce")
    rr["credits_used"] = pd.to_numeric(rr["credits_used"], errors="coerce")
    rr["request_cost"] = pd.to_numeric(rr["request_cost"], errors="coerce")
    rr["is_cancelled"] = _is_cancelled_status(rr["status"])
    human_ui = rr[_is_true_stream(rr["stream"])].copy()
    human_ui = human_ui.dropna(subset=["response_time_seconds"])
    human_ui["duration_group"] = human_ui["response_time_seconds"].apply(
        lambda x: "< 90 seconds" if x < 90 else ">= 90 seconds"
    )

    wait_base = human_ui.copy()
    wait_effect = (
        wait_base.groupby("duration_group", as_index=False)["is_cancelled"]
        .mean()
        .rename(columns={"is_cancelled": "cancel_rate"})
    )
    wait_counts = (
        wait_base.groupby("duration_group", as_index=False)
        .agg(
            request_count=("is_cancelled", "size"),
            cancelled_count=("is_cancelled", "sum"),
        )
    )
    wait_effect = wait_effect.merge(wait_counts, on="duration_group", how="left")
    wait_effect["duration_group"] = pd.Categorical(
        wait_effect["duration_group"], categories=["< 90 seconds", ">= 90 seconds"], ordered=True
    )
    wait_effect = wait_effect.sort_values("duration_group")

    inefficiency = (
        human_ui.groupby("duration_group", as_index=False).agg(
            **{
                "Median LLM Calls": ("llm_calls", "median"),
                "Median Sources Found": ("num_sources", "median"),
            }
        )
    )
    inefficiency["duration_group"] = pd.Categorical(
        inefficiency["duration_group"], categories=["< 90 seconds", ">= 90 seconds"], ordered=True
    )
    inefficiency = inefficiency.sort_values("duration_group")
    inefficiency_long = inefficiency.melt(
        id_vars="duration_group",
        value_vars=["Median LLM Calls", "Median Sources Found"],
        var_name="metric",
        value_name="value",
    )

    cancelled_only = rr.loc[rr["is_cancelled"]].copy()
    cancelled_only["billing_status"] = cancelled_only["credits_used"].fillna(0).apply(
        lambda x: "Unbilled (0 credits)" if x == 0 else "Billed (>0 credits)"
    )
    billing_dist = (
        cancelled_only.groupby("billing_status", as_index=False)
        .size()
        .rename(columns={"size": "requests"})
    )

    col_left, col_right = st.columns(2)
    with col_left:
        fig_wait = px.bar(
            wait_effect,
            x="duration_group",
            y="cancel_rate",
            title="<b>Cancellation Rate by Wait Time</b>",
            labels={"duration_group": "Duration Group", "cancel_rate": "Cancel Rate"},
            color="duration_group",
            color_discrete_sequence=["#4C78A8", "#E45756"],
            text=wait_effect["cancel_rate"].map(lambda v: f"{100.0 * v:.2f}%"),
            custom_data=["request_count", "cancelled_count"],
        )
        fig_wait.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "Duration: %{x}<br>"
                "Cancel Rate: %{y:.2%}<br>"
                "Total Requests: %{customdata[0]:,.0f}<br>"
                "Cancelled Requests: %{customdata[1]:,.0f}<extra></extra>"
            ),
        )
        fig_wait.update_layout(
            template="simple_white",
            showlegend=False,
            title_font=dict(size=20),
            xaxis_title_font=dict(size=14),
            yaxis_title_font=dict(size=14),
            font=dict(size=13),
        )
        fig_wait.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_wait, use_container_width=True)

    with col_right:
        fig_ineff = px.bar(
            inefficiency_long,
            x="duration_group",
            y="value",
            color="metric",
            barmode="group",
            title="<b>Technical Inefficiency by Wait Time</b>",
            labels={"duration_group": "Duration Group", "value": "Average", "metric": ""},
            color_discrete_map={"Median LLM Calls": "#E45756", "Median Sources Found": "#72B7B2"},
            text=inefficiency_long["value"].map(lambda v: f"{v:.2f}"),
        )
        fig_ineff.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Duration: %{x}<br>%{fullData.name}: %{y:.2f}<extra></extra>",
        )
        fig_ineff.update_layout(
            template="simple_white",
            title_font=dict(size=20),
            xaxis_title_font=dict(size=14),
            yaxis_title_font=dict(size=14),
            font=dict(size=13),
            legend_title_text="",
        )
        st.plotly_chart(fig_ineff, use_container_width=True)

    fig_billing = px.pie(
        billing_dist,
        names="billing_status",
        values="requests",
        hole=0.5,
        title="<b>Billing Status of Cancelled Requests</b>",
        color="billing_status",
        color_discrete_map={"Unbilled (0 credits)": "#E45756", "Billed (>0 credits)": "#4C78A8"},
    )
    fig_billing.update_traces(
        hovertemplate="%{label}<br>Requests: %{value:,.0f}<br>Share: %{percent:.2%}<extra></extra>"
    )
    fig_billing.update_layout(
        template="simple_white",
        title_font=dict(size=20),
        font=dict(size=13),
        legend_title_text="",
    )
    st.plotly_chart(fig_billing, use_container_width=True)


def _prepare_finops_data(
    infrastructure_costs: pd.DataFrame,
    hourly_usage: pd.DataFrame,
    research_requests: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    infra = _lowercase_columns(infrastructure_costs)
    hourly = _lowercase_columns(hourly_usage)
    rr = _lowercase_columns(research_requests)

    if "hour" not in infra.columns:
        return None
    infra_cols = [c for c in infra.columns if c.startswith("infra_")]
    model_cols = [c for c in infra.columns if c.startswith("model_")]
    if not infra_cols or not model_cols:
        return None

    infra = infra.copy()
    infra["hour"] = pd.to_datetime(infra["hour"], errors="coerce", utc=True)
    infra = infra.dropna(subset=["hour"]).copy()
    for col in infra_cols + model_cols:
        infra[col] = pd.to_numeric(infra[col], errors="coerce").fillna(0.0)

    infra["infra_total_cost"] = infra[infra_cols].sum(axis=1)
    infra["ai_total_cost"] = infra[model_cols].sum(axis=1)
    infra["total_hourly_cost"] = infra["infra_total_cost"] + infra["ai_total_cost"]
    total_hardware_cost = float(infra["infra_total_cost"].sum())
    total_ai_cost = float(infra["ai_total_cost"].sum())

    infra["day"] = infra["hour"].dt.floor("d")
    infra["day_of_week"] = infra["hour"].dt.day_name()
    infra["hour_of_day"] = infra["hour"].dt.hour
    infra_daily = (
        infra.groupby("day", as_index=False)
        .agg(
            infra_total_cost=("infra_total_cost", "sum"),
            total_daily_cost=("total_hourly_cost", "sum"),
        )
    )
    heatmap_data = (
        infra.groupby(["day_of_week", "hour_of_day"], as_index=False)["infra_total_cost"]
        .mean()
        .rename(columns={"infra_total_cost": "mean_infra_cost"})
    )

    if {"hour", "request_count"}.issubset(hourly.columns):
        hourly_r = hourly.copy()
        hourly_r["hour"] = pd.to_datetime(hourly_r["hour"], errors="coerce", utc=True)
        hourly_r["request_count"] = pd.to_numeric(hourly_r["request_count"], errors="coerce").fillna(0.0)
        hourly_r = hourly_r.dropna(subset=["hour"]).copy()
        hourly_r["day"] = hourly_r["hour"].dt.floor("d")
        req_daily = hourly_r.groupby("day", as_index=False)["request_count"].sum()
        req_daily = req_daily.rename(columns={"request_count": "total_requests"})
    elif "timestamp" in rr.columns:
        rr["timestamp"] = pd.to_datetime(rr["timestamp"], errors="coerce", utc=True)
        rr = rr.dropna(subset=["timestamp"]).copy()
        rr["day"] = rr["timestamp"].dt.floor("d")
        req_daily = rr.groupby("day", as_index=False).size().rename(columns={"size": "total_requests"})
    else:
        return None

    daily_agg = infra_daily.merge(req_daily, on="day", how="outer")
    daily_agg["infra_total_cost"] = pd.to_numeric(
        daily_agg["infra_total_cost"], errors="coerce"
    ).fillna(0.0)
    daily_agg["total_daily_cost"] = pd.to_numeric(
        daily_agg["total_daily_cost"], errors="coerce"
    ).fillna(0.0)
    daily_agg["total_requests"] = pd.to_numeric(
        daily_agg["total_requests"], errors="coerce"
    ).fillna(0.0)
    daily_agg = daily_agg.sort_values("day").reset_index(drop=True)
    if not daily_agg.empty:
        last_day = daily_agg["day"].max()
        daily_agg = daily_agg[daily_agg["day"] < last_day].copy()

    daily_agg["month"] = daily_agg["day"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    monthly_agg = (
        daily_agg.groupby("month", as_index=False)[
            ["total_requests", "infra_total_cost", "total_daily_cost"]
        ]
        .sum()
        .sort_values("month")
    )
    monthly_agg["month_label"] = monthly_agg["month"].dt.strftime("%Y-%m")

    dead_days = daily_agg[daily_agg["total_requests"].eq(0)].copy()
    wasted_zero_traffic_cost = float(dead_days["total_daily_cost"].sum())
    dead_days_count = int(len(dead_days))

    metrics = {
        "total_hardware_cost": total_hardware_cost,
        "total_ai_cost": total_ai_cost,
        "wasted_zero_traffic_cost": wasted_zero_traffic_cost,
        "dead_days_count": dead_days_count,
    }
    return metrics, daily_agg, monthly_agg, heatmap_data


def render_product_analysis_and_cost(
    users: pd.DataFrame, hourly_usage: pd.DataFrame, research_requests: pd.DataFrame
) -> None:
    st.title("Research API Product Analysis")

    lifecycle, _joined_users_count = _build_hourly_lifecycle(users, hourly_usage)
    if lifecycle.empty:
        st.error("Could not build lifecycle table from users and hourly usage.")
        return
    research_first = lifecycle["first_source"].eq("research")
    research_first_count = int(research_first.sum())
    active_joined_users_count = int(lifecycle["user_id"].nunique())
    acquisition_pct = (
        100.0 * research_first_count / active_joined_users_count
        if active_joined_users_count > 0
        else 0.0
    )

    q2_data = _prepare_q2_economics(users, research_requests)
    if q2_data is None:
        st.error("Missing required columns for economics analysis.")
        return
    q2_metrics, user_dist, request_cost_dist, cost_by_model_user = q2_data
    rr_cost = _lowercase_columns(research_requests)
    total_request_cost = 0.0
    if "request_cost" in rr_cost.columns:
        total_request_cost = float(
            pd.to_numeric(rr_cost["request_cost"], errors="coerce").fillna(0).sum()
        )

    m1, m2 = st.columns(2)
    with m1:
        st.metric(
            "Research API users acquisition precentage",
            f"{acquisition_pct:.2f}%",
            help=(
                "Out of 16,324 total users, 12,895 joined on/after Nov 1, 2025. "
                "Among them, 12,006 had at least one activity after their join date. "
                "Out of those active new users, 2,270 used the Research API as their first activity."
            ),
        )
    with m2:
        st.metric("Total Request Cost", _format_compact_cost(total_request_cost))

    col1, col2 = st.columns(2)

    with col1:
        no_return_df = _single_row_no_return_by_first_request(lifecycle)
        if no_return_df.empty:
            st.warning("Not enough data for single-row usage by first request type.")
        else:
            label_order = no_return_df["first_request_label"].tolist()
            fig_retention = px.bar(
                no_return_df,
                x="first_request_label",
                y="pct_single_row",
                title="<b>No Further Logged Usage After First Action</b>",
                labels={
                    "first_request_label": "First request type",
                    "pct_single_row": "Users with only 1 usage row (%)",
                },
                text=no_return_df["pct_single_row"].map(lambda x: f"{x:.2f}%"),
                color="first_source",
                color_discrete_map=FIRST_REQUEST_TYPE_COLORS,
                category_orders={"first_request_label": label_order},
                custom_data=["user_count", "single_row_count"],
            )
            fig_retention.update_traces(
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "%{x}<br>"
                    "Share with 1 row only: %{y:.2f}%<br>"
                    "Users in segment: %{customdata[0]:,.0f}<br>"
                    "Users with 1 row: %{customdata[1]:,.0f}<extra></extra>"
                ),
            )
            fig_retention.update_layout(
                template="simple_white",
                showlegend=False,
                title_font=dict(size=20),
                xaxis_title_font=dict(size=14),
                yaxis_title_font=dict(size=14),
                font=dict(size=13),
                margin=dict(t=60, b=40, l=30, r=30),
                yaxis=dict(range=[0, 30]),
            )
            st.plotly_chart(fig_retention, use_container_width=True)

    with col2:
        latency_points = _prepare_latency_points(research_requests)
        if latency_points is None:
            st.warning("Missing `model` or `response_time_seconds` in research data.")
        else:
            if latency_points.empty:
                st.warning("No usable Mini/Pro response-time data found.")
            else:
                fig_latency = px.box(
                    latency_points,
                    x="model",
                    y="response_time_seconds",
                    title="<b>Response Time Distribution by Model (Mini vs Pro)</b>",
                    labels={
                        "response_time_seconds": "Response Time (seconds)",
                        "model": "Model",
                    },
                    points=False,
                    color="model",
                    color_discrete_map=MODEL_COLORS,
                )
                fig_latency.update_layout(
                    template="simple_white",
                    title_font=dict(size=20),
                    xaxis_title_font=dict(size=14),
                    yaxis_title_font=dict(size=14),
                    font=dict(size=13),
                    legend_title_text="",
                    margin=dict(t=60, b=40, l=30, r=30),
                )
                fig_latency.update_traces(
                    hovertemplate=(
                        "Model: %{x}<br>"
                        "Q1: %{q1:.2f} sec<br>"
                        "Median: %{median:.2f} sec<br>"
                        "Q3: %{q3:.2f} sec<br>"
                        "Min: %{lowerfence:.2f} sec<br>"
                        "Max: %{upperfence:.2f} sec<extra></extra>"
                    )
                )
                st.plotly_chart(fig_latency, use_container_width=True)

    # Pareto: concentration of research traffic.
    rr_cols = _lowercase_columns(research_requests)
    if "user_id" not in rr_cols.columns:
        st.warning("Missing `user_id` in research requests for Pareto chart.")
        return
    pareto_data = _prepare_pareto(research_requests)
    if pareto_data is None:
        st.warning("No research requests available for Pareto chart.")
        return
    pareto, y_at_5 = pareto_data
    fig_pareto = px.line(
        pareto,
        x="cum_users_pct",
        y="cum_requests_pct",
        title="<b>Research API Traffic Concentration (Pareto Curve)</b>",
        labels={
            "cum_users_pct": "Cumulative % of Users",
            "cum_requests_pct": "Cumulative % of Total Requests",
        },
    )
    fig_pareto.add_trace(
        go.Scatter(
            x=pareto["cum_users_pct"],
            y=pareto["cum_users_pct"],
            mode="lines",
            name="Linear baseline",
            line=dict(color="#FF7F0E", width=2, dash="dot"),
        )
    )
    fig_pareto.add_vline(x=5.0, line_dash="dash", line_color="gray")
    fig_pareto.add_hline(y=y_at_5, line_dash="dash", line_color="gray")
    fig_pareto.update_traces(
        selector=dict(type="scatter", mode="lines"),
        line=dict(color="#0057D9", width=3),
        fill="tozeroy",
        fillcolor="rgba(0,87,217,0.30)",
    )
    fig_pareto.update_traces(
        selector=dict(name="Linear baseline"),
        line=dict(color="#FF7F0E", width=2, dash="dot"),
        fill=None,
    )
    if len(fig_pareto.data) >= 1:
        fig_pareto.data[0].hovertemplate = "Users: %{x:.2f}%<br>Requests: %{y:.2f}%<extra></extra>"
    if len(fig_pareto.data) >= 2:
        fig_pareto.data[1].hovertemplate = "Users: %{x:.2f}%<br>Linear: %{y:.2f}%<extra></extra>"
    fig_pareto.update_layout(
        template="simple_white",
        title_font=dict(size=20),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        font=dict(size=13),
        margin=dict(t=60, b=40, l=30, r=30),
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig_user_dist = px.pie(
            user_dist,
            values="users",
            names="user_type",
            hole=0.5,
            title="<b>User Base: Free vs. Paying</b>",
            color="user_type",
            color_discrete_map=USER_COLORS,
        )
        fig_user_dist.update_layout(
            template="simple_white",
            title_font=dict(size=20),
            font=dict(size=13),
            legend_title_text="",
        )
        fig_user_dist.update_traces(
            hovertemplate="%{label}: %{value:,.2f} users<br>Share: %{percent:.2%}<extra></extra>"
        )
        st.plotly_chart(fig_user_dist, use_container_width=True)

    with col4:
        fig_avg_cost = px.box(
            request_cost_dist,
            x="model",
            y="request_cost",
            title="<b>Request Cost Distribution by Model</b>",
            labels={"model": "Model", "request_cost": "Average Request Cost ($)"},
            color="model",
            color_discrete_map=MODEL_COLORS_UPPER,
            points=False,
        )
        fig_avg_cost.update_layout(
            template="simple_white",
            showlegend=False,
            title_font=dict(size=20),
            xaxis_title_font=dict(size=14),
            yaxis_title_font=dict(size=14),
            font=dict(size=13),
        )
        fig_avg_cost.update_traces(
            hovertemplate="Model: %{x}<br>Request Cost: $%{y:,.2f}<extra></extra>"
        )
        fig_avg_cost.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_avg_cost, use_container_width=True)

    fig_stacked = px.bar(
        cost_by_model_user,
        x="model",
        y="request_cost",
        color="user_type",
        barmode="stack",
        title="<b>Total Infrastructure Cost by Model and User Type</b>",
        labels={
            "model": "Model",
            "request_cost": "Total Request Cost ($)",
            "user_type": "User Type",
        },
        color_discrete_map=USER_COLORS,
    )
    fig_stacked.update_layout(
        template="simple_white",
        title_font=dict(size=20),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        font=dict(size=13),
        legend_title_text="",
    )
    fig_stacked.update_traces(
        hovertemplate="Model: %{x}<br>User Type: %{fullData.name}<br>Total Cost: $%{y:,.2f}<extra></extra>"
    )
    fig_stacked.update_yaxes(tickprefix="$")
    st.plotly_chart(fig_stacked, use_container_width=True)

    _render_q3_cancellation_section(research_requests)


def render_infrastructure_and_cost_analysis(
    infrastructure_costs: pd.DataFrame,
    hourly_usage: pd.DataFrame,
    research_requests: pd.DataFrame,
) -> None:
    st.header("Part 2: Infrastructure & FinOps - The AI Illusion")

    prepared = _prepare_finops_data(infrastructure_costs, hourly_usage, research_requests)
    if prepared is None:
        st.error("Missing required fields for Infrastructure & FinOps analysis.")
        return
    finops_metrics, daily_agg, monthly_agg, heatmap_data = prepared

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Total Hardware Cost", f"${finops_metrics['total_hardware_cost']:,.2f}")
    with k2:
        st.metric("Total AI/Model Cost", f"${finops_metrics['total_ai_cost']:,.2f}")
    with k3:
        st.metric(
            "Wasted Zero-Traffic Cost",
            f"${finops_metrics['wasted_zero_traffic_cost']:,.2f}",
            delta=f"{finops_metrics['dead_days_count']:,} dead days",
        )

    col1, col2 = st.columns(2)
    with col1:
        budget_split = pd.DataFrame(
            {
                "category": ["Hardware & Infrastructure", "AI & LLM Tokens"],
                "cost": [
                    finops_metrics["total_hardware_cost"],
                    finops_metrics["total_ai_cost"],
                ],
            }
        )
        fig_donut = px.pie(
            budget_split,
            names="category",
            values="cost",
            hole=0.5,
            title="<b>Budget Split: Infrastructure vs Model Cost</b>",
            color="category",
            color_discrete_map={
                "Hardware & Infrastructure": "#4C78A8",
                "AI & LLM Tokens": "#E45756",
            },
        )
        fig_donut.update_traces(
            hovertemplate="%{label}<br>Cost: $%{value:,.2f}<br>Share: %{percent:.2%}<extra></extra>"
        )
        fig_donut.update_layout(
            template="simple_white",
            title_font=dict(size=20),
            font=dict(size=13),
            legend_title_text="",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        growth_daily = daily_agg.sort_values("day").copy()
        growth_daily["requests_ma7"] = (
            growth_daily["total_requests"].rolling(window=7, min_periods=1).mean()
        )
        growth_daily["total_cost_ma7"] = (
            growth_daily["total_daily_cost"].rolling(window=7, min_periods=1).mean()
        )
        fig_growth = make_subplots(specs=[[{"secondary_y": True}]])
        fig_growth.add_trace(
            go.Scatter(
                x=growth_daily["day"],
                y=growth_daily["requests_ma7"],
                name="Total Requests (7D MA)",
                mode="lines+markers",
                line=dict(color="#4C78A8", width=3),
                hovertemplate="Day: %{x}<br>Requests (7D MA): %{y:,.0f}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig_growth.add_trace(
            go.Scatter(
                x=growth_daily["day"],
                y=growth_daily["total_cost_ma7"],
                name="Total Cost (7D MA)",
                mode="lines+markers",
                line=dict(color="#E45756", width=3),
                hovertemplate="Day: %{x}<br>Total Cost (7D MA): $%{y:,.2f}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig_growth.update_layout(
            template="simple_white",
            title="<b>The Growth Paradox: Requests vs Total Cost Over Time</b>",
            title_font=dict(size=20),
            font=dict(size=13),
            legend_title_text="",
            margin=dict(t=70, b=40, l=30, r=30),
        )
        fig_growth.update_xaxes(title_text="Time")
        fig_growth.update_yaxes(title_text="Total Requests", secondary_y=False)
        fig_growth.update_yaxes(
            title_text="Total Daily Cost ($) — Infra + Model",
            tickprefix="$",
            secondary_y=True,
        )
        st.plotly_chart(fig_growth, use_container_width=True)

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    heatmap_data["day_of_week"] = pd.Categorical(
        heatmap_data["day_of_week"], categories=day_order, ordered=True
    )
    heatmap_data["hour_of_day"] = pd.to_numeric(
        heatmap_data["hour_of_day"], errors="coerce"
    ).astype("Int64")
    heatmap_pivot = (
        heatmap_data.pivot_table(
            index="day_of_week",
            columns="hour_of_day",
            values="mean_infra_cost",
            aggfunc="mean",
        )
        .reindex(day_order)
        .reindex(columns=list(range(24)))
    )
    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Mean Infra Cost ($)"),
        title="<b>Mean Infrastructure Cost by Day of Week and Hour</b>",
        color_continuous_scale=COOLWARM_SCALE,
        aspect="auto",
    )
    fig_heatmap.update_traces(
        hovertemplate=(
            "Day: %{y}<br>"
            "Hour: %{x}:00<br>"
            "Mean Infra Cost: $%{z:,.2f}<extra></extra>"
        )
    )
    fig_heatmap.update_layout(
        template="simple_white",
        title_font=dict(size=20),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        font=dict(size=13),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Tavily Dashboard",
        page_icon="📊",
        layout="wide",
    )

    (
        hourly_usage,
        infrastructure_costs,
        research_requests,
        users,
    ) = load_datasets_from_zip()

    page = st.sidebar.radio(
        "Pages",
        ["Product Analysis", "Infrastructure & Cost Analysis"],
    )

    if page == "Product Analysis":
        render_product_analysis_and_cost(users, hourly_usage, research_requests)
    else:
        render_infrastructure_and_cost_analysis(
            infrastructure_costs,
            hourly_usage,
            research_requests,
        )


if __name__ == "__main__":
    main()
