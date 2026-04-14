import math
import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots



# -------------------------
# data loading and utilities
# -------------------------
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
    # Normalize column names early so all downstream checks/use are case-robust.
    users_l = _lowercase_columns(users)
    hourly_l = _lowercase_columns(hourly_usage)

    # Ensure the minimal fields needed for lifecycle construction exist.
    required_users = {"user_id", "created_at"}
    required_hourly = {"user_id", "hour", "request_type"}
    if not required_users.issubset(users_l.columns) or not required_hourly.issubset(hourly_l.columns):
        return pd.DataFrame(), 0

    # Keep only required user columns and coerce types.
    users_l = users_l[["user_id", "created_at"]].copy()
    users_l["created_at"] = pd.to_datetime(users_l["created_at"], errors="coerce", utc=True)
    users_l["user_id"] = pd.to_numeric(users_l["user_id"], errors="coerce")
    # Drop invalid rows before casting to integer ids.
    users_l = users_l.dropna(subset=["user_id", "created_at"]).copy()
    users_l["user_id"] = users_l["user_id"].astype(int)

    # Limit acquisition cohort to users created from product launch window onward.
    nov_start = pd.Timestamp("2025-11-01", tz="UTC")
    new_users = users_l.loc[users_l["created_at"] >= nov_start, ["user_id", "created_at"]].drop_duplicates(
        subset=["user_id"]
    )

    # Prepare hourly usage rows with normalized types.
    hourly_l = hourly_l[["user_id", "hour", "request_type"]].copy()
    hourly_l["hour"] = pd.to_datetime(hourly_l["hour"], errors="coerce", utc=True)
    hourly_l["user_id"] = pd.to_numeric(hourly_l["user_id"], errors="coerce")
    hourly_l = hourly_l.dropna(subset=["user_id", "hour"]).copy()
    hourly_l["user_id"] = hourly_l["user_id"].astype(int)

    # Keep only events for cohort users and only activity after signup timestamp.
    valid_events = new_users.merge(hourly_l, on="user_id", how="inner")
    valid_events = valid_events.loc[valid_events["hour"] >= valid_events["created_at"]].copy()
    # Sort to make "first" and "last" operations deterministic.
    valid_events = valid_events.sort_values(["user_id", "hour"]).reset_index(drop=True)
    if valid_events.empty:
        return pd.DataFrame(), int(new_users["user_id"].nunique())

    # Build first-touch and last-touch event views per user.
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

    # Merge first/last touches and enrich with per-user activity depth.
    lifecycle = first_actions.merge(last_actions, on="user_id", how="inner")
    lifecycle["first_source"] = lifecycle["first_source"].astype(str).str.strip().str.lower()
    usage_row_counts = valid_events.groupby("user_id").size().reset_index(name="usage_row_count")
    lifecycle = lifecycle.merge(usage_row_counts, on="user_id", how="left")
    lifecycle["usage_row_count"] = lifecycle["usage_row_count"].fillna(0).astype(int)
    # "single_row_only" is the abandonment proxy used by chart 1.
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
    # Convert boolean target into an explicit aggregation-safe column.
    tmp = lifecycle.assign(_single=lifecycle["single_row_only"].astype(bool))
    out = (
        tmp.groupby("first_source", as_index=False)
        .agg(user_count=("user_id", "count"), single_row_count=("_single", "sum"))
        .sort_values("user_count", ascending=False)
        .reset_index(drop=True)
    )
    # Derive abandonment percentage per first-touch request type.
    out["single_row_count"] = out["single_row_count"].astype(int)
    out["pct_single_row"] = 100.0 * out["single_row_count"] / out["user_count"]
    out["first_request_label"] = out["first_source"].astype(str).str.lower()
    return out


def _prepare_user_and_cost_breakdowns(
    users: pd.DataFrame, research_requests: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict] | None:
    # Normalize inputs so all downstream column access is stable.
    users_l = _lowercase_columns(users)
    rr = _lowercase_columns(research_requests)

    # Validate required columns for user segmentation and request-cost charts.
    required_users = {"user_id", "has_paygo", "plan"}
    required_rr = {"user_id", "model", "request_cost"}
    if not required_users.issubset(users_l.columns) or not required_rr.issubset(rr.columns):
        return None

    # Build canonical user table with paying/free classification.
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

    # Clean research requests for cost modeling.
    rr = rr.copy()
    rr["user_id"] = pd.to_numeric(rr["user_id"], errors="coerce")
    rr["request_cost"] = pd.to_numeric(rr["request_cost"], errors="coerce")
    rr = rr.dropna(subset=["user_id", "request_cost"]).copy()
    rr["user_id"] = rr["user_id"].astype(int)
    rr["model"] = rr["model"].astype(str).str.strip().str.lower()

    # Join request rows with user segments so we can slice by user_type/model.
    merged = rr.merge(
        users_l[["user_id", "is_paying_user", "user_type"]], on="user_id", how="left"
    )
    merged["is_paying_user"] = merged["is_paying_user"].fillna(False).astype(bool)
    merged["user_type"] = merged["user_type"].fillna("Free Users")

    # Calculate free-user Pro usage burden and a Mini-routing counterfactual.
    free_pro = merged[(~merged["is_paying_user"]) & (merged["model"] == "pro")].copy()
    free_pro_request_count = int(len(free_pro))
    total_pro_cost_free = float(free_pro["request_cost"].sum())
    free_pro_avg_cost = float(free_pro["request_cost"].mean()) if free_pro_request_count > 0 else 0.0
    mini_avg_cost = float(
        rr.loc[rr["model"] == "mini", "request_cost"].mean()
    ) if (rr["model"] == "mini").any() else 0.0
    hypothetical_mini_cost = float(free_pro_request_count * mini_avg_cost)
    potential_savings = float(total_pro_cost_free - hypothetical_mini_cost)

    # Dataset for chart 3: user base split (free vs paying).
    user_dist = (
        users_l.drop_duplicates(subset=["user_id"])
        .groupby("user_type", as_index=False)["user_id"]
        .nunique()
        .rename(columns={"user_id": "users"})
    )
    # Dataset for chart 4: request cost distribution per model.
    request_cost_dist = rr[rr["model"].isin(["mini", "pro"])][["model", "request_cost"]].copy()
    request_cost_dist["model"] = request_cost_dist["model"].str.upper()
    # Dataset for chart 5: total request cost by model and user segment.
    cost_by_model_user = (
        merged.groupby(["model", "user_type"], as_index=False)["request_cost"].sum()
    )
    cost_by_model_user = cost_by_model_user[cost_by_model_user["model"].isin(["mini", "pro"])]
    economics_summary = {
        "free_pro_request_count": free_pro_request_count,
        "total_pro_cost_free": total_pro_cost_free,
        "free_pro_avg_cost": free_pro_avg_cost,
        "mini_avg_cost": mini_avg_cost,
        "hypothetical_mini_cost": hypothetical_mini_cost,
        "potential_savings": potential_savings,
    }
    return user_dist, request_cost_dist, cost_by_model_user, economics_summary


def _prepare_pareto(research_requests: pd.DataFrame) -> tuple[pd.DataFrame, float] | None:
    # Normalize and validate the user_id key.
    rr = _lowercase_columns(research_requests)
    if "user_id" not in rr.columns:
        return None
    rr["user_id"] = pd.to_numeric(rr["user_id"], errors="coerce")
    rr = rr.dropna(subset=["user_id"]).copy()
    rr["user_id"] = rr["user_id"].astype(int)
    # Count request volume per user and sort descending for Pareto accumulation.
    counts = rr.groupby("user_id").size().sort_values(ascending=False)
    if counts.empty:
        return None

    # Build cumulative share curve of traffic over cumulative users.
    pareto = pd.DataFrame({"requests": counts.values})
    pareto["cum_requests_pct"] = 100.0 * pareto["requests"].cumsum() / pareto["requests"].sum()
    pareto["cum_users_pct"] = 100.0 * (pareto.index + 1) / len(pareto)
    # Prefix (0,0) point so the curve starts at origin.
    pareto = pd.concat(
        [
            pd.DataFrame({"cum_users_pct": [0.0], "cum_requests_pct": [0.0]}),
            pareto[["cum_users_pct", "cum_requests_pct"]],
        ],
        ignore_index=True,
    )
    # Read the y-value at first point where cumulative users reaches 5%.
    y_at_5 = float(
        pareto.loc[pareto["cum_users_pct"] >= 5.0, "cum_requests_pct"].head(1).fillna(0.0).iloc[0]
    )
    return pareto, y_at_5


def _prepare_cancelled_response_times(research_requests: pd.DataFrame) -> pd.DataFrame | None:
    # Keep only cancelled requests with valid numeric response time.
    rr = _lowercase_columns(research_requests)
    if not {"status", "response_time_seconds"}.issubset(rr.columns):
        return None
    rr["is_cancelled"] = _is_cancelled_status(rr["status"])
    rr["response_time_seconds"] = pd.to_numeric(rr["response_time_seconds"], errors="coerce")
    return rr.loc[rr["is_cancelled"]].dropna(subset=["response_time_seconds"])


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


def _format_k_cost(value: float) -> str:
    return f"${value / 1_000:.2f}K"


def _format_k_plain(value: float) -> str:
    return f"{value / 1_000:.2f}K"


# ------------------------------
# infrastructure data preparation
# ------------------------------
def _prepare_finops_data(
    infrastructure_costs: pd.DataFrame,
    hourly_usage: pd.DataFrame,
    research_requests: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    # Normalize all inputs before schema checks and numeric coercion.
    infra = _lowercase_columns(infrastructure_costs)
    hourly = _lowercase_columns(hourly_usage)
    rr = _lowercase_columns(research_requests)

    # We need hourly infrastructure data plus infra/model cost columns.
    if "hour" not in infra.columns:
        return None
    infra_cols = [c for c in infra.columns if c.startswith("infra_")]
    model_cols = [c for c in infra.columns if c.startswith("model_")]
    if not infra_cols or not model_cols:
        return None

    # Parse timestamp and coerce all cost components to numeric.
    infra = infra.copy()
    infra["hour"] = pd.to_datetime(infra["hour"], errors="coerce", utc=True)
    infra = infra.dropna(subset=["hour"]).copy()
    for col in infra_cols + model_cols:
        infra[col] = pd.to_numeric(infra[col], errors="coerce").fillna(0.0)

    # Derive total infra/model/hourly cost and page-level KPI totals.
    infra["infra_total_cost"] = infra[infra_cols].sum(axis=1)
    infra["ai_total_cost"] = infra[model_cols].sum(axis=1)
    infra["total_hourly_cost"] = infra["infra_total_cost"] + infra["ai_total_cost"]
    total_hardware_cost = float(infra["infra_total_cost"].sum())
    total_ai_cost = float(infra["ai_total_cost"].sum())

    # Build day-level aggregation and day/hour heatmap source.
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

    # Build daily request volume either from hourly_usage or research timestamps fallback.
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

    # Join daily cost and request views into a unified time series.
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
    # Drop last partial day to avoid incomplete-day distortion.
    if not daily_agg.empty:
        last_day = daily_agg["day"].max()
        daily_agg = daily_agg[daily_agg["day"] < last_day].copy()

    # Prepare monthly rollup for potential future use.
    daily_agg["month"] = daily_agg["day"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    monthly_agg = (
        daily_agg.groupby("month", as_index=False)[
            ["total_requests", "infra_total_cost", "total_daily_cost"]
        ]
        .sum()
        .sort_values("month")
    )
    monthly_agg["month_label"] = monthly_agg["month"].dt.strftime("%Y-%m")

    # Compute dead-day waste KPI: cost on days with zero requests.
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


def _prepare_cancellation_chart_data(
    research_requests: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | int]] | None:
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
        return None

    rr = rr.copy()
    rr["response_time_seconds"] = pd.to_numeric(rr["response_time_seconds"], errors="coerce")
    rr["llm_calls"] = pd.to_numeric(rr["llm_calls"], errors="coerce")
    rr["num_sources"] = pd.to_numeric(rr["num_sources"], errors="coerce")
    rr["credits_used"] = pd.to_numeric(rr["credits_used"], errors="coerce")
    rr["request_cost"] = pd.to_numeric(rr["request_cost"], errors="coerce")
    rr["is_cancelled"] = _is_cancelled_status(rr["status"])
    rr["stream_flag"] = _is_true_stream(rr["stream"])
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
                "median LLM calls": ("llm_calls", "median"),
                "median sources found": ("num_sources", "median"),
            }
        )
    )
    inefficiency["duration_group"] = pd.Categorical(
        inefficiency["duration_group"], categories=["< 90 seconds", ">= 90 seconds"], ordered=True
    )
    inefficiency = inefficiency.sort_values("duration_group")
    inefficiency_long = inefficiency.melt(
        id_vars="duration_group",
        value_vars=["median LLM calls", "median sources found"],
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

    stream_summary = (
        rr.groupby("stream_flag", as_index=False)
        .agg(total_requests=("is_cancelled", "size"), cancelled_requests=("is_cancelled", "sum"))
        .copy()
    )
    stream_summary["cancellation_rate_pct"] = (
        100.0 * stream_summary["cancelled_requests"] / stream_summary["total_requests"].clip(lower=1)
    )
    stream_metrics = {
        "streaming_total_requests": 0,
        "streaming_cancelled_requests": 0,
        "streaming_cancellation_rate_pct": 0.0,
        "non_streaming_total_requests": 0,
        "non_streaming_cancelled_requests": 0,
        "non_streaming_cancellation_rate_pct": 0.0,
    }
    for _, row in stream_summary.iterrows():
        is_streaming = bool(row["stream_flag"])
        prefix = "streaming" if is_streaming else "non_streaming"
        stream_metrics[f"{prefix}_total_requests"] = int(row["total_requests"])
        stream_metrics[f"{prefix}_cancelled_requests"] = int(row["cancelled_requests"])
        stream_metrics[f"{prefix}_cancellation_rate_pct"] = float(row["cancellation_rate_pct"])

    return wait_effect, inefficiency_long, billing_dist, stream_metrics


# -----------------------------------------------------
# product analysis rendering helpers (dashboard order)
# -----------------------------------------------------
def _render_product_top_metrics(
    acquisition_pct: float,
    total_request_cost: float,
    success_rate_pct: float,
    cancellation_rate_pct: float,
    success_request_count: int,
    cancelled_request_count: int,
    streaming_cancellation_rate_pct: float,
    non_streaming_cancellation_rate_pct: float,
    streaming_cancelled_requests: int,
    streaming_total_requests: int,
    non_streaming_cancelled_requests: int,
    non_streaming_total_requests: int,
) -> None:
    with st.container():
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "Research API users acquisition percentage",
                f"{acquisition_pct:.2f}%",
                help=(
                    "out of 16,324 total users, 12,895 joined on/after Nov 1, 2025. "
                    "out of them, 12,006 had at least one activity after their join date. "
                    "out of those active new users, 2,270 used the Research API as their first activity."
                ),
            )
        with m2:
            st.metric(
                "Success rate",
                f"{success_rate_pct:.2f}%",
                help=(
                    "share of research API requests with status success. "
                    f"total success requests: {success_request_count:,}."
                ),
            )
        with m3:
            st.metric(
                "Non-streaming cancellation rate",
                f"{non_streaming_cancellation_rate_pct:.2f}%",
                help=(
                    f"cancelled requests: {non_streaming_cancelled_requests:,} "
                    f"out of {non_streaming_total_requests:,} non-streaming requests."
                ),
            )

    with st.container():
        m4, m5, m6 = st.columns(3)
        with m4:
            st.metric(
                "Total requests costs",
                _format_compact_cost(total_request_cost),
                help="sum of all research API request costs.",
            )
        with m5:
            st.metric(
                "Cancellation rate",
                f"{cancellation_rate_pct:.2f}%",
                help=(
                    "share of research API requests with status cancelled. "
                    f"total cancelled requests: {cancelled_request_count:,}."
                ),
            )
        with m6:
            st.metric(
                "Streaming cancellation rate",
                f"{streaming_cancellation_rate_pct:.2f}%",
                help=(
                    f"cancelled requests: {streaming_cancelled_requests:,} "
                    f"out of {streaming_total_requests:,} streaming requests."
                ),
            )


def _render_abandonment_chart(lifecycle: pd.DataFrame) -> None:
    no_return_df = _single_row_no_return_by_first_request(lifecycle)
    if no_return_df.empty:
        st.warning("Not enough data for single-row usage by first request type.")
        return

    no_return_df = no_return_df.copy()
    no_return_df["first_request_label_display"] = (
        no_return_df["first_request_label"].astype(str).str.capitalize()
    )
    label_order = no_return_df["first_request_label_display"].tolist()
    bar_colors = [
        "#E45756"
        if label.lower() == "research"
        else "#4C78A8"
        for label in no_return_df["first_request_label_display"]
    ]
    worst_row = no_return_df.loc[no_return_df["pct_single_row"].idxmax()]
    research_row = no_return_df.loc[
        no_return_df["first_request_label_display"].str.lower().eq("research")
    ]
    research_rate_text = f"{worst_row['pct_single_row']:.2f}%"
    if not research_row.empty:
        research_rate_text = f"{research_row.iloc[0]['pct_single_row']:.2f}%"

    fig_retention = px.bar(
        no_return_df,
        x="first_request_label_display",
        y="pct_single_row",
        title="Abandonment rate by platform features",
        labels={
            "first_request_label_display": "first platform feature",
            "pct_single_row": "Abandonment rate (%)",
        },
        text=no_return_df["pct_single_row"].map(lambda x: f"{x:.2f}%"),
        category_orders={"first_request_label_display": label_order},
        custom_data=["user_count", "single_row_count"],
    )
    fig_retention.update_traces(
        marker_color=bar_colors,
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "%{x}<br>"
            "Total users: %{customdata[0]:,.0f}<br>"
            "Abandoned users: %{customdata[1]:,.0f}<br>"
            "Abandonment rate: %{y:.2f}%<extra></extra>"
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
    st.caption(
        "This chart analyzes user retention based on the initial platform interaction. "
        "It shows the abandonment rate, defined as no subsequent engagement across the platform after first usage, "
        "segmented by the first feature used. "
        f"Notably, the Research feature shows an abandonment rate of {research_rate_text}, "
        "which is significantly higher than other platform features."
    )


def _render_traffic_share_chart(research_requests: pd.DataFrame) -> None:
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
        title="Traffic share distribution over users share",
        labels={
            "cum_users_pct": "Share of research API users (%)",
            "cum_requests_pct": "Share of research API traffic (%)",
        },
    )
    fig_pareto.add_vline(x=5.0, line_dash="dash", line_color="#4C78A8", line_width=3)
    fig_pareto.add_hline(y=y_at_5, line_dash="dash", line_color="gray")
    fig_pareto.update_traces(
        selector=dict(type="scatter", mode="lines"),
        line=dict(color="#4C78A8", width=5),
        fill="tozeroy",
        fillcolor="rgba(76,120,168,0.30)",
    )
    if len(fig_pareto.data) >= 1:
        fig_pareto.data[0].hovertemplate = "users: %{x:.2f}%<br>requests: %{y:.2f}%<extra></extra>"
    fig_pareto.update_layout(
        template="simple_white",
        title_font=dict(size=20),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        font=dict(size=13),
        margin=dict(t=60, b=40, l=30, r=30),
    )
    st.plotly_chart(fig_pareto, use_container_width=True)
    st.caption(
        "This chart shows how product traffic is distributed across users. "
        "It compares cumulative user share to cumulative request share. "
        f"For example, the top 5% of users account for about {y_at_5:.2f}% of total traffic, "
        "showing a small portion of heavy users."
    )


def _render_user_base_chart(user_dist: pd.DataFrame) -> None:
    user_dist_display = user_dist.copy()
    user_dist_display["user_type_display"] = (
        user_dist_display["user_type"]
        .astype(str)
        .replace({"Free Users": "Free users", "Paying Users": "Paying users"})
    )
    fig_user_dist = px.pie(
        user_dist_display,
        values="users",
        names="user_type_display",
        hole=0.5,
        title="Free tier and paying users share",
        color="user_type_display",
        color_discrete_map={"Free users": "#E45756", "Paying users": "#4C78A8"},
    )
    fig_user_dist.update_layout(
        template="simple_white",
        title_font=dict(size=20),
        font=dict(size=13),
        legend_title_text="",
    )
    fig_user_dist.update_traces(
        hovertemplate="%{label}: %{value:,.0f}<br>Share: %{percent:.2%}<extra></extra>",
        textfont_color="white",
    )
    st.plotly_chart(fig_user_dist, use_container_width=True)
    total_users = float(user_dist_display["users"].sum())
    paying_users = float(
        user_dist_display.loc[
            user_dist_display["user_type_display"].eq("Paying users"), "users"
        ].sum()
    )
    paying_share = 100.0 * paying_users / total_users if total_users > 0 else 0.0
    st.caption(
        "This chart shows the distribution of the user base between free and paying users. "
        f"Paying users represent about {paying_share:.2f}% of total users, "
        "indicating that the paying segment remains relatively limited."
    )


def _render_request_cost_distribution_chart(request_cost_dist: pd.DataFrame) -> None:
    model_colors_upper = {"MINI": "#72B7B2", "PRO": "#E45756"}
    mini_median_cost = float(
        request_cost_dist.loc[request_cost_dist["model"].eq("MINI"), "request_cost"].median()
    )
    pro_median_cost = float(
        request_cost_dist.loc[request_cost_dist["model"].eq("PRO"), "request_cost"].median()
    )
    fig_avg_cost = px.box(
        request_cost_dist,
        x="model",
        y="request_cost",
        title="Request cost distribution by model",
        labels={"model": "Model", "request_cost": "Request cost ($)"},
        color="model",
        color_discrete_map=model_colors_upper,
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
        opacity=0.8,
        line=dict(width=2),
        hovertemplate="model: %{x}<br>request cost: $%{y:,.2f}<extra></extra>"
    )
    fig_avg_cost.update_traces(
        selector=dict(name="MINI"),
        fillcolor="rgba(45, 102, 173, 0.8)",
        line=dict(color="#4C78A8", width=2),
    )
    fig_avg_cost.update_traces(
        selector=dict(name="PRO"),
        fillcolor="rgba(214, 68, 67, 0.8)",
        line=dict(color="#E45756", width=2),
    )
    fig_avg_cost.update_yaxes(tickprefix="$")
    st.plotly_chart(fig_avg_cost, use_container_width=True)
    if pd.notna(mini_median_cost) and pd.notna(pro_median_cost) and mini_median_cost > 0:
        pro_vs_mini_ratio = pro_median_cost / mini_median_cost
        st.caption(
            "This chart shows the distribution of request costs by model tier. "
            f"By median, PRO requests cost about {pro_vs_mini_ratio:.2f}x more than MINI requests, "
            "highlighting the unit-cost gap between models."
        )
    else:
        st.caption(
            "This chart shows the distribution of request costs by model tier, "
            "highlighting the unit-cost gap between models."
        )


def _render_total_cost_by_model_user_chart(cost_by_model_user: pd.DataFrame, economics_summary: dict) -> None:
    cost_by_model_user_display = cost_by_model_user.copy()
    cost_by_model_user_display["model_display"] = (
        cost_by_model_user_display["model"].astype(str).str.upper()
    )
    cost_by_model_user_display["user_type_display"] = (
        cost_by_model_user_display["user_type"]
        .astype(str)
        .replace({"Paying Users": "Paying users", "Free Users": "Free users"})
    )
    user_colors = {"Paying users": "#4C78A8", "Free users": "#E45756"}
    fig_stacked = px.bar(
        cost_by_model_user_display,
        x="model",
        y="request_cost",
        color="user_type_display",
        barmode="stack",
        title="Total request cost by model and user type",
        labels={
            "model": "Model",
            "request_cost": "Total request cost ($)",
            "user_type_display": "User type",
        },
        color_discrete_map=user_colors,
        category_orders={"user_type_display": ["Paying users", "Free users"]},
        custom_data=["model_display"],
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
        hovertemplate="Model: %{customdata[0]}<br>User type: %{fullData.name}<br>Total cost: $%{y:,.0f}<extra></extra>"
    )
    fig_stacked.update_yaxes(tickprefix="$")
    st.plotly_chart(fig_stacked, use_container_width=True)
    free_pro_requests_compact = f"{economics_summary['free_pro_request_count'] / 1000:.0f}K"
    total_pro_cost_free_compact = _format_compact_cost(economics_summary["total_pro_cost_free"]).replace(
        "$", "\\$"
    )
    hypothetical_mini_cost_compact = _format_compact_cost(economics_summary["hypothetical_mini_cost"]).replace(
        "$", "\\$"
    )
    potential_savings_compact = _format_compact_cost(economics_summary["potential_savings"]).replace(
        "$", "\\$"
    )
    st.caption(
        "Free users currently generate substantial spend on the Pro model. "
        f"They made {free_pro_requests_compact} Pro requests at about "
        f"\\${economics_summary['free_pro_avg_cost']:,.0f} per request, creating roughly "
        f"{total_pro_cost_free_compact} in direct cost. "
        f"If those requests were routed to Mini (about \\${economics_summary['mini_avg_cost']:,.0f} per request), "
        f"cost would be about {hypothetical_mini_cost_compact}, "
        f"with potential savings of about {potential_savings_compact}."
    )


def _render_cancelled_response_time_histogram(research_requests: pd.DataFrame) -> None:
    cancelled_points = _prepare_cancelled_response_times(research_requests)
    if cancelled_points is None:
        st.warning("Missing `status` or `response_time_seconds` in research data.")
        return
    if cancelled_points.empty:
        st.warning("No usable cancelled-request response-time data found.")
        return

    max_response_time = float(cancelled_points["response_time_seconds"].max())
    histogram_end = max(30.0, math.ceil(max_response_time / 30.0) * 30.0)
    bin_edges = list(range(0, int(histogram_end) + 30, 30))
    bucket_labels = [f"{start}-{start + 30}" for start in bin_edges[:-1]]
    cancelled_points = cancelled_points.copy()
    cancelled_points["duration_bucket"] = pd.cut(
        cancelled_points["response_time_seconds"],
        bins=bin_edges,
        right=False,
        include_lowest=True,
        labels=bucket_labels,
    )
    bucket_counts = (
        cancelled_points.groupby("duration_bucket", observed=False)
        .size()
        .reindex(bucket_labels, fill_value=0)
        .reset_index(name="cancelled_requests")
    )

    fig_cancelled_hist = go.Figure(
        data=[
            go.Bar(
                x=bucket_counts["duration_bucket"],
                y=bucket_counts["cancelled_requests"],
                marker=dict(color="#2D66AD"),
                hovertemplate=(
                    "Response time bucket: %{x} sec<br>"
                    "Cancelled requests: %{y:,.0f}<extra></extra>"
                ),
            )
        ]
    )
    fig_cancelled_hist.update_layout(
        template="simple_white",
        showlegend=False,
        title="Cancelled requests response time (30-second bins)",
        title_font=dict(size=20),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        font=dict(size=13),
        margin=dict(t=60, b=40, l=30, r=30),
    )
    fig_cancelled_hist.update_xaxes(title_text="Response time (seconds)")
    fig_cancelled_hist.update_yaxes(title_text="Cancelled requests")
    st.plotly_chart(fig_cancelled_hist, use_container_width=True)
    peak_bucket_row = bucket_counts.loc[bucket_counts["cancelled_requests"].idxmax()]
    peak_bucket_label = str(peak_bucket_row["duration_bucket"])
    peak_bucket_count = int(peak_bucket_row["cancelled_requests"])
    total_cancelled = int(bucket_counts["cancelled_requests"].sum())
    cancelled_over_90 = int(
        cancelled_points.loc[cancelled_points["response_time_seconds"] >= 90.0].shape[0]
    )
    over_90_pct = (100.0 * cancelled_over_90 / total_cancelled) if total_cancelled > 0 else 0.0
    st.caption(
        "This chart shows the distribution of cancelled requests by response-time bins. "
        f"The highest concentration appears in the {peak_bucket_label} second bin "
        f"({peak_bucket_count:,} cancelled requests). "
        f"Overall, about {over_90_pct:.1f}% of cancellations occur after 90 seconds, "
        "indicating that longer waits are strongly associated with cancellation behavior."
    )


def _render_cancellation_rate_by_wait_time_chart(wait_effect: pd.DataFrame) -> None:
    fig_wait = px.bar(
        wait_effect,
        x="duration_group",
        y="cancel_rate",
        title="Cancellation rate by wait time",
        labels={"duration_group": "Duration group", "cancel_rate": "Cancel rate"},
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
            "Total requests: %{customdata[0]:,.0f}<br>"
            "Total cancelled: %{customdata[1]:,.0f}<br>"
            "Cancel rate: %{y:.2%}<extra></extra>"
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
    fast_row = wait_effect.loc[wait_effect["duration_group"].astype(str).eq("< 90 seconds")]
    slow_row = wait_effect.loc[wait_effect["duration_group"].astype(str).eq(">= 90 seconds")]
    fast_rate = float(fast_row["cancel_rate"].iloc[0]) if not fast_row.empty else 0.0
    slow_rate = float(slow_row["cancel_rate"].iloc[0]) if not slow_row.empty else 0.0
    fast_cancelled = int(fast_row["cancelled_count"].iloc[0]) if not fast_row.empty else 0
    slow_cancelled = int(slow_row["cancelled_count"].iloc[0]) if not slow_row.empty else 0
    rate_gap = (slow_rate - fast_rate) * 100.0
    st.caption(
        "This chart compares cancellation rates across wait-time duration groups for streaming-enabled requests only. "
        f"For requests under 90 seconds, the cancellation rate is {fast_rate * 100.0:.1f}% "
        f"({fast_cancelled:,} cancelled requests), while for requests at or above 90 seconds "
        f"it rises to {slow_rate * 100.0:.1f}% ({slow_cancelled:,} cancelled requests). "
        f"This represents a gap of {rate_gap:.1f} percentage points, highlighting a clear wait-time threshold effect."
    )


def _render_technical_inefficiency_by_wait_time_chart(inefficiency_long: pd.DataFrame) -> None:
    llm_fast = inefficiency_long.loc[
        inefficiency_long["duration_group"].astype(str).eq("< 90 seconds")
        & inefficiency_long["metric"].eq("median LLM calls"),
        "value",
    ]
    llm_slow = inefficiency_long.loc[
        inefficiency_long["duration_group"].astype(str).eq(">= 90 seconds")
        & inefficiency_long["metric"].eq("median LLM calls"),
        "value",
    ]
    sources_fast = inefficiency_long.loc[
        inefficiency_long["duration_group"].astype(str).eq("< 90 seconds")
        & inefficiency_long["metric"].eq("median sources found"),
        "value",
    ]
    sources_slow = inefficiency_long.loc[
        inefficiency_long["duration_group"].astype(str).eq(">= 90 seconds")
        & inefficiency_long["metric"].eq("median sources found"),
        "value",
    ]
    llm_fast_val = float(llm_fast.iloc[0]) if not llm_fast.empty else 0.0
    llm_slow_val = float(llm_slow.iloc[0]) if not llm_slow.empty else 0.0
    sources_fast_val = float(sources_fast.iloc[0]) if not sources_fast.empty else 0.0
    sources_slow_val = float(sources_slow.iloc[0]) if not sources_slow.empty else 0.0

    fig_ineff = px.bar(
        inefficiency_long,
        x="duration_group",
        y="value",
        color="metric",
        barmode="group",
        title="Technical inefficiency by wait time",
        labels={"duration_group": "Duration group", "value": "Average", "metric": ""},
        color_discrete_map={"median LLM calls": "#E45756", "median sources found": "#4C78A8"},
        text=inefficiency_long["value"].map(lambda v: f"{v:.2f}"),
    )
    fig_ineff.update_traces(
        textposition="outside",
        cliponaxis=False,
        hovertemplate="Duration: %{x}<br>%{fullData.name}: %{y:.2f}<extra></extra>",
    )
    fig_ineff.update_layout(
        template="simple_white",
        showlegend=False,
        title_font=dict(size=20),
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        font=dict(size=13),
        legend_title_text="",
    )
    st.plotly_chart(fig_ineff, use_container_width=True)
    st.caption(
        "This chart compares technical workload across wait-time groups. "
        f"When duration is under 90 seconds, median LLM calls are {llm_fast_val:.1f} and median sources found are {sources_fast_val:.1f}. "
        f"For durations at or above 90 seconds, median LLM calls increase to {llm_slow_val:.1f}, "
        "while median sources found did not change. "
        "This pattern indicates higher computation with no value."
    )


def _render_cancelled_request_billing_status_chart(billing_dist: pd.DataFrame) -> None:
    total_cancelled = int(billing_dist["requests"].sum())
    unbilled_row = billing_dist.loc[billing_dist["billing_status"].eq("Unbilled (0 credits)"), "requests"]
    billed_row = billing_dist.loc[billing_dist["billing_status"].eq("Billed (>0 credits)"), "requests"]
    unbilled_count = int(unbilled_row.iloc[0]) if not unbilled_row.empty else 0
    billed_count = int(billed_row.iloc[0]) if not billed_row.empty else 0
    unbilled_pct = (100.0 * unbilled_count / total_cancelled) if total_cancelled > 0 else 0.0
    billed_pct = (100.0 * billed_count / total_cancelled) if total_cancelled > 0 else 0.0

    fig_billing = px.pie(
        billing_dist,
        names="billing_status",
        values="requests",
        hole=0.5,
        title="Billing status of cancelled requests",
        color="billing_status",
        color_discrete_map={"Unbilled (0 credits)": "#E45756", "Billed (>0 credits)": "#4C78A8"},
    )
    fig_billing.update_traces(
        hovertemplate="%{label}<br>Requests: %{value:,.0f}<br>Share: %{percent:.2%}<extra></extra>",
        textfont_color="white",
    )
    fig_billing.update_layout(
        template="simple_white",
        title_font=dict(size=20),
        font=dict(size=13),
        legend_title_text="",
    )
    st.plotly_chart(fig_billing, use_container_width=True)
    st.caption(
        "This chart shows how cancelled requests are split by billing outcome. "
        f"Unbilled cancellations account for about {unbilled_pct:.1f}% ({unbilled_count:,} requests), "
        f"while billed cancellations account for about {billed_pct:.1f}% ({billed_count:,} requests). "
        "This means the company is currently paying for most cancelled requests through unbilled credits."
    )


def render_product_analysis(
    users: pd.DataFrame, hourly_usage: pd.DataFrame, research_requests: pd.DataFrame
) -> None:
    st.title("Research API product analysis")

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

    q2_data = _prepare_user_and_cost_breakdowns(users, research_requests)
    if q2_data is None:
        st.error("Missing required columns for economics analysis.")
        return
    user_dist, request_cost_dist, cost_by_model_user, economics_summary = q2_data

    rr_cost = _lowercase_columns(research_requests)
    total_request_cost = 0.0
    success_rate_pct = 0.0
    cancellation_rate_pct = 0.0
    success_request_count = 0
    cancelled_request_count = 0
    streaming_cancellation_rate_pct = 0.0
    non_streaming_cancellation_rate_pct = 0.0
    streaming_cancelled_requests = 0
    streaming_total_requests = 0
    non_streaming_cancelled_requests = 0
    non_streaming_total_requests = 0
    if "request_cost" in rr_cost.columns:
        total_request_cost = float(
            pd.to_numeric(rr_cost["request_cost"], errors="coerce").fillna(0).sum()
        )
    if "status" in rr_cost.columns:
        status_norm = rr_cost["status"].astype(str).str.strip().str.lower()
        is_cancelled = _is_cancelled_status(status_norm)
        denominator = len(status_norm)
        if denominator > 0:
            success_request_count = int(status_norm.eq("success").sum())
            cancelled_request_count = int(is_cancelled.sum())
            success_rate_pct = 100.0 * float(success_request_count) / denominator
            cancellation_rate_pct = (
                100.0 * float(cancelled_request_count) / denominator
            )
        if "stream" in rr_cost.columns:
            stream_flag = _is_true_stream(rr_cost["stream"])
            streaming_total_requests = int(stream_flag.sum())
            non_streaming_total_requests = int((~stream_flag).sum())
            streaming_cancelled_requests = int((is_cancelled & stream_flag).sum())
            non_streaming_cancelled_requests = int((is_cancelled & ~stream_flag).sum())
            if streaming_total_requests > 0:
                streaming_cancellation_rate_pct = (
                    100.0 * float(streaming_cancelled_requests) / streaming_total_requests
                )
            if non_streaming_total_requests > 0:
                non_streaming_cancellation_rate_pct = (
                    100.0 * float(non_streaming_cancelled_requests) / non_streaming_total_requests
                )

    # 1) top metrics
    _render_product_top_metrics(
        acquisition_pct,
        total_request_cost,
        success_rate_pct,
        cancellation_rate_pct,
        success_request_count,
        cancelled_request_count,
        streaming_cancellation_rate_pct,
        non_streaming_cancellation_rate_pct,
        streaming_cancelled_requests,
        streaming_total_requests,
        non_streaming_cancelled_requests,
        non_streaming_total_requests,
    )

    # 2) first row charts
    col1, col2 = st.columns(2)
    with col1:
        _render_abandonment_chart(lifecycle)
    with col2:
        _render_traffic_share_chart(research_requests)

    # 3-6) remaining charts in dashboard order
    col3, col4 = st.columns(2)
    with col3:
        _render_user_base_chart(user_dist)
    with col4:
        _render_request_cost_distribution_chart(request_cost_dist)
    _render_total_cost_by_model_user_chart(cost_by_model_user, economics_summary)

    # 6-7) response and cancellation overview
    prepared = _prepare_cancellation_chart_data(research_requests)
    col5, col6 = st.columns(2)
    with col5:
        _render_cancelled_response_time_histogram(research_requests)
    with col6:
        if prepared is None:
            st.warning("Missing required columns for Q3 cancellation analysis.")
        else:
            wait_effect, inefficiency_long, billing_dist, _stream_metrics = prepared
            _render_cancellation_rate_by_wait_time_chart(wait_effect)

    # 8-9) cancellation diagnostics side by side
    if prepared is not None:
        col7, col8 = st.columns(2)
        with col7:
            _render_technical_inefficiency_by_wait_time_chart(inefficiency_long)
        with col8:
            _render_cancelled_request_billing_status_chart(billing_dist)


# --------------------------------------------
# infrastructure and cost analysis page render
# --------------------------------------------
def render_infrastructure_and_cost_analysis(
    users: pd.DataFrame,
    infrastructure_costs: pd.DataFrame,
    hourly_usage: pd.DataFrame,
    research_requests: pd.DataFrame,
) -> None:
    coolwarm_scale = [
        [0.0, "#3B4CC0"],
        [0.2, "#6F92F3"],
        [0.4, "#AFC7FD"],
        [0.5, "#DDDCDC"],
        [0.6, "#F7B89C"],
        [0.8, "#E7745B"],
        [1.0, "#B40426"],
    ]

    st.title("Infrastructure & cost analysis")

    prepared = _prepare_finops_data(infrastructure_costs, hourly_usage, research_requests)
    if prepared is None:
        st.error("missing required fields for infrastructure & finops analysis.")
        return
    finops_metrics, daily_agg, monthly_agg, heatmap_data = prepared

    total_cost_base = finops_metrics["total_hardware_cost"] + finops_metrics["total_ai_cost"]
    infra_share_pct = (100.0 * finops_metrics["total_hardware_cost"] / total_cost_base) if total_cost_base > 0 else 0.0
    model_share_pct = (100.0 * finops_metrics["total_ai_cost"] / total_cost_base) if total_cost_base > 0 else 0.0

    users_l = _lowercase_columns(users).copy()
    hourly_l = _lowercase_columns(hourly_usage).copy()
    infra_l = _lowercase_columns(infrastructure_costs).copy()
    growth_start = pd.Timestamp("2025-11-01", tz="UTC")
    growth_end = pd.Timestamp("2026-03-01", tz="UTC")

    user_start = user_end = 0.0
    if {"created_at", "user_id"}.issubset(users_l.columns):
        users_l["created_at"] = pd.to_datetime(users_l["created_at"], errors="coerce", utc=True)
        users_l = users_l.dropna(subset=["created_at"]).copy()
        users_l["month"] = users_l["created_at"].dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
        users_monthly = users_l.groupby("month").size().rename("new_users")
        user_start = float(users_monthly.get(growth_start, 0.0))
        user_end = float(users_monthly.get(growth_end, 0.0))

    req_start = req_end = 0.0
    if {"hour", "request_count"}.issubset(hourly_l.columns):
        hourly_l["hour"] = pd.to_datetime(hourly_l["hour"], errors="coerce", utc=True)
        hourly_l["request_count"] = pd.to_numeric(hourly_l["request_count"], errors="coerce").fillna(0.0)
        hourly_l = hourly_l.dropna(subset=["hour"]).copy()
        hourly_l["month"] = hourly_l["hour"].dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
        req_monthly = hourly_l.groupby("month")["request_count"].sum()
        req_start = float(req_monthly.get(growth_start, 0.0))
        req_end = float(req_monthly.get(growth_end, 0.0))

    infra_start = infra_end = 0.0
    infra_cols = [c for c in infra_l.columns if c.startswith("infra_")]
    if "hour" in infra_l.columns and infra_cols:
        infra_l["hour"] = pd.to_datetime(infra_l["hour"], errors="coerce", utc=True)
        infra_l = infra_l.dropna(subset=["hour"]).copy()
        for col in infra_cols:
            infra_l[col] = pd.to_numeric(infra_l[col], errors="coerce").fillna(0.0)
        infra_l["infra_total"] = infra_l[infra_cols].sum(axis=1)
        infra_l["month"] = infra_l["hour"].dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
        infra_monthly = infra_l.groupby("month")["infra_total"].sum()
        infra_start = float(infra_monthly.get(growth_start, 0.0))
        infra_end = float(infra_monthly.get(growth_end, 0.0))

    def _growth_ratio(start_value: float, end_value: float) -> float:
        if start_value <= 0:
            return 0.0
        return end_value / start_value

    def _format_m_count(value: float) -> str:
        return f"{value / 1_000_000:.1f}M"

    users_growth_ratio = _growth_ratio(user_start, user_end)
    requests_growth_ratio = _growth_ratio(req_start, req_end)
    infra_growth_ratio = _growth_ratio(infra_start, infra_end)

    k1, g1, g2, g3 = st.columns(4)
    with k1:
        st.metric(
            "Total infrastructure cost",
            _format_k_cost(total_cost_base),
            help=(
                "Total measured platform cost across infrastructure and model usage. "
                f"Within this total, infrastructure is about {infra_share_pct:.1f}% and model costs are about {model_share_pct:.1f}%."
            ),
        )
    with g1:
        st.metric(
            "New users growth (Nov to Mar)",
            f"X{users_growth_ratio:.2f}",
            help=(
                f"Ratio of monthly new users in Mar 2026 vs Nov 2025 "
                f"({user_end:,.0f} vs {user_start:,.0f})."
            ),
        )
    with g2:
        st.metric(
            "Requests growth (Nov to Mar)",
            f"X{requests_growth_ratio:.2f}",
            help=(
                f"Ratio of monthly requests in Mar 2026 vs Nov 2025 "
                f"({_format_m_count(req_end)} vs {_format_m_count(req_start)})."
            ),
        )
    with g3:
        st.metric(
            "Infrastructure growth (Nov to Mar)",
            f"X{infra_growth_ratio:.2f}",
            help=(
                f"Ratio of monthly infrastructure cost in Mar 2026 vs Nov 2025 "
                f"({_format_k_plain(infra_end)} vs {_format_k_plain(infra_start)})."
            ),
        )

    hourly_stability = pd.DataFrame(columns=["infra_total"])
    if "infra_total" in infra_l.columns:
        hourly_stability = infra_l[["infra_total"]].dropna().copy()
    hourly_min = float(hourly_stability["infra_total"].min()) if not hourly_stability.empty else 0.0
    hourly_median = float(hourly_stability["infra_total"].median()) if not hourly_stability.empty else 0.0
    hourly_max = float(hourly_stability["infra_total"].max()) if not hourly_stability.empty else 0.0

    col1, col2 = st.columns(2)
    with col1:
        budget_split = pd.DataFrame(
            {
                "category": ["Infrastructure costs", "Model costs"],
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
            title="Share of infrastructure and model costs out of the total cost",
            color="category",
            color_discrete_map={
                "Infrastructure costs": "#4C78A8",
                "Model costs": "#E45756",
            },
        )
        fig_donut.update_traces(
            hovertemplate="%{label}<br>Cost: $%{value:,.2f}<br>Share: %{percent:.2%}<extra></extra>",
            textfont_color="white",
        )
        fig_donut.update_layout(
            template="simple_white",
            title_font=dict(size=20),
            font=dict(size=13),
            legend_title_text="",
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.caption(
            "This chart shows the split between infrastructure and model spending. "
            f"Infrastructure accounts for about {infra_share_pct:.1f}% of total measured cost, "
            f"while model costs account for about {model_share_pct:.1f}%."
        )

    with col2:
        fig_stability = px.box(
            hourly_stability,
            y="infra_total",
            title="Hourly infrastructure cost stability",
            labels={"infra_total": "Hourly infrastructure cost ($)"},
            points=False,
        )
        fig_stability.update_traces(
            marker_color="#4C78A8",
            line=dict(color="#4C78A8", width=2),
            hovertemplate="Hourly infrastructure cost: $%{y:,.2f}<extra></extra>",
        )
        fig_stability.update_layout(
            template="simple_white",
            showlegend=False,
            title_font=dict(size=20),
            yaxis_title_font=dict(size=14),
            font=dict(size=13),
            margin=dict(t=60, b=40, l=30, r=30),
        )
        st.plotly_chart(fig_stability, use_container_width=True)
        st.caption(
            "This chart summarizes hourly infrastructure cost stability. "
            f"Hourly cost ranges from about ${hourly_min:,.0f} to ${hourly_max:,.0f}, "
            f"with a median of about ${hourly_median:,.0f}, indicating a relatively stable operating band."
        )

    # chart 3) requests vs cost trend
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
            name="Total requests (7d ma)",
            mode="lines+markers",
            line=dict(color="#4C78A8", width=3),
            hovertemplate="Day: %{x}<br>Requests (7d ma): %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_growth.add_trace(
        go.Scatter(
            x=growth_daily["day"],
            y=growth_daily["total_cost_ma7"],
            name="Total daily infrastructure cost (7d ma)",
            mode="lines+markers",
            line=dict(color="#E45756", width=3),
            hovertemplate="Day: %{x}<br>Total daily infrastructure cost (7d ma): $%{y:,.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig_growth.update_layout(
        template="simple_white",
        title="Requests vs total cost over time",
        title_font=dict(size=20),
        font=dict(size=13),
        legend_title_text="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        height=500,
        margin=dict(t=70, b=40, l=30, r=30),
    )
    fig_growth.update_xaxes(title_text="Time")
    fig_growth.update_yaxes(title_text="Total requests", secondary_y=False)
    fig_growth.update_yaxes(
        title_text="Total daily infrastructure cost ($)",
        tickprefix="$",
        secondary_y=True,
    )
    st.plotly_chart(fig_growth, use_container_width=True)
    corr_daily = growth_daily[["total_requests", "total_daily_cost"]].dropna()
    pearson_r_daily = (
        float(corr_daily["total_requests"].corr(corr_daily["total_daily_cost"]))
        if len(corr_daily) > 1
        else 0.0
    )
    st.caption(
        "This chart shows day-level aggregation with 7-day moving averages for total requests and total daily infrastructure cost on two vertical axes. "
        f"Using raw daily data (without moving-average smoothing), requests and total daily infrastructure cost show a strong positive correlation (Pearson r = {pearson_r_daily:.2f})."
    )

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
        labels=dict(x="hour of day", y="day of week", color="mean infra cost ($)"),
        title="mean infrastructure cost by day of week and hour",
        color_continuous_scale=coolwarm_scale,
        aspect="auto",
    )
    fig_heatmap.update_traces(
        hovertemplate=(
            "day: %{y}<br>"
            "hour: %{x}:00<br>"
            "mean infra cost: $%{z:,.2f}<extra></extra>"
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
        page_title="Tavily dashboard",
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
        "pages",
        ["product analysis", "infrastructure & cost analysis"],
    )

    if page == "product analysis":
        render_product_analysis(users, hourly_usage, research_requests)
    else:
        render_infrastructure_and_cost_analysis(
            users,
            infrastructure_costs,
            hourly_usage,
            research_requests,
        )


if __name__ == "__main__":
    main()
