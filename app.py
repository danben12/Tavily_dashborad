import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


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
    lifecycle["retained_30d"] = (
        (lifecycle["last_event_ts"] - lifecycle["first_event_ts"]).dt.days >= 30
    )
    return lifecycle, int(new_users["user_id"].nunique())


def _retention_by_segment(lifecycle: pd.DataFrame) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame(columns=["segment", "retention_rate"])
    out = lifecycle.copy()
    out["segment"] = out["first_source"].map(
        {"query": "First Action = Query", "research": "First Action = Research"}
    )
    out = (
        out.groupby("segment", as_index=False)["retained_30d"]
        .mean()
        .rename(columns={"retained_30d": "retention_rate"})
    )
    return out


def render_product_analysis_and_cost(
    users: pd.DataFrame, hourly_usage: pd.DataFrame, research_requests: pd.DataFrame
) -> None:
    lifecycle, joined_users_count = _build_hourly_lifecycle(users, hourly_usage)
    if lifecycle.empty:
        st.error("Could not build lifecycle table from users and hourly usage.")
        return
    research_first = lifecycle["first_source"].eq("research")
    research_first_count = int(research_first.sum())
    active_joined_users_count = int(lifecycle["user_id"].nunique())
    acquisition_pct = (
        100.0 * research_first_count / joined_users_count if joined_users_count > 0 else 0.0
    )
    if research_first_count == 0:
        churn_pct = 0.0
    else:
        retention_pct = 100.0 * lifecycle.loc[research_first, "retained_30d"].mean()
        churn_pct = 100.0 - retention_pct

    st.metric(
        "Research API Acquisition (New Users)",
        f"{acquisition_pct:.1f}%",
        delta=f"-{churn_pct:.1f}% churn (Research-first users)",
        delta_color="inverse",
        help=(
            "Acquisition = users with first hourly request_type = research divided by all users "
            "created on/after 2025-11-01. Churn = 100% - retention for research-first users, "
            "where retention means last hourly activity is at least 30 days after first hourly activity."
        ),
    )
    st.caption(
        f"Joined users since Nov 1: {joined_users_count:,} | Active joined users: "
        f"{active_joined_users_count:,} | Research-first users: {research_first_count:,}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.caption(
            "Calculation: users are split by first-ever action (Query vs Research), then "
            "retention = share with any activity 30+ days after first activity."
        )
        retention_df = _retention_by_segment(lifecycle)
        if retention_df.empty:
            st.warning("Not enough data to calculate retention by first action.")
        else:
            retention_df["retention_rate"] = retention_df["retention_rate"] * 100.0
            fig_retention = px.bar(
                retention_df,
                x="segment",
                y="retention_rate",
                title="Retention Rate by First Action",
                labels={"segment": "First Action", "retention_rate": "Retention Rate (%)"},
                text=retention_df["retention_rate"].map(lambda x: f"{x:.1f}%"),
            )
            fig_retention.update_layout(template="simple_white")
            st.plotly_chart(fig_retention, use_container_width=True)

    with col2:
        st.caption(
            "Calculation: average `response_time_seconds` grouped by `model` for mini and pro only."
        )
        rr = _lowercase_columns(research_requests)
        if not {"model", "response_time_seconds"}.issubset(rr.columns):
            st.warning("Missing `model` or `response_time_seconds` in research data.")
        else:
            rr["model"] = rr["model"].astype(str).str.lower().str.strip()
            rr["response_time_seconds"] = pd.to_numeric(
                rr["response_time_seconds"], errors="coerce"
            )
            model_latency = (
                rr[rr["model"].isin(["mini", "pro"])]
                .dropna(subset=["response_time_seconds"])
                .groupby("model", as_index=False)["response_time_seconds"]
                .mean()
                .rename(columns={"response_time_seconds": "avg_response_time_seconds"})
            )
            if model_latency.empty:
                st.warning("No usable Mini/Pro response-time data found.")
            else:
                fig_latency = px.bar(
                    model_latency,
                    x="avg_response_time_seconds",
                    y="model",
                    orientation="h",
                    title="Average Response Time by Model (Mini vs Pro)",
                    labels={
                        "avg_response_time_seconds": "Average Response Time (seconds)",
                        "model": "Model",
                    },
                    text=model_latency["avg_response_time_seconds"].map(lambda x: f"{x:.2f}s"),
                )
                fig_latency.update_layout(template="simple_white")
                st.plotly_chart(fig_latency, use_container_width=True)

    # Pareto: concentration of research traffic.
    st.caption(
        "Calculation: count research requests per user, sort descending, then plot cumulative % users "
        "vs cumulative % total requests. Dashed lines mark 5% users and their request share."
    )
    rr = _lowercase_columns(research_requests)
    if "user_id" not in rr.columns:
        st.warning("Missing `user_id` in research requests for Pareto chart.")
        return
    rr["user_id"] = pd.to_numeric(rr["user_id"], errors="coerce")
    rr = rr.dropna(subset=["user_id"]).copy()
    rr["user_id"] = rr["user_id"].astype(int)
    counts = rr.groupby("user_id").size().sort_values(ascending=False)
    if counts.empty:
        st.warning("No research requests available for Pareto chart.")
        return

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

    fig_pareto = px.line(
        pareto,
        x="cum_users_pct",
        y="cum_requests_pct",
        title="Research API Traffic Concentration (Pareto Curve)",
        labels={
            "cum_users_pct": "Cumulative % of Users",
            "cum_requests_pct": "Cumulative % of Total Requests",
        },
    )
    y_at_5 = float(
        pareto.loc[pareto["cum_users_pct"] >= 5.0, "cum_requests_pct"].head(1).fillna(0.0).iloc[0]
    )
    fig_pareto.add_vline(x=5.0, line_dash="dash", line_color="gray")
    fig_pareto.add_hline(y=y_at_5, line_dash="dash", line_color="gray")
    fig_pareto.update_layout(template="simple_white")
    st.plotly_chart(fig_pareto, use_container_width=True)


def render_infrastructure_and_cost_analysis() -> None:
    st.header("Infrastructure & Cost Analysis")
    st.write("Content placeholder")


def main() -> None:
    st.set_page_config(
        page_title="Tavily Dashboard",
        page_icon="📊",
        layout="wide",
    )

    (
        hourly_usage,
        Infrastructure_costs,
        research_requests,
        users,
    ) = load_datasets_from_zip()
    _ = (hourly_usage, Infrastructure_costs, research_requests, users)

    page = st.sidebar.radio(
        "Pages",
        ["Product Analysis", "Infrastructure & Cost Analysis"],
    )

    if page == "Product Analysis":
        render_product_analysis_and_cost(users, hourly_usage, research_requests)
    else:
        render_infrastructure_and_cost_analysis()


if __name__ == "__main__":
    main()
