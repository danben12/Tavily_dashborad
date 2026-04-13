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


def _build_events(hourly_usage: pd.DataFrame, research_requests: pd.DataFrame) -> pd.DataFrame:
    h = _lowercase_columns(hourly_usage)
    r = _lowercase_columns(research_requests)

    required_h = {"user_id", "hour"}
    required_r = {"user_id", "timestamp"}
    if not required_h.issubset(h.columns) or not required_r.issubset(r.columns):
        return pd.DataFrame(columns=["user_id", "event_ts", "source"])

    h_events = h[["user_id", "hour"]].copy()
    h_events["event_ts"] = pd.to_datetime(h_events["hour"], errors="coerce", utc=True)
    h_events["source"] = "query"

    r_events = r[["user_id", "timestamp"]].copy()
    r_events["event_ts"] = pd.to_datetime(r_events["timestamp"], errors="coerce", utc=True)
    r_events["source"] = "research"

    events = pd.concat(
        [
            h_events[["user_id", "event_ts", "source"]],
            r_events[["user_id", "event_ts", "source"]],
        ],
        ignore_index=True,
    )
    events["user_id"] = pd.to_numeric(events["user_id"], errors="coerce")
    events = events.dropna(subset=["user_id", "event_ts"]).copy()
    events["user_id"] = events["user_id"].astype(int)
    return events.sort_values(["user_id", "event_ts", "source"]).reset_index(drop=True)


def _retention_by_cohort(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["cohort", "retention_rate"])

    first_actions = (
        events.groupby("user_id", as_index=False)
        .first()[["user_id", "event_ts", "source"]]
        .rename(columns={"event_ts": "first_event_ts", "source": "first_source"})
    )
    last_activity = (
        events.groupby("user_id", as_index=False)["event_ts"]
        .max()
        .rename(columns={"event_ts": "last_event_ts"})
    )

    user_lifecycle = first_actions.merge(last_activity, on="user_id", how="inner")
    user_lifecycle["retained_30d"] = (
        user_lifecycle["last_event_ts"]
        >= user_lifecycle["first_event_ts"] + pd.Timedelta(days=30)
    )
    user_lifecycle["cohort"] = user_lifecycle["first_source"].map(
        {"query": "First Action = Query", "research": "First Action = Research"}
    )

    out = (
        user_lifecycle.groupby("cohort", as_index=False)["retained_30d"]
        .mean()
        .rename(columns={"retained_30d": "retention_rate"})
    )
    return out


def render_product_analysis_and_cost(
    users: pd.DataFrame, hourly_usage: pd.DataFrame, research_requests: pd.DataFrame
) -> None:
    st.header("Q1: Research API - Acquisition, Retention & Usage Concentration")

    users_l = _lowercase_columns(users)
    if not {"user_id", "created_at"}.issubset(users_l.columns):
        st.error("Missing required columns in users dataset (`user_id`, `created_at`).")
        return

    events = _build_events(hourly_usage, research_requests)
    if events.empty:
        st.error("Could not build events from usage datasets.")
        return

    users_l["created_at"] = pd.to_datetime(users_l["created_at"], errors="coerce", utc=True)
    users_l["user_id"] = pd.to_numeric(users_l["user_id"], errors="coerce")
    users_l = users_l.dropna(subset=["user_id", "created_at"]).copy()
    users_l["user_id"] = users_l["user_id"].astype(int)

    # Only consider activity on/after account creation for lifecycle metrics.
    events = events.merge(users_l[["user_id", "created_at"]], on="user_id", how="inner")
    events = events.loc[events["event_ts"] >= events["created_at"], ["user_id", "event_ts", "source"]]
    if events.empty:
        st.error("No valid activity after account creation was found.")
        return
    events = events.sort_values(["user_id", "event_ts", "source"]).reset_index(drop=True)

    first_actions = (
        events.groupby("user_id", as_index=False)
        .first()[["user_id", "event_ts", "source"]]
        .rename(columns={"event_ts": "first_event_ts", "source": "first_source"})
    )
    last_activity = (
        events.groupby("user_id", as_index=False)["event_ts"]
        .max()
        .rename(columns={"event_ts": "last_event_ts"})
    )
    lifecycle = first_actions.merge(last_activity, on="user_id", how="inner")
    lifecycle["retained_30d"] = (
        lifecycle["last_event_ts"] >= lifecycle["first_event_ts"] + pd.Timedelta(days=30)
    )

    # KPI 1: Acquisition + churn for users first acquired by Research API.
    join_cutoff = pd.Timestamp("2025-11-01", tz="UTC")
    new_users = users_l.loc[users_l["created_at"] > join_cutoff, ["user_id"]].drop_duplicates()
    new_lifecycle = new_users.merge(lifecycle, on="user_id", how="inner")

    if len(new_lifecycle) == 0:
        acquisition_pct = 0.0
        churn_pct = 0.0
    else:
        research_first = new_lifecycle["first_source"].eq("research")
        acquisition_pct = 100.0 * research_first.mean()
        research_first_cohort = new_lifecycle.loc[research_first]
        if len(research_first_cohort) == 0:
            churn_pct = 0.0
        else:
            retention_pct = 100.0 * research_first_cohort["retained_30d"].mean()
            churn_pct = 100.0 - retention_pct

    st.metric(
        "Research API Acquisition (New Users)",
        f"{acquisition_pct:.1f}%",
        delta=f"-{churn_pct:.1f}% churn (Research-first cohort)",
        delta_color="inverse",
    )

    col1, col2 = st.columns(2)

    with col1:
        retention_df = _retention_by_cohort(events)
        if retention_df.empty:
            st.warning("Not enough data to calculate retention cohorts.")
        else:
            retention_df["retention_rate"] = retention_df["retention_rate"] * 100.0
            fig_retention = px.bar(
                retention_df,
                x="cohort",
                y="retention_rate",
                title="Retention Rate by First Action Cohort",
                labels={"cohort": "Cohort", "retention_rate": "Retention Rate (%)"},
                text=retention_df["retention_rate"].map(lambda x: f"{x:.1f}%"),
            )
            fig_retention.update_layout(template="simple_white")
            st.plotly_chart(fig_retention, use_container_width=True)

    with col2:
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
