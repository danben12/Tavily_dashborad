import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data
def load_datasets_from_zip() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required datasets from data.zip in the app directory."""
    app_dir = Path(__file__).resolve().parent
    zip_path = app_dir / "data.zip"
    required_files = (
        "hourly_usage.csv",
        "infrastructure_costs.csv",
        "research_requests.csv",
        "users.csv",
    )

    if not zip_path.is_file():
        raise FileNotFoundError(f"Could not find data archive at '{zip_path}'.")

    loaded_frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = set(zf.namelist())
        missing = [name for name in required_files if name not in members]
        if missing:
            raise FileNotFoundError(
                f"Missing required files in data.zip: {', '.join(missing)}"
            )
        for name in required_files:
            with zf.open(name) as dataset_file:
                loaded_frames[name] = pd.read_csv(dataset_file)

    return (
        loaded_frames["hourly_usage.csv"],
        loaded_frames["infrastructure_costs.csv"],
        loaded_frames["research_requests.csv"],
        loaded_frames["users.csv"],
    )


def render_product_analysis_and_cost() -> None:
    st.header("Product Analysis and Cost")
    st.write("Content placeholder")


def render_infrastructure_and_cost_analysis() -> None:
    st.header("Infrastructure & Cost Analysis")
    st.write("Content placeholder")


def main() -> None:
    st.set_page_config(
        page_title="Tavily Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("Tavily Dashboard")

    (
        hourly_usage,
        Infrastructure_costs,
        research_requests,
        users,
    ) = load_datasets_from_zip()
    _ = (hourly_usage, Infrastructure_costs, research_requests, users)

    page = st.sidebar.radio(
        "Pages",
        ["Product Analysis and Cost", "Infrastructure & Cost Analysis"],
    )

    if page == "Product Analysis and Cost":
        render_product_analysis_and_cost()
    else:
        render_infrastructure_and_cost_analysis()


if __name__ == "__main__":
    main()
