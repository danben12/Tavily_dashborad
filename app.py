import zipfile
from pathlib import Path

import pandas as pd
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


def render_product_analysis_and_cost() -> None:
    st.header("Product Analysis")
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
        render_product_analysis_and_cost()
    else:
        render_infrastructure_and_cost_analysis()


if __name__ == "__main__":
    main()
