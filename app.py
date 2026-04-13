import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data
def load_dataset(filename: str = "hourly_usage.csv") -> pd.DataFrame:
    """Load a CSV dataset from zip/app folder/parent folder."""
    app_dir = Path(__file__).resolve().parent
    parent_dir = app_dir.parent
    zip_path = app_dir / "data.zip"

    if zip_path.is_file():
        with zipfile.ZipFile(zip_path, "r") as zf:
            if filename in zf.namelist():
                with zf.open(filename) as dataset_file:
                    return pd.read_csv(dataset_file)

    for candidate in (app_dir / filename, parent_dir / filename):
        if candidate.is_file():
            return pd.read_csv(candidate)

    raise FileNotFoundError(
        f"Could not find '{filename}'. Tried: '{zip_path}' (zip member), "
        f"'{app_dir / filename}', '{parent_dir / filename}'."
    )


def render_product_analysis_and_cost() -> None:
    st.header("Product Analysis and Cost")
    df = load_dataset()
    st.write("Dataset loaded successfully.")
    st.dataframe(df.head(20), use_container_width=True)


def render_infrastructure_and_cost_analysis() -> None:
    st.header("Infrastructure & Cost Analysis")
    df = load_dataset()
    st.write("Dataset loaded successfully.")
    st.dataframe(df.head(20), use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Tavily Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("Tavily Dashboard")

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
