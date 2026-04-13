from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data
def load_dataset(filename: str = "hourly_usage.csv") -> pd.DataFrame:
    """Load a CSV dataset from the repository root."""
    repo_root = Path(__file__).resolve().parent.parent
    dataset_path = repo_root / filename
    return pd.read_csv(dataset_path)


def render_product_analysis_and_cost() -> None:
    st.header("Product Analysis and Cost")
    df = load_dataset()
    st.write("Dataset loaded from repository root.")
    st.dataframe(df.head(20), use_container_width=True)


def render_infrastructure_and_cost_analysis() -> None:
    st.header("Infrastructure & Cost Analysis")
    df = load_dataset()
    st.write("Dataset loaded from repository root.")
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
