# Tavily_dashborad

Interactive **Streamlit** dashboard for Dan Benbenisti’s Data Analyst home assignment ([Tavily](https://tavily.com/)).

## Run locally

```bash
cd Tavily_dashborad
pip install -r requirements.txt
streamlit run app.py
```

## Data

The app loads CSVs in this order:

1. **`data.zip`** in the same folder as `app.py` (matches GitHub / Streamlit Cloud layout), or  
2. Loose **`*.csv`** files next to `app.py`, or  
3. Loose **`*.csv`** files in the **parent** folder (handy when the repo sits inside the assignment directory that already contains the four tables).

Expected files: `hourly_usage.csv`, `infrastructure_costs.csv`, `research_requests.csv`, `users.csv`.

## Pages

- **Product Analysis** — placeholder (content cleared)  
- **Infrastructure & Cost Analysis** — hourly spend by component  
