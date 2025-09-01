import streamlit as st
from dfs_rl.utils.data import find_weeks
from backtesting.backtester import backtest_week

import os

from dfs.constraints import DEFAULT_MIN_SPEND_PCT

st.set_page_config(page_title="Backtester", layout="wide")


weeks = find_weeks()
if not weeks:
    st.warning("No historical weeks found under data/historical/YYYY/YYYY-MM-DD")
    st.stop()

label_to_path = {lab: path for lab, path in weeks}
choice = st.selectbox("Select week:", list(label_to_path.keys()))
week_dir = label_to_path[choice]
n = st.slider("Lineups per agent", 20, 300, 150, 10)
min_salary_pct = st.sidebar.slider(
    "Min salary spend (% of cap)", 0.90, 1.00, float(os.getenv("MIN_SALARY_PCT", DEFAULT_MIN_SPEND_PCT)), 0.005
)

if st.button("Run Backtest"):
    with st.spinner("Backtesting..."):
        out = backtest_week(week_dir, n_lineups_per_agent=n, min_salary_pct=min_salary_pct)
    st.success("Done")
    st.subheader(f"Generated lineups (≥{min_salary_pct:.0%} cap spend)")
    st.dataframe(out["generated"].head(50), width="stretch")
    if out["scored"] is not None:
        st.subheader("Scored vs contest (rank & winnings)")
        st.dataframe(out["scored"].head(50), width="stretch")
