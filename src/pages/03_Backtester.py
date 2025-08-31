import streamlit as st
from dfs_rl.utils.data import find_weeks
from backtesting.backtester import backtest_week

st.set_page_config(page_title="Backtester", layout="wide")


weeks = find_weeks()
if not weeks:
    st.warning("No historical weeks found under data/historical/YYYY/YYYY-MM-DD")
    st.stop()

label_to_path = {lab: path for lab, path in weeks}
choice = st.selectbox("Select week:", list(label_to_path.keys()))
week_dir = label_to_path[choice]
n = st.slider("Lineups per agent", 20, 300, 150, 10)

if st.button("Run Backtest"):
    with st.spinner("Backtesting..."):
        out = backtest_week(week_dir, n_lineups_per_agent=n)
    st.success("Done")
    st.subheader("Generated lineups")
    st.dataframe(out["generated"].head(50), use_container_width=True)
    if out["scored"] is not None:
        st.subheader("Scored vs contest (percentile & sim_rank by projections)")
        st.dataframe(out["scored"].head(50), use_container_width=True)
