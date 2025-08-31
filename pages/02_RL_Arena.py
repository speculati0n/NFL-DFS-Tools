import streamlit as st
import pandas as pd
from dfs_rl.utils.data import find_weeks, load_week_folder
from dfs_rl.arena import run_tournament

st.set_page_config(page_title="RL Arena", layout="wide")
st.title("\ud83c\udfc8 RL Arena \u2014 Multi-Agent Lineup Tournament")

weeks = find_weeks()
if not weeks:
    st.warning("No historical weeks found under data/historical/YYYY/YYYY-MM-DD")
    st.stop()

label_to_path = {lab: path for lab, path in weeks}
choice = st.selectbox("Select week:", list(label_to_path.keys()))
week_dir = label_to_path[choice]
bundle = load_week_folder(week_dir)

st.caption(f"Players in pool: {len(bundle['projections'])}")
n = st.slider("Lineups per agent", 20, 300, 150, 10)

if st.button("Run Arena"):
    with st.spinner("Generating lineups..."):
        df = run_tournament(bundle["projections"], n_lineups_per_agent=n, train_pg=True)
    st.success("Done")
    st.dataframe(df.head(50), use_container_width=True)
    st.download_button("Download all lineups (CSV)", df.to_csv(index=False).encode(), file_name="arena_lineups.csv")
