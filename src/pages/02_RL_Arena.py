import streamlit as st
import pandas as pd
import numpy as np
from dfs_rl.utils.data import find_weeks, load_week_folder
from dfs_rl.arena import run_tournament
from backtesting.backtester import _find_points_col

st.set_page_config(page_title="RL Arena", layout="wide")


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
    if bundle["contest_files"]:
        board = pd.read_csv(bundle["contest_files"][0])
        # Prefer actual scores when available, otherwise fall back to projections
        scores = df["actual"] if "actual" in df.columns else df.get("proj")
        if scores is None:
            scores = pd.Series(np.zeros(len(df)))

        pts_col = _find_points_col(board)
        if pts_col is not None:
            s = pd.to_numeric(board[pts_col], errors="coerce").dropna()
            s = s.sort_values(ascending=False).reset_index(drop=True)
            arr = scores.fillna(0).to_numpy()
            ranks = np.searchsorted(-s.to_numpy(), -arr, side="left") + 1
            df["contest_rank"] = ranks
            df["field_size"] = len(s)
            if "amount_won" in board.columns:
                payouts = board[["rank", "amount_won"]].drop_duplicates("rank")
                df = df.merge(
                    payouts, left_on="contest_rank", right_on="rank", how="left"
                ).drop(columns=["rank"])
                df["amount_won"] = pd.to_numeric(
                    df["amount_won"], errors="coerce"
                ).fillna(0.0)
        else:
            df["contest_rank"] = scores.rank(ascending=False, method="min")
            df["field_size"] = len(scores)
    st.dataframe(df.head(50), use_container_width=True)
    st.download_button(
        "Download all lineups (CSV)",
        df.to_csv(index=False).encode(),
        file_name="arena_lineups.csv",
    )
