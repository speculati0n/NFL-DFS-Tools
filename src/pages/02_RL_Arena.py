import streamlit as st
import pandas as pd
import numpy as np
from dfs_rl.utils.data import find_weeks, load_week_folder
from dfs_rl.arena import run_tournament
from backtesting.backtester import _find_points_col
from dfs_rl.utils.historical_outcomes import attach_historical_outcomes

import os

from dfs.constraints import DEFAULT_MIN_SPEND_PCT

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
min_salary_pct = st.sidebar.slider(
    "Min salary spend (% of cap)", 0.90, 1.00, float(os.getenv("MIN_SALARY_PCT", DEFAULT_MIN_SPEND_PCT)), 0.005
)
seed = st.sidebar.number_input("Seed", value=0, step=1)
np.random.seed(int(seed))

if st.button("Run Arena"):
    with st.spinner("Generating lineups..."):
        df = run_tournament(
            bundle["projections"],
            n_lineups_per_agent=n,
            train_pg=True,
            min_salary_pct=min_salary_pct,
            seed=int(seed),
        )
        st.success("Done")
    if bundle["contest_files"]:
        board = pd.read_csv(bundle["contest_files"][0])
        pts_col = _find_points_col(board)
        if pts_col and pts_col in df.columns:
            scores = df[pts_col]
            s = board.sort_values(pts_col, ascending=False)[pts_col]
            arr = scores.fillna(0).to_numpy()
            ranks = np.searchsorted(-s.to_numpy(), -arr, side="left") + 1
            df["contest_rank"] = ranks
            df["field_size"] = len(s)
            if "amount_won" in board.columns:
                payouts = board[["rank", "amount_won"]].drop_duplicates("rank")
                df = df.merge(payouts, left_on="contest_rank", right_on="rank", how="left")

    selected_date_iso = choice[-10:]
    HIST_ROOT = "data/historical"
    df = attach_historical_outcomes(
        generated_df=df,
        date_like=selected_date_iso,
        base_dir=HIST_ROOT,
    )

    st.subheader(f"Generated lineups (≥{min_salary_pct:.0%} cap spend)")
    base_cols = [
        "agent",
        "iteration",
        "salary",
        "projections_proj",
        "projections_actpts",
        "contest_rank",
        "amount_won",
        "field_size",
        "entries_per_user",
        "entry_fee",
        "contest_name",
        "matches_found",
        "QB",
        "RB1",
        "RB2",
        "WR1",
        "WR2",
        "WR3",
        "TE",
        "FLEX",
        "DST",
    ]
    extra_cols = [
        "stack_bucket",
        "double_te",
        "flex_pos",
        "dst_conflicts",
        "reward",
    ]
    cols_to_show = base_cols + [c for c in extra_cols if c in df.columns]
    st.dataframe(df[[c for c in cols_to_show if c in df.columns]].head(50), width="stretch")
    st.download_button(
        "Download all lineups (CSV)",
        df[cols_to_show].to_csv(index=False).encode(),
        file_name="arena_lineups.csv",
    )
