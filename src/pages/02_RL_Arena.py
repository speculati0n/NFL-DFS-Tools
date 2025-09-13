import streamlit as st
import pandas as pd
import numpy as np
import json
from dfs_rl.utils.data import find_weeks, load_week_folder
from dfs_rl.arena import run_tournament
from backtesting.backtester import _find_points_col
from dfs_rl.utils.historical_outcomes import (
    attach_historical_outcomes,
    standardize_scoreboard_cols,
)

import os

from dfs.constraints import DEFAULT_MIN_SPEND_PCT
from utils import get_config_path

st.set_page_config(page_title="RL Arena", layout="wide")


weeks = find_weeks()
if not weeks:
    st.warning("No historical data found under data/historical")
    st.stop()

label_to_key = {lab: key for lab, key in weeks}
choice = st.selectbox("Select week:", list(label_to_key.keys()))
week_key = label_to_key[choice]
bundle = load_week_folder(week_key)

cfg_path = get_config_path()
with open(cfg_path) as f:
    cfg = json.load(f)

st.caption(f"Players in pool: {len(bundle['projections'])}")
n = st.slider("Lineups per agent", 20, 300, 150, 10)
min_salary_pct = st.sidebar.slider(
    "Min salary spend (% of cap)", 0.90, 1.00, float(cfg.get("min_salary_ratio", DEFAULT_MIN_SPEND_PCT)), 0.005
)
max_double_stack_rate = st.sidebar.slider(
    "Max double stack rate", 0.0, 1.0, float(cfg.get("max_double_stack_rate", 0.35)), 0.05
)
max_opp_wr_bringback_rate = st.sidebar.slider(
    "Max opp WR bring-back rate", 0.0, 1.0, float(cfg.get("max_opp_wr_bringback_rate", 0.35)), 0.05
)
flex_mix = cfg.get("target_flex_mix", {"WR": 0.62, "RB": 0.33, "TE": 0.05})
flex_wr = st.sidebar.number_input("Target FLEX WR %", 0.0, 1.0, float(flex_mix.get("WR", 0.62)))
flex_rb = st.sidebar.number_input("Target FLEX RB %", 0.0, 1.0, float(flex_mix.get("RB", 0.33)))
flex_te = st.sidebar.number_input("Target FLEX TE %", 0.0, 1.0, float(flex_mix.get("TE", 0.05)))
rw_cfg = cfg.get("reward_weights", {})
with st.sidebar.expander("Reward Weights"):
    new_rw = {}
    for k, v in rw_cfg.items():
        new_rw[k] = st.number_input(k, value=float(v))
seed = st.sidebar.number_input("Seed", value=0, step=1)
np.random.seed(int(seed))

if st.button("Run Arena"):
    cfg.update(
        {
            "min_salary_ratio": float(min_salary_pct),
            "max_double_stack_rate": float(max_double_stack_rate),
            "max_opp_wr_bringback_rate": float(max_opp_wr_bringback_rate),
            "target_flex_mix": {"WR": flex_wr, "RB": flex_rb, "TE": flex_te},
            "reward_weights": new_rw,
        }
    )
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    with st.spinner("Generating lineups..."):
        df = run_tournament(
            bundle["projections"],
            n_lineups_per_agent=n,
            train_pg=True,
            seed=int(seed),
            cfg=cfg,
            save_to_simulator=True,
            sim_out_dir="uploads2/saved_lineups",
            tag_source="agent",
        )
        st.success("Done")
    if bundle["contest_files"]:
        board = pd.read_csv(bundle["contest_files"][0])
        board = standardize_scoreboard_cols(board)
        pts_col_board = _find_points_col(board)
        pts_col_df = _find_points_col(df)
        if pts_col_board and pts_col_df:
            scores = df[pts_col_df]
            s = board.sort_values(pts_col_board, ascending=False)[pts_col_board]
            arr = scores.fillna(0).to_numpy()
            ranks = np.searchsorted(-s.to_numpy(), -arr, side="left") + 1
            df["contest_rank"] = ranks
            field_size_val = int(board["field_size"].iloc[0]) if "field_size" in board.columns else len(s)
            df["field_size"] = field_size_val
            if "entries_per_user" in board.columns:
                df["entries_per_user"] = int(board["entries_per_user"].iloc[0])
            if "entry_fee" in board.columns:
                df["entry_fee"] = board["entry_fee"].iloc[0]
            if "contest_name" in board.columns:
                df["contest_name"] = board["contest_name"].iloc[0]
            if "amount_won" in board.columns:
                payouts = board[["rank", "amount_won"]].drop_duplicates("rank")
                df = df.merge(payouts, left_on="contest_rank", right_on="rank", how="left")
                na_mask = df["amount_won"].isna()
                if na_mask.any() and "score" in board.columns:
                    sb = board.sort_values("score", ascending=False)[["score", "amount_won"]]
                    b_scores = sb["score"].to_numpy()
                    b_payouts = sb["amount_won"].to_numpy()
                    lineup_scores = df.loc[na_mask, pts_col_df].fillna(0).to_numpy()
                    idx = np.searchsorted(-b_scores, -lineup_scores, side="right") - 1
                    idx[idx < 0] = 0
                    df.loc[na_mask, "amount_won"] = b_payouts[idx]
                    df.loc[na_mask, "field_size"] = field_size_val
                    if "entries_per_user" in board.columns:
                        df.loc[na_mask, "entries_per_user"] = int(board["entries_per_user"].iloc[0])
                    if "entry_fee" in board.columns:
                        df.loc[na_mask, "entry_fee"] = board["entry_fee"].iloc[0]
                    if "contest_name" in board.columns:
                        df.loc[na_mask, "contest_name"] = board["contest_name"].iloc[0]

    selected_date_iso = choice
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
        "flex_pos",
        "bringback_type",
        "any_vs_dst_count",
        "reward",
    ]
    cols_to_show = base_cols + [c for c in extra_cols if c in df.columns]
    st.dataframe(df[[c for c in cols_to_show if c in df.columns]].head(50), width="stretch")
    st.download_button(
        "Download all lineups (CSV)",
        df[cols_to_show].to_csv(index=False).encode(),
        file_name="arena_lineups.csv",
    )
