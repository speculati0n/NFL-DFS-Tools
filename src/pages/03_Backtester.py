import streamlit as st
from dfs_rl.utils.data import find_weeks
from backtesting.backtester import backtest_week
from dfs_rl.utils.historical_outcomes import attach_historical_outcomes

import os

from dfs.constraints import DEFAULT_MIN_SPEND_PCT
from utils import get_config_path
import json

st.set_page_config(page_title="Backtester", layout="wide")


weeks = find_weeks()
if not weeks:
    st.warning("No historical data found under data/historical")
    st.stop()

label_to_key = {lab: key for lab, key in weeks}
choice = st.selectbox("Select week:", list(label_to_key.keys()))
week_key = label_to_key[choice]
n = st.slider("Lineups per agent", 20, 300, 150, 10)
cfg_path = get_config_path()
with open(cfg_path) as f:
    cfg = json.load(f)
min_salary_pct = st.sidebar.slider(
    "Min salary spend (% of cap)", 0.90, 1.00, float(cfg.get("min_salary_ratio", DEFAULT_MIN_SPEND_PCT)), 0.005
)

if st.button("Run Backtest"):
    cfg["min_salary_ratio"] = float(min_salary_pct)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    with st.spinner("Backtesting..."):
        out = backtest_week(week_key, n_lineups_per_agent=n, min_salary_pct=min_salary_pct)
    st.success("Done")
    selected_date_iso = choice
    HIST_ROOT = "data/historical"
    out["generated"] = attach_historical_outcomes(
        generated_df=out["generated"],
        date_like=selected_date_iso,
        base_dir=HIST_ROOT,
    )
    if out["scored"] is not None:
        out["scored"] = attach_historical_outcomes(
            generated_df=out["scored"],
            date_like=selected_date_iso,
            base_dir=HIST_ROOT,
        )

    st.subheader(f"Generated lineups (≥{min_salary_pct:.0%} cap spend)")
    cols_to_show = [
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
        "stack_bucket",
        "flex_pos",
        "bringback_type",
        "any_vs_dst_count",
        "reward",
    ]
    gen = out["generated"][ [c for c in cols_to_show if c in out["generated"].columns] ]
    st.dataframe(gen.head(50), width="stretch")
    st.download_button(
        "Download all lineups (CSV)",
        gen.to_csv(index=False).encode(),
        file_name="backtester_lineups.csv",
    )
    if out["scored"] is not None:
        st.subheader("Scored vs contest (rank & winnings)")
        scored = out["scored"][ [c for c in cols_to_show if c in out["scored"].columns] ]
        st.dataframe(scored.head(50), width="stretch")
        st.download_button(
            "Download scored lineups (CSV)",
            scored.to_csv(index=False).encode(),
            file_name="backtester_scored.csv",
        )
