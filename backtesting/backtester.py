import numpy as np
import pandas as pd
from typing import Dict, Optional

from dfs_rl.utils.data import load_week_folder
from dfs_rl.arena import run_tournament

POINTS_COLS = ["score","dk_points","lineup_points","points","FPTS","total_points"]

def _find_points_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c.lower() in [x.lower() for x in POINTS_COLS]:
            return c
    return None

def backtest_week(week_dir: str, n_lineups_per_agent: int = 150) -> Dict[str, pd.DataFrame]:
    bundle = load_week_folder(week_dir)
    pool = bundle["projections"].copy()
    gen = run_tournament(pool, n_lineups_per_agent=n_lineups_per_agent, train_pg=False)

    # If we have a contest file with lineup points, compare
    scored = None
    if bundle["contest_files"]:
        board = pd.read_csv(bundle["contest_files"][0])
        pts_col = _find_points_col(board)
        if pts_col is not None:
            s = pd.to_numeric(board[pts_col], errors="coerce").dropna().sort_values(ascending=False).reset_index(drop=True)
            gen = gen.copy()
            scores = gen["actual"] if "actual" in gen.columns else gen["proj"]
            arr = scores.fillna(0).to_numpy()
            ranks = np.searchsorted(-s.to_numpy(), -arr, side="left") + 1
            gen["contest_rank"] = ranks
            gen["field_size"] = len(s)
            if "amount_won" in board.columns:
                payouts = board[["rank", "amount_won"]].drop_duplicates("rank")
                gen = gen.merge(payouts, left_on="contest_rank", right_on="rank", how="left").drop(columns=["rank"])
                gen["amount_won"] = pd.to_numeric(gen["amount_won"], errors="coerce").fillna(0.0)
            gen["percentile"] = gen["contest_rank"] / gen["field_size"]
            scored = gen
        else:
            gen = gen.copy()
            scores = gen["actual"] if "actual" in gen.columns else gen["proj"]
            gen["contest_rank"] = scores.rank(ascending=False, method="min")
            gen["percentile"] = gen["contest_rank"] / len(gen)
            scored = gen

    return {"generated": gen, "scored": scored}
