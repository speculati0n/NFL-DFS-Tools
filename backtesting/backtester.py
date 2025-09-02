import numpy as np
import pandas as pd
from typing import Dict, Optional

from dfs_rl.utils.data import load_week_folder
from dfs_rl.arena import run_tournament
from dfs_rl.utils.historical_outcomes import standardize_scoreboard_cols

# Search order: first actual contest scores, then projection columns used when
# no real scores are available.
POINTS_COLS = [
    "score",            # canonical contest score
    "dk_points",        # DraftKings' exported score column
    "FPTS",             # common shorthand
    "points",
    "lineup_points",
    "total_points",
    "projections_actpts",  # fallback to historical actual points projections
    "projections_proj",    # final fallback to projected points
]


def _find_points_col(df: pd.DataFrame) -> Optional[str]:
    """Return the first matching points column in :data:`POINTS_COLS`.

    The search prioritizes actual contest scores (``score``, ``dk_points``,
    ``FPTS`` and similar) before falling back to projection columns
    (``projections_actpts`` then ``projections_proj``).  Matching is
    case-insensitive and returns the original column name from *df*.
    """

    lower_map = {c.lower(): c for c in df.columns}
    for name in POINTS_COLS:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None

def backtest_week(week_dir: str, n_lineups_per_agent: int = 150) -> Dict[str, pd.DataFrame]:
    bundle = load_week_folder(week_dir)
    pool = bundle["projections"].copy()
    gen = run_tournament(pool, n_lineups_per_agent=n_lineups_per_agent, train_pg=False)

    # If we have a contest file with lineup points, compare
    scored = None
    if bundle["contest_files"]:
        board = pd.read_csv(bundle["contest_files"][0])
        board = standardize_scoreboard_cols(board)
        pts_col_board = _find_points_col(board)
        pts_col_gen = _find_points_col(gen)
        if pts_col_board and pts_col_gen:
            scores = gen[pts_col_gen]
            s = board.sort_values(pts_col_board, ascending=False)[pts_col_board]
            arr = scores.fillna(0).to_numpy()
            ranks = np.searchsorted(-s.to_numpy(), -arr, side="left") + 1
            gen["contest_rank"] = ranks
            field_size_val = int(board["field_size"].iloc[0]) if "field_size" in board.columns else len(s)
            gen["field_size"] = field_size_val
            if "entries_per_user" in board.columns:
                gen["entries_per_user"] = int(board["entries_per_user"].iloc[0])
            if "entry_fee" in board.columns:
                gen["entry_fee"] = board["entry_fee"].iloc[0]
            if "contest_name" in board.columns:
                gen["contest_name"] = board["contest_name"].iloc[0]
            if "amount_won" in board.columns:
                payouts = board[["rank", "amount_won"]].drop_duplicates("rank")
                gen = gen.merge(payouts, left_on="contest_rank", right_on="rank", how="left")
                na_mask = gen["amount_won"].isna()
                if na_mask.any() and "score" in board.columns:
                    sb = board.sort_values("score", ascending=False)[["score", "amount_won"]]
                    b_scores = sb["score"].to_numpy()
                    b_payouts = sb["amount_won"].to_numpy()
                    lineup_scores = gen.loc[na_mask, pts_col_gen].fillna(0).to_numpy()
                    idx = np.searchsorted(-b_scores, -lineup_scores, side="right") - 1
                    idx[idx < 0] = 0
                    gen.loc[na_mask, "amount_won"] = b_payouts[idx]
                    gen.loc[na_mask, "field_size"] = field_size_val
                    if "entries_per_user" in board.columns:
                        gen.loc[na_mask, "entries_per_user"] = int(board["entries_per_user"].iloc[0])
                    if "entry_fee" in board.columns:
                        gen.loc[na_mask, "entry_fee"] = board["entry_fee"].iloc[0]
                    if "contest_name" in board.columns:
                        gen.loc[na_mask, "contest_name"] = board["contest_name"].iloc[0]
            gen["percentile"] = gen["contest_rank"] / gen["field_size"]
            scored = gen
        else:
            gen = gen.copy()
            if pts_col_gen:
                gen["contest_rank"] = gen[pts_col_gen].rank(ascending=False, method="min")
                gen["percentile"] = gen["contest_rank"] / len(gen)
            scored = gen

    return {"generated": gen, "scored": scored}
