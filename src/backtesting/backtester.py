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
        import pandas as pd
        board = pd.read_csv(bundle["contest_files"][0])
        pts_col = _find_points_col(board)
        if pts_col is not None:
            # percentile rank vs field by projections (or points if you wish to map)
            s = pd.to_numeric(board[pts_col], errors="coerce").dropna().sort_values(ascending=False).reset_index(drop=True)
            gen = gen.copy()
            # naive: treat our 'proj' as comparable to points for ranking ballpark
            gen["sim_rank"] = gen["proj"].rank(ascending=False, method="min")
            gen["field_size"] = len(s)
            gen["percentile"] = gen["sim_rank"] / gen["field_size"]
            scored = gen
        else:
            # fallback: rank just within our generated set
            gen = gen.copy()
            gen["sim_rank"] = gen["proj"].rank(ascending=False, method="min")
            gen["percentile"] = gen["sim_rank"] / len(gen)
            scored = gen

    return {"generated": gen, "scored": scored}
