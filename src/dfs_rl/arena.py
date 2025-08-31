from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

from dfs_rl.envs.dk_nfl_env import DKNFLEnv
from dfs_rl.agents.random_agent import RandomAgent
from dfs_rl.agents.pg_agent import PGAgent
from dfs_rl.agents.greedy_agent import GreedyAgent

POINTS_COLS = [
    "projections_actpts",
    "score",
    "dk_points",
    "lineup_points",
    "points",
    "FPTS",
    "total_points",
]


def _find_points_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c.lower() in [x.lower() for x in POINTS_COLS]:
            return c
    return None

def _run_agent(env: DKNFLEnv, agent, train: bool) -> Tuple[list,int,float]:
    obs, info = env.reset()
    total = 0.0
    steps = 0
    while True:
        a = agent.act(info["action_mask"])
        obs, r, done, trunc, info = env.step(a)
        total += float(r)
        steps += 1
        if done or steps > 20:
            if train and hasattr(agent, "update"):
                agent.update(total)
            return info.get("lineup_indices", []), steps, total

def run_tournament(pool: pd.DataFrame, n_lineups_per_agent: int = 150, train_pg: bool = True) -> pd.DataFrame:
    env = DKNFLEnv(pool)
    n = len(pool)
    agents = {
        "random": RandomAgent(seed=1),
        "pg": PGAgent(n_players=n, seed=2),
        "greedy": GreedyAgent(pool["projections_proj"].to_numpy()),
    }

    pts_col = _find_points_col(pool)

    rows = []
    slot_cols = ["QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST"]

    for name, agent in agents.items():
        for i in range(n_lineups_per_agent):
            idxs, steps, reward = _run_agent(env, agent, train=(train_pg and name == "pg"))
            if len(idxs) != len(slot_cols):
                # skip incomplete lineups (shouldn't happen but guard anyway)
                continue

            L = pool.iloc[idxs].copy()
            row = {"agent": name, "iteration": i, "salary": int(L["salary"].sum())}

            for slot, idx in zip(slot_cols, idxs):
                row[slot] = pool.loc[idx, "name"]

            # store projections and actual score if available
            row["projections_proj"] = float(L["projections_proj"].sum())
            if pts_col and pts_col in L.columns:
                total = float(L[pts_col].sum())
                row[pts_col] = total
                if pts_col.lower() != "score":
                    row["score"] = total
            else:
                # fall back to projection as "score" if actuals absent
                row["score"] = row["projections_proj"]

            rows.append(row)

    return pd.DataFrame(rows)
