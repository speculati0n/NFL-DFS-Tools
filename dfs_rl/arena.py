from typing import List, Tuple
import numpy as np
import pandas as pd

from dfs_rl.envs.dk_nfl_env import DKNFLEnv
from dfs_rl.agents.random_agent import RandomAgent
from dfs_rl.agents.pg_agent import PGAgent

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
        "pg": PGAgent(n_players=n, seed=2)
    }
    rows = []
    for name, agent in agents.items():
        for i in range(n_lineups_per_agent):
            idxs, steps, reward = _run_agent(env, agent, train=(train_pg and name=="pg"))
            L = pool.iloc[idxs].copy()
            rows.append({
                "agent": name,
                "lineup_idx": i,
                "salary": int(L["salary"].sum()),
                "proj": float(L["projections_proj"].sum()),
                "actual": float(L["projections_actpts"].sum()) if "projections_actpts" in L.columns else np.nan,
                "players": "|".join(L["name"].tolist())
            })
    return pd.DataFrame(rows)
