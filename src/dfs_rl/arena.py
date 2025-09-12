from typing import List, Tuple, Optional
import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any

from dfs_rl.envs.dk_nfl_env import DKNFLEnv
from dfs_rl.agents.random_agent import RandomAgent
from dfs_rl.agents.pg_agent import PGAgent
from dfs_rl.agents.greedy_agent import GreedyAgent
from dfs.constraints import sanitize_salary, DEFAULT_MIN_SPEND_PCT, DEFAULT_SALARY_CAP
from dfs.rl_reward import compute_reward_from_weights
from dfs.stacks import compute_presence_and_counts, compute_features, classify_bucket
from dfs_rl.utils.lineup import lineup_key
from utils import get_config_path

def _run_agent(env: DKNFLEnv, agent, train: bool) -> Tuple[list, int, float, Dict[str, Any]]:
    states, actions, rewards = [], [], []
    obs, info = env.reset()
    done = False
    while not done:
        mask = info["action_mask"]
        if hasattr(agent, "sample"):
            a, logp = agent.sample(mask)
            actions.append((a, logp))
        else:
            a = agent.act(mask)
        obs, r, done, _, info = env.step(a)
        rewards.append(r)
        states.append(obs)
    if train and hasattr(agent, "update"):
        agent.update(states, actions, rewards)
    return info.get("lineup_indices", []), len(rewards), float(sum(rewards)), info


def run_tournament(
    pool: pd.DataFrame,
    n_lineups_per_agent: int = 150,
    train_pg: bool = True,
    min_salary_pct: float | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    if min_salary_pct is None:
        min_salary_pct = float(os.getenv("MIN_SALARY_PCT", DEFAULT_MIN_SPEND_PCT))
    if seed is not None:
        np.random.seed(int(seed))

    pool = pool.copy()
    pool["salary"] = pool["salary"].apply(sanitize_salary)

    cfg: Dict[str,Any] = {}
    try:
        with open(get_config_path()) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    rw = cfg.get("reward_weights", {})
    env = DKNFLEnv(pool, min_salary_pct=min_salary_pct, rl_reward_cfg={})
    n = len(pool)
    base_seed = int(seed) if seed is not None else 0
    agents = {
        "random": RandomAgent(pool["salary"].to_numpy(), seed=base_seed + 1),
        "pg": PGAgent(n_players=n, seed=base_seed + 2),
        "greedy": GreedyAgent(env, epsilon=0.05),
    }

    rows = []
    slot_cols = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]

    seen = []
    for name, agent in agents.items():
        for i in range(n_lineups_per_agent):
            idxs, steps, reward, info = _run_agent(
                env, agent, train=(train_pg and name == "pg")
            )
            if len(idxs) != len(slot_cols):
                continue

            row = info.copy()
            if float(row.get("salary", 0.0)) < DEFAULT_SALARY_CAP * min_salary_pct:
                continue

            flags, counts = compute_presence_and_counts(row)
            feats = compute_features(row)
            bucket = classify_bucket(flags)
            bringback_type = "None"
            if counts.get("QB+OppRB", 0) or counts.get("QB+WR+OppRB", 0):
                bringback_type = "Opp RB"
            elif any(counts.get(k, 0) for k in ["QB+OppWR", "QB+WR+OppWR", "QB+WR+WR+OppWR"]):
                bringback_type = "Opp WR"
            row.update(
                {
                    "agent": name,
                    "iteration": i,
                    "stack_bucket": bucket,
                    "flex_pos": feats.get("flex_pos", ""),
                    "bringback_type": bringback_type,
                    "any_vs_dst_count": int(feats.get("feat_any_vs_dst", 0)),
                }
            )
            # 2023+2025 stack tuning
            key = lineup_key(row)
            rwd = compute_reward_from_weights(row, rw)
            if key in seen:
                rwd -= 0.1
            else:
                seen.append(key)
            row["reward"] = rwd
            rows.append({**row, "_key": key})

    df = pd.DataFrame(rows)
    if "_key" in df.columns:
        df = df.drop_duplicates("_key").drop(columns=["_key"])
    return df
