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
from dfs.rl_reward import compute_reward_components
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
    rl_reward_cfg = cfg.get("rl_reward", {})

    if rl_reward_cfg.get("force_active"):
        w_flip = {k: -v for k, v in (rl_reward_cfg.get("weights", {})).items()}
        cfg_flip = {**rl_reward_cfg, "weights": w_flip}
        env_a = DKNFLEnv(pool, min_salary_pct=min_salary_pct, rl_reward_cfg=rl_reward_cfg)
        env_b = DKNFLEnv(pool, min_salary_pct=min_salary_pct, rl_reward_cfg=cfg_flip)
        ga = GreedyAgent(env_a, epsilon=0.0)
        gb = GreedyAgent(env_b, epsilon=0.0)
        idx_a, _, _, _ = _run_agent(env_a, ga, train=False)
        idx_b, _, _, _ = _run_agent(env_b, gb, train=False)
        if idx_a == idx_b:
            raise RuntimeError("RL reward appears inactive; flipping weights produced same lineup")

    env = DKNFLEnv(pool, min_salary_pct=min_salary_pct, rl_reward_cfg=rl_reward_cfg)
    n = len(pool)
    base_seed = int(seed) if seed is not None else 0
    agents = {
        "random": RandomAgent(pool["salary"].to_numpy(), seed=base_seed + 1),
        "pg": PGAgent(n_players=n, seed=base_seed + 2),
        "greedy": GreedyAgent(env, epsilon=rl_reward_cfg.get("epsilon_greedy", 0.05)),
    }

    rows = []
    slot_cols = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]

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
            row.update({"agent": name, "iteration": i, "reward": reward})
            comps = compute_reward_components(row, env.rl_reward_cfg)
            row.update({f"rw_{k}": v for k, v in comps.items()})
            rows.append(row)

    return pd.DataFrame(rows)
