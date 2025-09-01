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
from dfs.constraints import (
    Player,
    Lineup,
    validate_lineup,
    repair_to_min_salary,
    sanitize_salary,
    DEFAULT_SALARY_CAP,
    DEFAULT_MIN_SPEND_PCT,
)
from dfs.stacks import (
    compute_presence_and_counts,
    classify_bucket,
    compute_features,
)
from dfs.rl_reward import compute_reward
from utils import get_config_path

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

def _run_agent(env: DKNFLEnv, agent, train: bool) -> Tuple[list, int, float]:
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


def run_tournament(
    pool: pd.DataFrame,
    n_lineups_per_agent: int = 150,
    train_pg: bool = True,
    min_salary_pct: float | None = None,
) -> pd.DataFrame:
    if min_salary_pct is None:
        min_salary_pct = float(os.getenv("MIN_SALARY_PCT", DEFAULT_MIN_SPEND_PCT))

    pool = pool.copy()
    pool["salary"] = pool["salary"].apply(sanitize_salary)

    players: List[Player] = []
    pool_by_pos = {"QB": [], "RB": [], "WR": [], "TE": [], "DST": []}
    for idx, row in pool.iterrows():
        p = Player(
            id=str(idx),
            name=row["name"],
            pos=row["pos"],
            team=row.get("team"),
            opp=row.get("opp"),
            salary=int(row["salary"]),
            proj=float(row["projections_proj"]),
        )
        players.append(p)
        pool_by_pos[p.pos].append(p)

    cfg: Dict[str,Any] = {}
    try:
        with open(get_config_path()) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    rl_reward_cfg = (cfg.get("rl") or {}).get("reward", {})

    env = DKNFLEnv(pool, min_salary_pct=min_salary_pct, rl_reward_cfg=rl_reward_cfg)
    n = len(pool)
    agents = {
        "random": RandomAgent(pool["salary"].to_numpy(), seed=1),
        "pg": PGAgent(n_players=n, seed=2),
        "greedy": GreedyAgent(pool["projections_proj"].to_numpy()),
    }

    pts_col = _find_points_col(pool)

    rows = []
    slot_cols = ["QB", "RB1", "RB2", "WR1", "WR2", "WR3", "TE", "FLEX", "DST"]

    for name, agent in agents.items():
        for i in range(n_lineups_per_agent):
            idxs, steps, reward = _run_agent(
                env, agent, train=(train_pg and name == "pg")
            )
            if len(idxs) != len(slot_cols):
                continue

            lineup = Lineup()
            for slot, idx in zip(slot_cols, idxs):
                setattr(lineup, slot, players[idx])

            if not validate_lineup(
                lineup, cap=DEFAULT_SALARY_CAP, min_pct=min_salary_pct
            ):
                before = lineup.salary()
                lineup = repair_to_min_salary(
                    lineup, pool_by_pos, cap=DEFAULT_SALARY_CAP, min_pct=min_salary_pct
                )
                if lineup.salary() != before:
                    print(f"REPAIRED from ${before} to ${lineup.salary()}")

            if not validate_lineup(
                lineup, cap=DEFAULT_SALARY_CAP, min_pct=min_salary_pct
            ):
                print(
                    f"Discarding invalid lineup from {name} with salary {lineup.salary()}"
                )
                continue

            row: Dict[str,Any] = {"agent": name, "iteration": i, "salary": lineup.salary()}
            for slot in slot_cols:
                p = getattr(lineup, slot)
                row[slot] = p.name
                row[f"{slot}_team"] = p.team
                row[f"{slot}_opp"] = p.opp
                row[f"{slot}_pos"] = p.pos

            row["projections_proj"] = lineup.projection()
            if pts_col:
                Ldf = pool.iloc[idxs]
                if pts_col in Ldf.columns:
                    total_pts = float(Ldf[pts_col].sum())
                    row[pts_col] = total_pts
                    if pts_col.lower() != "score":
                        row["score"] = total_pts
                else:
                    row["score"] = row["projections_proj"]
            else:
                row["score"] = row["projections_proj"]

            flags, counts = compute_presence_and_counts(row)
            feats = compute_features(row)
            bucket = classify_bucket(flags)
            row["stack_bucket"] = bucket
            for k,v in flags.items():
                row[f"stack_flags__{k}"] = v
            for k,v in counts.items():
                row[f"stack_count__{k}"] = v
            for k,v in feats.items():
                row[k] = v

            r = compute_reward(row, rl_reward_cfg)
            row.update({
                "reward_total": r["total"],
                "r_base": r["base"],
                "r_salary_pen": r["salary_pen"],
                "r_stack_bonus": r["stack_bonus"],
                "r_feature_pen": r["feature_pen"],
                "r_flex_bonus": r["flex_bonus"],
                "r_dist_pen": r["dist_pen"],
            })

            rows.append(row)

    return pd.DataFrame(rows)
