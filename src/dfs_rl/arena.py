from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
import numpy as np
import pandas as pd

from dfs_rl.envs.dk_nfl_env import DKNFLEnv, compute_reward
from dfs_rl.agents.random_agent import RandomAgent
from dfs_rl.agents.pg_agent import PGAgent
from dfs_rl.utils.lineups import lineup_key, SLOTS
from src.dfs.stacks import compute_features, compute_presence_and_counts, classify_bucket

POINTS_COLS = [
    "projections_actpts",
    "score",
    "dk_points",
    "lineup_points",
    "ProjPoints",
    "projections_proj",
]

def _find_points_col(pool: pd.DataFrame) -> Optional[str]:
    for c in POINTS_COLS:
        if c in pool.columns:
            return c
    return None

def _build_lineup(pool: pd.DataFrame, idxs: List[int]) -> Dict[str, Any]:
    row = pool.iloc[idxs]
    lineup: Dict[str, Any] = {}
    for slot, i in zip(SLOTS, idxs):
        r = pool.iloc[i]
        pid = r.get("Id") or r.get("id") or r.get("player_id") or r.get("playerid") or i
        lineup[f"{slot}_id"] = pid
        lineup[f"{slot}_name"] = r.get("Name") or r.get("name")
        lineup[f"{slot}_team"] = r.get("team")
        lineup[f"{slot}_opp"] = r.get("opp")
        lineup[f"{slot}_pos"] = (r.get("pos") or "").upper()
        lineup[f"{slot}_salary"] = r.get("Salary") or r.get("salary")
        lineup[f"{slot}_proj"] = r.get("projections_proj") or r.get("ProjPoints") or 0.0
    return lineup

def _stack_bonus_from_weights(lineup: Dict[str, Any], rw: Dict[str, float]) -> float:
    """Turn reward_weights into a single scalar bonus (or penalty)."""
    flags, counts = compute_presence_and_counts(lineup)
    feats = compute_features(lineup)
    total = 0.0
    for key, w in (rw or {}).items():
        # stack counts
        if key in counts:
            total += float(w) * counts.get(key, 0)
            continue
        # feature keys
        if key == "Double TE":
            total += float(w) * int(feats.get("feat_double_te", 0))
        elif key == "Any vs DST (per player)":
            total += float(w) * int(feats.get("feat_any_vs_dst", 0))
        elif key == "FLEX=WR":
            total += float(w) * int(feats.get("flex_is_wr", 0))
        elif key == "FLEX=RB":
            total += float(w) * int(feats.get("flex_is_rb", 0))
        elif key == "FLEX=TE":
            total += float(w) * int(feats.get("flex_is_te", 0))
    return float(total)

def run_tournament(pool: pd.DataFrame,
                   n_lineups_per_agent: int = 150,
                   train_pg: bool = True,
                   seed: Optional[int] = None,
                   cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(int(seed))

    pool = pool.copy()
    # sanitize salary if needed
    if "salary" in pool.columns:
        pool["salary"] = pool["salary"].astype(int)

    cfg = cfg or {}
    rl_cfg: Dict[str, Any] = cfg.get("rl", {})
    rw: Dict[str, float] = cfg.get("reward_weights", {}) or {}

    env = DKNFLEnv(pool)
    agents = {
        "random": RandomAgent(seed=1),
        "pg": PGAgent(n_players=len(pool), seed=2, cfg=rl_cfg),
    }

    pts_col = _find_points_col(pool) or "projections_proj"
    seen_keys = set()
    exposure_count: Counter[str] = Counter()
    max_exp = rl_cfg.get("max_player_exposure")

    def accept(lineup_dict: Dict[str, Any], key: Tuple[str, ...]) -> bool:
        if key in seen_keys:
            return False
        if max_exp is not None:
            cap = int(max_exp * max(n_lineups_per_agent, 1))
            for pid in key:
                if exposure_count[pid] >= cap:
                    return False
        seen_keys.add(key)
        for pid in key:
            exposure_count[pid] += 1
        return True

    rows = []
    for name, agent in agents.items():
        for _ in range(n_lineups_per_agent):
            obs, info = env.reset()
            done = False
            while not done:
                action = agent.act(obs, info)
                obs, reward, done, truncated, info = env.step(action)

            idxs = info.get("idxs") or env.state.get("idxs")
            lineup_dict = _build_lineup(pool, idxs)

            key = lineup_key(lineup_dict)
            bonus = _stack_bonus_from_weights(lineup_dict, rw)
            base_points = float(sum(lineup_dict.get(f"{s}_proj", 0.0) for s in SLOTS))
            final_reward = compute_reward(lineup_dict, base_points, bonus, rl_cfg)

            feats = compute_features(lineup_dict)
            if accept(lineup_dict, key):
                rows.append({
                    "agent": name,
                    "reward": final_reward,
                    "lineup_key": "|".join(key),
                    "bucket": classify_bucket(compute_presence_and_counts(lineup_dict)[0]),
                    "double_te": feats.get("feat_double_te"),
                    "flex_pos": feats.get("flex_pos"),
                    "dst_conflicts": feats.get("feat_any_vs_dst"),
                })

    df = pd.DataFrame(rows).sort_values("reward", ascending=False).drop_duplicates("lineup_key").reset_index(drop=True)
    return df
