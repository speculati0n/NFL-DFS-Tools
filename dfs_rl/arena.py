from typing import List, Tuple, Optional
from collections import Counter
import numpy as np
import pandas as pd

from dfs_rl.envs.dk_nfl_env import DKNFLEnv, compute_reward
from dfs_rl.agents.random_agent import RandomAgent
from dfs_rl.agents.pg_agent import PGAgent
from dfs_rl.utils.lineups import lineup_key, jaccard_similarity, SLOTS
from src.dfs.stacks import compute_features, compute_presence_and_counts, classify_bucket

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
def _build_lineup(pool: pd.DataFrame, idxs: List[int]) -> dict:
    lineup: dict = {}
    for slot, idx in zip(SLOTS, idxs):
        row = pool.iloc[idx]
        lineup[f"{slot}"] = row.get("name")
        pid = row.get("player_id") or row.get("id")
        if pid is not None:
            lineup[f"{slot}_id"] = pid
        lineup[f"{slot}_team"] = row.get("team")
        lineup[f"{slot}_opp"] = row.get("opp")
        lineup[f"{slot}_pos"] = row.get("pos")
    return lineup


def run_tournament(pool: pd.DataFrame, n_lineups_per_agent: int = 150,
                   train_pg: bool = True, cfg: Optional[dict] = None) -> pd.DataFrame:
    cfg = cfg or {}
    env = DKNFLEnv(pool)
    n = len(pool)
    agents = {
        "random": RandomAgent(seed=1),
        "pg": PGAgent(n_players=n, seed=2, cfg=cfg),
    }

    rl_cfg = cfg.get("rl", {})
    pts_col = _find_points_col(pool) or "projections_proj"
    seen_keys_global = set()
    exposure_count: Counter[str] = Counter()

    def accept_lineup_if_unique(lu: dict) -> Tuple[bool, tuple]:
        key = lineup_key(lu)
        if key in seen_keys_global:
            return False, key
        max_exp = rl_cfg.get("max_player_exposure")
        if max_exp is not None:
            pool_size = cfg.get("arena_pool_size") or 1
            cap = int(max_exp * pool_size)
            for pid in key:
                if exposure_count[pid] >= cap:
                    return False, key
        min_div = rl_cfg.get("min_jaccard_diversity")
        if min_div is not None and seen_keys_global:
            ids = list(key)
            comp = list(seen_keys_global)[-200:]
            if comp:
                sim = max(jaccard_similarity(ids, list(k)) for k in comp)
                if sim >= min_div:
                    return False, key
        seen_keys_global.add(key)
        for pid in key:
            exposure_count[pid] += 1
        return True, key

    rows = []
    for name, agent in agents.items():
        for i in range(n_lineups_per_agent):
            attempts = 0
            accepted = False
            key = tuple()
            lineup_dict = {}
            while attempts < rl_cfg.get("max_resample_attempts", 25) and not accepted:
                idxs, steps, base_reward = _run_agent(env, agent, train=(train_pg and name == "pg"))
                lineup_dict = _build_lineup(pool, idxs)
                accepted, key = accept_lineup_if_unique(lineup_dict)
                attempts += 1
            base_points = float(pool.loc[idxs, pts_col].sum())
            stack_bonus = 0.0
            reward = compute_reward(lineup_dict, base_points, stack_bonus, rl_cfg, seen_keys_global)
            feats = compute_features(lineup_dict)
            flags, _ = compute_presence_and_counts(lineup_dict)
            bucket = classify_bucket(flags)
            rows.append({
                "agent": name,
                "reward": reward,
                "lineup_key": "|".join(key),
                "stack_bucket": bucket,
                "double_te": feats.get("feat_double_te"),
                "flex_pos": feats.get("flex_pos"),
                "dst_conflicts": feats.get("feat_any_vs_dst"),
                "is_duplicate": 0 if accepted else 1
            })

    df = pd.DataFrame(rows)
    dupes = int(df.duplicated("lineup_key", keep=False).sum())
    if rl_cfg.get("dedupe_on_collect", True):
        df = (df.sort_values(["reward"], ascending=False)
                .drop_duplicates("lineup_key", keep="first")
                .reset_index(drop=True))
    df.attrs["duplicates"] = dupes
    return df
