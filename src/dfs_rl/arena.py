from typing import List, Tuple, Optional
from collections import Counter
import numpy as np
import pandas as pd

from dfs_rl.envs.dk_nfl_env import DKNFLEnv, compute_reward
from dfs_rl.agents.random_agent import RandomAgent
from dfs_rl.agents.pg_agent import PGAgent
from dfs_rl.utils.lineups import lineup_key, jaccard_similarity, SLOTS

# Use the same feature/count utilities as the optimizer/analysis
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

def _build_lineup(pool: pd.DataFrame, idxs: List[int]) -> dict:
    """Return a DK-classic lineup dict from row indices."""
    row = pool.iloc[idxs]
    lineup = {}
    for slot, i in zip(SLOTS, idxs):
        r = pool.iloc[i]
        pid = r.get("Id") or r.get("id") or r.get("player_id") or r.get("playerid") or i
        lineup[f"{slot}_id"] = pid
        lineup[f"{slot}_name"] = r.get("Name") or r.get("name")
        lineup[f"{slot}_team"] = r.get("team")
        lineup[f"{slot}_opp"] = r.get("opp")
        lineup[f"{slot}_pos"] = r.get("pos")
        lineup[f"{slot}_salary"] = r.get("Salary") or r.get("salary")
        lineup[f"{slot}_proj"] = r.get("projections_proj") or r.get("ProjPoints") or 0.0
    return lineup

def _stack_bonus_from_weights(lineup: dict, weights: dict) -> float:
    """
    Score a lineup using the same semantics as analysis/optimizer:
      - counts from compute_presence_and_counts()
      - features from compute_features()
      - apply reward_weights map
    """
    flags, counts = compute_presence_and_counts(lineup)
    feats = compute_features(lineup)
    total = 0.0
    for k, w in (weights or {}).items():
        if k in counts:
            total += float(w) * counts.get(k, 0)
        elif k == "Double TE":
            total += float(w) * int(feats.get("feat_double_te", 0))
        elif k == "Any vs DST (per player)":
            total += float(w) * int(feats.get("feat_any_vs_dst", 0))
        elif k == "FLEX=WR":
            total += float(w) * int(feats.get("flex_is_wr", 0))
        elif k == "FLEX=RB":
            total += float(w) * int(feats.get("flex_is_rb", 0))
        elif k == "FLEX=TE":
            total += float(w) * int(feats.get("flex_is_te", 0))
    return float(total)

def _run_agent(env: DKNFLEnv, agent, train: bool) -> Tuple[List[int], int, float]:
    """Rollout one lineup and optionally train the agent."""
    obs, info = env.reset()
    done = False
    steps = 0
    while not done:
        action = agent.act(obs, info)
        obs, reward, done, truncated, info = env.step(action)
        if train and hasattr(agent, "train_step"):
            agent.train_step(obs, reward, done, info)
        steps += 1
    return env.state["idxs"], steps, float(info.get("sum_proj", 0.0))

def run_tournament(pool: pd.DataFrame, n_lineups_per_agent: int = 150,
                   train_pg: bool = True, cfg: Optional[dict] = None) -> pd.DataFrame:
    cfg = cfg or {}
    env = DKNFLEnv(pool)
    agents = {
        "random": RandomAgent(seed=1),
        "pg": PGAgent(n_players=len(pool), seed=2, cfg=cfg),
    }

    rl_cfg = cfg.get("rl", {})
    rw = cfg.get("reward_weights", {}) or {}

    pts_col = _find_points_col(pool) or "projections_proj"
    seen_keys_global = set()
    exposure_count: Counter[str] = Counter()

    def accept_lineup_if_unique(lineup: dict) -> Tuple[bool, tuple]:
        key = lineup_key(lineup)
        if key in seen_keys_global:
            return False, key
        max_exp = rl_cfg.get("max_player_exposure")
        if max_exp is not None:
            pool_size = cfg.get("arena_pool_size") or 1
            cap = int(max_exp * pool_size)
            for pid in key:
                if exposure_count[pid] >= cap:
                    return False, key
        seen_keys_global.add(key)
        for pid in key:
            exposure_count[pid] += 1
        return True, key

    rows = []
    for name, agent in agents.items():
        for _ in range(n_lineups_per_agent):
            attempts, accepted, key = 0, False, tuple()
            lineup_dict = {}
            while attempts < rl_cfg.get("max_resample_attempts", 25) and not accepted:
                idxs, steps, base_points = _run_agent(env, agent, train=(train_pg and name == "pg"))
                lineup_dict = _build_lineup(pool, idxs)
                accepted, key = accept_lineup_if_unique(lineup_dict)
                attempts += 1
            # stack-aware reward: add weighted bonus/penalties
            stack_bonus = _stack_bonus_from_weights(lineup_dict, rw)
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

