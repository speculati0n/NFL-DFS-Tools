# rl arena that honors reward_weights, enforces <=10% off optimal proj (tunable),
# and can save accepted lineups for the simulator tagged as "agent".
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
import inspect
import numpy as np
import pandas as pd
import json
import os

# Optional exact ILP for "optimal projection" baseline; falls back to greedy if unavailable
try:
    import pulp as plp
    HAVE_PULP = True
except Exception:
    HAVE_PULP = False

from dfs_rl.envs.dk_nfl_env import DKNFLEnv, compute_reward
from dfs_rl.agents.random_agent import RandomAgent
from dfs_rl.agents.pg_agent import PGAgent
from dfs_rl.utils.lineups import lineup_key, SLOTS

# Prefer unified stack module if present; fallback to older stacks util
try:
    from stack_metrics import compute_features, compute_presence_and_counts, classify_bucket
except Exception:
    from src.dfs.stacks import compute_features, compute_presence_and_counts, classify_bucket

# ---------- config helpers ----------
def _get_config_path() -> str:
    try:
        from utils import get_config_path
        return get_config_path()
    except Exception:
        # try common paths
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, ".."))
        for p in [
            os.path.join(root, "config.json"),
            os.path.join(root, "uploads", "config.json"),
            os.path.join(os.getcwd(), "config.json"),
        ]:
            if os.path.exists(p):
                return p
        return ""

def _load_cfg_if_needed(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if cfg:
        return cfg
    path = _get_config_path()
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

# ---------- points / projections ----------
POINTS_COLS = [
    "projections_actpts",
    "score",
    "dk_points",
    "lineup_points",
    "ProjPoints",
    "projections_proj",
    "projection",
]

def _find_points_col(pool: pd.DataFrame) -> str:
    for c in POINTS_COLS:
        if c in pool.columns:
            return c
    # default expected column
    return "projections_proj"

def _points_sum_from_lineup(lineup: Dict[str, Any]) -> float:
    total = 0.0
    for slot in SLOTS:
        total += float(lineup.get(f"{slot}_proj", 0.0) or 0.0)
    return total

# ---------- lineup build / export ----------
def _build_lineup(pool: pd.DataFrame, idxs: List[int]) -> Dict[str, Any]:
    lineup: Dict[str, Any] = {}
    for slot, i in zip(SLOTS, idxs):
        r = pool.iloc[i]
        pid = r.get("Id") or r.get("id") or r.get("player_id") or r.get("playerid") or int(i)
        lineup[f"{slot}_id"] = pid
        lineup[f"{slot}_name"] = r.get("Name") or r.get("name")
        lineup[f"{slot}_team"] = r.get("team")
        lineup[f"{slot}_opp"]  = r.get("opp")
        lineup[f"{slot}_pos"]  = (r.get("pos") or "").upper()
        lineup[f"{slot}_salary"] = r.get("Salary") or r.get("salary")
        lineup[f"{slot}_proj"]   = (
            r.get("projections_proj")
            or r.get("ProjPoints")
            or r.get("projection")
            or 0.0
        )
    return lineup

def _lineup_wide_row(lineup: Dict[str, Any], agent: str, reward: float, tag_source: str) -> Dict[str, Any]:
    row = {
        "source": tag_source,
        "agent": agent,
        "reward": float(reward),
        "total_proj": float(_points_sum_from_lineup(lineup)),
    }
    for slot in SLOTS:
        row[slot] = lineup.get(f"{slot}_name")
        row[f"{slot}_id"] = lineup.get(f"{slot}_id")
    return row

def _export_lineups_for_simulator(rows: List[Dict[str, Any]], out_dir: Optional[str]) -> Optional[str]:
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    # canonical filename
    out_path = os.path.join(out_dir, "agent_lineups_for_sim.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    # also write a "latest" file some simulators look for
    latest_path = os.path.join(out_dir, "saved_lineups.csv")
    df.to_csv(latest_path, index=False, encoding="utf-8")
    return out_path

# ---------- reward / stacks ----------
def _stack_bonus_from_weights(lineup: Dict[str, Any], rw: Dict[str, float]) -> float:
    """
    Convert reward_weights into a scalar bonus/penalty for this lineup.
    Supports:
      - Stack count keys present in counts (e.g., "QB+WR", "RB vs OppWR")
      - Feature keys: "Double TE", "Any vs DST (per player)", "FLEX=WR/RB/TE"
    """
    flags, counts = compute_presence_and_counts(lineup)
    feats = compute_features(lineup)
    total = 0.0

    for key, w in (rw or {}).items():
        w = float(w)
        # stack count signals
        if key in counts:
            total += w * int(counts.get(key, 0))
            continue
        # feature signals
        if key == "Double TE":
            total += w * int(feats.get("feat_double_te", 0))
        elif key == "Any vs DST (per player)":
            total += w * int(feats.get("feat_any_vs_dst", 0))
        elif key == "FLEX=WR":
            total += w * int(feats.get("flex_is_wr", 0))
        elif key == "FLEX=RB":
            total += w * int(feats.get("flex_is_rb", 0))
        elif key == "FLEX=TE":
            total += w * int(feats.get("flex_is_te", 0))
        # ignore unknown keys silently
    return float(total)

# ---------- optimal projection baseline ----------
def _solve_optimal_projection(pool: pd.DataFrame, salary_cap: int = 50000) -> float:
    """
    Compute the maximum projected points lineup under DK Classic roster + salary constraints.
    Uses PuLP ILP if available; otherwise uses a greedy fallback.
    Assumes slots: QB(1), RB(2), WR(3), TE(1), FLEX(1 from RB/WR/TE), DST(1).
    """
    pts_col = _find_points_col(pool)

    if HAVE_PULP:
        n = len(pool)
        # binary x_{i,s} for each player i and slot s
        slots = ["QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST"]
        x = {(i, s): plp.LpVariable(f"x_{i}_{s}", lowBound=0, upBound=1, cat="Binary")
             for i in range(n) for s in slots}

        prob = plp.LpProblem("MaxProj", plp.LpMaximize)

        # Each slot filled exactly once
        for s in slots:
            prob += plp.lpSum(x[(i, s)] for i in range(n)) == 1, f"slot_{s}_filled"

        # Each player at most one slot
        for i in range(n):
            prob += plp.lpSum(x[(i, s)] for s in slots) <= 1, f"player_{i}_once"

        # Slot eligibility
        for i in range(n):
            pos = str(pool.iloc[i].get("pos") or "").upper()
            elig = {
                "QB": pos == "QB",
                "RB1": pos == "RB",
                "RB2": pos == "RB",
                "WR1": pos == "WR",
                "WR2": pos == "WR",
                "WR3": pos == "WR",
                "TE": pos == "TE",
                "FLEX": pos in ("RB","WR","TE"),
                "DST": pos == "DST",
            }
            for s in slots:
                if not elig[s]:
                    prob += x[(i, s)] == 0, f"elig_{i}_{s}"

        # Salary cap
        salaries = []
        for i in range(n):
            sal = pool.iloc[i].get("Salary") or pool.iloc[i].get("salary") or 0
            try:
                sal = int(sal)
            except Exception:
                sal = 0
            salaries.append(sal)
        prob += plp.lpSum(x[(i, s)] * salaries[i] for i in range(n) for s in slots) <= salary_cap, "salary_cap"

        # Objective: maximize projections
        projs = []
        for i in range(n):
            p = pool.iloc[i].get(pts_col) or 0.0
            try:
                p = float(p)
            except Exception:
                p = 0.0
            projs.append(p)
        prob.solve(plp.PULP_CBC_CMD(msg=False))
        # Sum of chosen projections
        best = 0.0
        for i in range(n):
            choose = any(plp.value(x[(i, s)]) > 0.5 for s in slots)
            if choose:
                best += projs[i]
        return float(best)

    # Greedy fallback: simple slot-by-slot pick, adjust for cap
    def _simple_best(pool: pd.DataFrame, pos: str, k: int) -> List[int]:
        sub = pool[pool["pos"].str.upper() == pos].copy()
        sub["p"] = pd.to_numeric(sub[_find_points_col(pool)], errors="coerce").fillna(0.0)
        sub.sort_values("p", ascending=False, inplace=True)
        return list(sub.index[:k])

    required = {
        "QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1
    }
    idxs = []
    for pos, k in required.items():
        idxs += _simple_best(pool, pos, k)

    # FLEX: best remaining from RB/WR/TE
    remaining = pool.drop(index=idxs)
    flex_cand = remaining[remaining["pos"].str.upper().isin(["RB","WR","TE"])].copy()
    flex_cand["p"] = pd.to_numeric(flex_cand[_find_points_col(flex_cand)], errors="coerce").fillna(0.0)
    if not flex_cand.empty:
        idxs.append(int(flex_cand.sort_values("p", ascending=False).index[0]))

    # If salary exceeds cap, try simple replacement loop
    def _salary_of(indices: List[int]) -> int:
        s = 0
        for i in indices:
            sal = pool.loc[i].get("Salary") or pool.loc[i].get("salary") or 0
            try:
                s += int(sal)
            except Exception:
                pass
        return int(s)

    def _proj_of(indices: List[int]) -> float:
        return float(pd.to_numeric(pool.loc[indices][_find_points_col(pool)], errors="coerce").fillna(0.0).sum())

    cap = 50000
    cur = list(idxs)
    if _salary_of(cur) <= cap:
        return _proj_of(cur)

    # Replace most expensive with cheaper next-best heuristic
    for _ in range(200):
        if _salary_of(cur) <= cap:
            break
        # find candidate to replace: highest salary among current that has cheaper eligible alternatives
        cur_sals = [(i, pool.loc[i].get("Salary") or pool.loc[i].get("salary") or 0) for i in cur]
        cur_sals = sorted(cur_sals, key=lambda t: int(t[1]), reverse=True)
        replaced = False
        for i, sal in cur_sals:
            pos = str(pool.loc[i, "pos"]).upper()
            # eligible pool: same slot type or for RB/WR/TE allow swap within same pos (FLEX handled already)
            cand = pool[(pool["pos"].str.upper() == pos) & (~(pool.index.isin(cur)))]
            if cand.empty:
                continue
            # pick next best cheaper
            cand = cand.copy()
            cand["p"] = pd.to_numeric(cand[_find_points_col(cand)], errors="coerce").fillna(0.0)
            cand["s"] = pd.to_numeric(cand["Salary"] if "Salary" in cand.columns else cand.get("salary", 0), errors="coerce").fillna(0)
            cheaper = cand[cand["s"] < int(sal)]
            if cheaper.empty:
                continue
            j = int(cheaper.sort_values(["s","p"], ascending=[True, False]).index[0])
            cur.remove(i)
            cur.append(j)
            replaced = True
            break
        if not replaced:
            break
    return _proj_of(cur)

# ---------- main RL tournament ----------
def run_tournament(
    pool: pd.DataFrame,
    n_lineups_per_agent: int = 150,
    train_pg: bool = True,
    seed: Optional[int] = None,
    cfg: Optional[Dict[str, Any]] = None,
    save_to_simulator: bool = False,
    sim_out_dir: Optional[str] = None,
    tag_source: str = "agent",
    **kwargs
) -> pd.DataFrame:
    """
    Generate lineups with RL agents and rank them by a reward that includes
    stack- and feature-based bonuses/penalties from cfg['reward_weights'].

    New:
      - Enforces projected points within (1 - max_pct_off_optimal) of the optimal projected lineup.
        cfg["rl"]["max_pct_off_optimal"]  (default 0.10)
      - save_to_simulator: writes agent-tagged CSV(s) into sim_out_dir.
    """
    if seed is not None:
        np.random.seed(int(seed))

    pool = pool.copy()
    if "pos" in pool.columns:
        pool["pos"] = pool["pos"].astype(str)
    if "salary" in pool.columns:
        try:
            pool["salary"] = pool["salary"].astype(int)
        except Exception:
            pass

    cfg = _load_cfg_if_needed(cfg)
    rl_cfg: Dict[str, Any] = cfg.get("rl", {})
    rw: Dict[str, float] = cfg.get("reward_weights", {}) or {}

    # Thresholding vs optimal projections
    max_pct_off_opt = float(rl_cfg.get("max_pct_off_optimal", 0.10))  # 10% default
    salary_cap = int(rl_cfg.get("salary_cap", 50000))
    try:
        opt_proj = _solve_optimal_projection(pool, salary_cap=salary_cap)
    except Exception:
        opt_proj = None
    min_proj_allowed = None
    if opt_proj and opt_proj > 0:
        min_proj_allowed = (1.0 - max_pct_off_opt) * opt_proj

    env = DKNFLEnv(pool)
    agents = {
        "random": RandomAgent(seed=1),
        "pg": PGAgent(n_players=len(pool), seed=2, cfg=rl_cfg),
    }

    # Dedupe + exposure controls across *all* agents
    seen_keys: List[Tuple[str, ...]] = []
    exposure_count: Counter[str] = Counter()
    max_exp = rl_cfg.get("max_player_exposure")  # e.g., 0.35
    max_resample = int(rl_cfg.get("max_resample_attempts", 50))

    def accept(lineup_dict: Dict[str, Any], key: Tuple[str, ...]) -> bool:
        # de-dup
        if key in seen_keys:
            return False
        # exposure
        if max_exp is not None:
            cap = max(1, int(float(max_exp) * max(n_lineups_per_agent, 1)))
            for pid in key:
                if exposure_count[pid] >= cap:
                    return False
        # projection threshold
        if min_proj_allowed is not None:
            base_points = _points_sum_from_lineup(lineup_dict)
            if base_points < min_proj_allowed:
                return False
        # accept
        seen_keys.append(key)
        for pid in key:
            exposure_count[pid] += 1
        return True

    rows = []
    export_rows = []  # wide format for simulator
    for name, agent in agents.items():
        collected = 0
        attempts = 0
        try:
            sig = inspect.signature(agent.act)
            act_uses_info = "info" in sig.parameters
        except (TypeError, ValueError):
            act_uses_info = False

        while collected < n_lineups_per_agent and attempts < n_lineups_per_agent * max_resample:
            attempts += 1
            obs, info = env.reset()
            mask = info.get("action_mask") if isinstance(info, dict) else None
            if mask is None:
                mask = obs
            done = False
            while not done:
                if act_uses_info:
                    action = agent.act(mask, info)
                else:
                    action = agent.act(mask)
                obs, reward, done, truncated, info = env.step(action)
                next_mask = info.get("action_mask") if isinstance(info, dict) else None
                if next_mask is not None:
                    mask = next_mask

            idxs = (
                info.get("idxs")
                or info.get("lineup_indices")
                or getattr(env, "state", {}).get("idxs")
                or getattr(env, "state", {}).get("lineup_indices")
            )
            if not idxs:
                continue
            lineup_dict = _build_lineup(pool, idxs)
            key = lineup_key(lineup_dict)

            # stack-aware scalar: includes FLEX=TE & Double TE penalties
            bonus = _stack_bonus_from_weights(lineup_dict, rw)
            base_points = _points_sum_from_lineup(lineup_dict)

            # compute_reward tolerances
            try:
                final_reward = compute_reward(lineup_dict, base_points, bonus, rl_cfg, seen_keys)
            except TypeError:
                try:
                    final_reward = compute_reward(lineup_dict, base_points, bonus, rl_cfg)
                except TypeError:
                    try:
                        final_reward = compute_reward(lineup_dict, base_points + bonus, rl_cfg)
                    except TypeError:
                        final_reward = base_points + bonus

            feats = compute_features(lineup_dict)
            flags, _ = compute_presence_and_counts(lineup_dict)
            bucket = classify_bucket(flags)

            if accept(lineup_dict, key):
                rows.append({
                    "agent": name,
                    "reward": float(final_reward),
                    "lineup_key": "|".join(key),
                    "stack_bucket": bucket,
                    "double_te": int(feats.get("feat_double_te", 0)),
                    "flex_pos": feats.get("flex_pos"),
                    "any_vs_dst_count": int(feats.get("feat_any_vs_dst", 0)),
                    "total_proj": float(base_points),
                })
                export_rows.append(_lineup_wide_row(lineup_dict, name, float(final_reward), tag_source))
                collected += 1

    df = (
        pd.DataFrame(rows)
        .sort_values("reward", ascending=False)
        .drop_duplicates("lineup_key", keep="first")
        .reset_index(drop=True)
    )

    # Optional save for simulator
    if save_to_simulator:
        out_dir = sim_out_dir or cfg.get("paths", {}).get("simulator_saved_lineups_dir") or os.path.join("uploads2", "saved_lineups")
        _export_lineups_for_simulator(export_rows, out_dir)

    return df

