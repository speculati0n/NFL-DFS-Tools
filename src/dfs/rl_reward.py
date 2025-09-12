from typing import Dict, Any

from dfs.constraints import DEFAULT_SALARY_CAP
from stack_metrics import analyze_lineup, compute_presence_and_counts, compute_features, classify_bucket

STACK_KEYS = ["stack_bucket", "double_te", "flex_pos", "dst_conflicts"]

def compute_reward_components(row: Dict[str, Any], w: Dict[str, float]) -> Dict[str, float]:
    """Return weighted components of the reward for debugging."""
    comps = {
        "proj": w.get("proj", 1.0) * float(row.get("projections_proj", 0.0)),
        "salary_util": 0.0,
        "stack": 0.0,
        "double_te": 0.0,
        "dst_conflicts": 0.0,
        "flex_pref": 0.0,
        "validity": 0.0,
    }

    salary = float(row.get("salary", 0.0)) / float(DEFAULT_SALARY_CAP)
    comps["salary_util"] = w.get("salary_util", 0.0) * salary

    flags, _ = compute_presence_and_counts(row)
    if flags.get("QB+WR", 0):
        comps["stack"] += w.get("qb_wr_bonus", 0.0)
    if flags.get("QB+TE", 0):
        comps["stack"] += w.get("qb_te_bonus", 0.0)
    bringback_flags = [
        "QB+WR+OppWR",
        "QB+WR+OppTE",
        "QB+TE+OppWR",
        "QB+RB+OppWR",
        "QB+RB+OppTE",
        "QB+WR+WR+OppWR",
        "QB+WR+WR+OppTE",
        "QB+WR+WR+OppWR+OppWR",
    ]
    if any(flags.get(f, 0) for f in bringback_flags):
        comps["stack"] += w.get("bringback_bonus", 0.0)

    feats = compute_features(row)
    comps["double_te"] = w.get("double_te_penalty", 0.0) * int(
        feats.get("feat_double_te", 0)
    )
    comps["dst_conflicts"] = w.get("dst_conflict_penalty", 0.0) * int(
        feats.get("feat_any_vs_dst", 0)
    )
    fpos = feats.get("flex_pos", "")
    if fpos == "WR":
        comps["flex_pref"] = w.get("flex_wr_bonus", 0.0)
    elif fpos == "TE":
        comps["flex_pref"] = w.get("flex_te_penalty", 0.0)
    else:
        comps["flex_pref"] = 0.0

    return comps


def compute_reward(row: Dict[str, Any], w: Dict[str, float]) -> float:
    comps = compute_reward_components(row, w)
    return float(sum(comps.values()))


def compute_partial_reward(lineup_row_like: Dict[str, Any], w: Dict[str, float]) -> float:
    """Safe to call with incomplete lineups; missing slots contribute 0."""
    try:
        return compute_reward(lineup_row_like, w)
    except Exception:
        return 0.0


def compute_reward_from_weights(row: Dict[str, Any], rw: Dict[str, float]) -> float:
    """Compute reward using explicit stack weights from config.

    // 2023+2025 stack tuning
    """
    base = float(row.get("projections_proj", 0.0))
    flags, counts = compute_presence_and_counts(row)
    feats = compute_features(row)
    total = base
    for key, w in rw.items():
        if key in counts:
            total += w * counts.get(key, 0)
        elif key == "Double TE":
            total += w * int(feats.get("feat_double_te", 0))
        elif key == "Any vs DST (per player)":
            total += w * int(feats.get("feat_any_vs_dst", 0))
        elif key == "FLEX=WR":
            total += w * int(feats.get("flex_is_wr", 0))
        elif key == "FLEX=RB":
            total += w * int(feats.get("flex_is_rb", 0))
        elif key == "FLEX=TE":
            total += w * int(feats.get("flex_is_te", 0))
    return total
