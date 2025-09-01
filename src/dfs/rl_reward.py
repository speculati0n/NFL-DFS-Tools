from typing import Dict, Any
import math

def compute_reward(ln: Dict[str,Any], cfg: Dict[str,Any]) -> Dict[str,float]:
    r = {}

    metric = (cfg.get("base_metric") or "actual_points")
    if metric == "actual_points":
        base = float(ln.get("score") or 0.0)
    elif metric == "projected_points":
        base = float(ln.get("projections_proj") or 0.0)
    elif metric == "payout":
        base = float(ln.get("amount_won") or 0.0)
    elif metric == "finish_percentile":
        rk = float(ln.get("contest_rank") or ln.get("rank") or 0.0)
        fs = float(ln.get("field_size") or 1.0)
        base = 1.0 - (rk - 1.0) / max(fs, 1.0)
    else:
        base = float(ln.get("score") or 0.0)
    r["base"] = base

    floor = float(cfg.get("salary_floor", 49500))
    short = max(0.0, floor - float(ln.get("salary") or 0.0))
    r["salary_pen"] = (short/100.0) * float(cfg.get("salary_floor_penalty_per_100", -0.25))

    sb = 0.0
    for k, w in (cfg.get("stack_bonus") or {}).items():
        sb += float(w) * int(ln.get(f"stack_flags__{k}", 0))
    r["stack_bonus"] = sb

    feats = cfg.get("feature_penalties") or {}
    fp = 0.0
    if "any_vs_dst" in feats:
        fp += float(feats["any_vs_dst"]) * int(ln.get("feat_any_vs_dst", 0))
    if "double_te" in feats and int(ln.get("feat_double_te",0)) == 1:
        fp += float(feats["double_te"])
    r["feature_pen"] = fp

    flexb = 0.0
    fb = cfg.get("flex_bonus") or {}
    if int(ln.get("flex_is_wr",0)) == 1: flexb += float(fb.get("WR",0.0))
    if int(ln.get("flex_is_rb",0)) == 1: flexb += float(fb.get("RB",0.0))
    if int(ln.get("flex_is_te",0)) == 1: flexb += float(fb.get("TE",0.0))
    r["flex_bonus"] = flexb

    r["dist_pen"] = 0.0  # optional batch KL; apply in arena loop if target_distribution.enabled

    r["total"] = r["base"] + r["salary_pen"] + r["stack_bonus"] + r["feature_pen"] + r["flex_bonus"] + r["dist_pen"]
    return r
