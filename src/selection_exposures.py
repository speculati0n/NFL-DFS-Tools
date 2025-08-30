from typing import List, Dict
from collections import defaultdict, Counter



from stack_metrics import analyze_lineup


def select_lineups(candidates: List[List[str]], player_dict: Dict, targets: Dict, num_final: int) -> List[List[str]]:
    """
    Greedy selection of lineups to match exposure targets.
    """
    presence_tgt = targets.get("presence_targets_pct", {})
    mult_tgt = targets.get("multiplicity_targets_mean", {})
    bucket_tgt = targets.get("bucket_mix_pct", {})

    metrics = [analyze_lineup(l, player_dict) for l in candidates]

    # Warn if targets impossible
    for key, val in presence_tgt.items():
        if val > 0 and not any(m["presence"].get(key, 0) for m in metrics):
            print(f"Warning: presence target {key} cannot be met; not in pool")
    for key, val in bucket_tgt.items():
        if val > 0 and not any(m["bucket"] == key for m in metrics):
            print(f"Warning: bucket target {key} cannot be met; not in pool")
    for key, val in mult_tgt.items():
        if val > 0 and not any(m["counts"].get(key, 0) for m in metrics):
            print(f"Warning: multiplicity target {key} cannot be met; not in pool")

    remaining = list(range(len(candidates)))
    selected = []
    presence_sum = defaultdict(int)
    mult_sum = defaultdict(int)
    bucket_sum = defaultdict(int)
    total = 0

    def error(p_sum, m_sum, b_sum, t):
        e = 0.0
        for k, target in presence_tgt.items():
            cur = p_sum.get(k, 0) / t if t else 0
            e += (cur - target) ** 2
        for k, target in mult_tgt.items():
            cur = m_sum.get(k, 0) / t if t else 0
            e += 0.7 * (cur - target) ** 2
        for k, target in bucket_tgt.items():
            cur = b_sum.get(k, 0) / t if t else 0
            e += (cur - target) ** 2
        return e

    while len(selected) < num_final and remaining:
        best_idx = None
        best_err = float("inf")
        for idx in remaining:
            m = metrics[idx]
            p_new = presence_sum.copy()
            for k, v in m["presence"].items():
                if k in presence_tgt:
                    p_new[k] += v
            m_new = mult_sum.copy()
            for k, v in m["counts"].items():
                if k in mult_tgt:
                    m_new[k] += v
            b_new = bucket_sum.copy()
            b = m["bucket"]
            if b in bucket_tgt:
                b_new[b] += 1
            err = error(p_new, m_new, b_new, total + 1)
            if err < best_err:
                best_err = err
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        m = metrics[best_idx]
        for k, v in m["presence"].items():
            if k in presence_tgt:
                presence_sum[k] += v
        for k, v in m["counts"].items():
            if k in mult_tgt:
                mult_sum[k] += v
        b = m["bucket"]
        if b in bucket_tgt:
            bucket_sum[b] += 1
        total += 1
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]




    Parameters
    ----------
    lineups : list of lineups
        Each lineup is a list of player keys.
    player_dict : dict
        Mapping of player keys to player info.
    targets : dict
        Exposure targets from config (presence, multiplicity and bucket).


    presence_tot = Counter()
    mult_tot = Counter()
    bucket_tot = Counter()
    for lu in lineups:
        metrics = analyze_lineup(lu, player_dict)
        presence_tot.update(metrics["presence"])
        mult_tot.update(metrics["counts"])
        bucket_tot[metrics["bucket"]] += 1
    n = len(lineups)


