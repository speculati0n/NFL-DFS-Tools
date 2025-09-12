"""Generate lineup exposure summary.

// 2023+2025 stack tuning
"""
import json
from collections import Counter

from dfs_rl.utils.data import load_week_folder
from dfs_rl.arena import run_tournament
from stack_metrics import analyze_lineup, compute_presence_and_counts, compute_features, classify_bucket


def main() -> None:
    week = "2019-09-22"
    bundle = load_week_folder(week)
    pool = bundle["projections"]
    df = run_tournament(pool, n_lineups_per_agent=5000 // 3 + 1, train_pg=False)
    df = df.head(5000)

    counts = Counter()
    for _, row in df.iterrows():
        flags, _ = compute_presence_and_counts(row.to_dict())
        feats = compute_features(row.to_dict())
        counts["QB+WR"] += int(flags.get("QB+WR", 0) > 0)
        counts["QB+WR+TE"] += int(flags.get("QB+WR+TE", 0) > 0)
        counts["double_stack"] += int(flags.get("QB+WR+WR", 0) > 0)
        counts["bring_rb"] += int(row.get("bringback_type") == "Opp RB")
        counts["bring_wr"] += int(row.get("bringback_type") == "Opp WR")
        counts["flex_WR"] += int(row.get("flex_pos") == "WR")
        counts["flex_RB"] += int(row.get("flex_pos") == "RB")
        counts["flex_TE"] += int(row.get("flex_pos") == "TE")
        counts["any_vs_dst"] += int(row.get("any_vs_dst_count", 0) > 0)

    n = len(df)
    print(f"QB+WR: {counts['QB+WR']/n:.1%}")
    print(f"QB+WR+TE: {counts['QB+WR+TE']/n:.1%}")
    print(f"Double stacks: {counts['double_stack']/n:.1%}")
    print(
        f"Bring-backs RB/WR/None: {counts['bring_rb']/n:.1%} / {counts['bring_wr']/n:.1%} / {(n-counts['bring_rb']-counts['bring_wr'])/n:.1%}"
    )
    print(
        f"FLEX mix WR/RB/TE: {counts['flex_WR']/n:.1%} / {counts['flex_RB']/n:.1%} / {counts['flex_TE']/n:.1%}"
    )
    print(f"Any vs DST rate: {counts['any_vs_dst']/n:.1%}")


if __name__ == "__main__":
    main()

