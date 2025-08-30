import os
import sys
import re
from collections import Counter

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from selection_exposures import select_lineups, report_lineup_exposures
from stack_metrics import analyze_lineup


def build_player(pid, pos, team, opp):
    return {
        "ID": pid,
        "Position": pos,
        "Team": team,
        "Opponent": opp,
        "Name": f"{pos}_{pid}",
    }


player_dict = {
    ("qb_a", "QB", "A"): build_player(1, "QB", "A", "B"),
    ("wr_a1", "WR", "A"): build_player(2, "WR", "A", "B"),
    ("te_a1", "TE", "A"): build_player(3, "TE", "A", "B"),
    ("qb_b", "QB", "B"): build_player(4, "QB", "B", "A"),
    ("wr_b1", "WR", "B"): build_player(5, "WR", "B", "A"),
    ("te_b1", "TE", "B"): build_player(6, "TE", "B", "A"),
}


# Build pool: 30 QB+WR lineups, 30 QB+TE lineups
lineups = []
for _ in range(30):
    lineups.append([("qb_a", "QB", "A"), ("wr_a1", "WR", "A"), ("wr_b1", "WR", "B")])
for _ in range(30):
    lineups.append([("qb_a", "QB", "A"), ("te_a1", "TE", "A"), ("wr_b1", "WR", "B")])

targets = {
    "presence_targets_pct": {"QB+WR": 0.5, "QB+TE": 0.5},
    "multiplicity_targets_mean": {"QB+WR": 0.5, "QB+TE": 0.5},
    "bucket_mix_pct": {"QB+WR+OppWR": 0.5, "QB+TE+OppWR": 0.5},
}


def test_selector_hits_targets():
    selected = select_lineups(lineups, player_dict, targets, 20)
    presence_total = Counter()
    mult_total = Counter()
    bucket_total = Counter()
    for l in selected:
        metrics = analyze_lineup(l, player_dict)
        presence_total.update(metrics["presence"])
        mult_total.update(metrics["counts"])
        bucket_total[metrics["bucket"]] += 1
    n = len(selected)
    for k, tgt in targets["presence_targets_pct"].items():
        assert abs(presence_total[k] / n - tgt) <= 0.02
    for k, tgt in targets["multiplicity_targets_mean"].items():
        assert abs(mult_total[k] / n - tgt) <= 0.05
    for k, tgt in targets["bucket_mix_pct"].items():
        assert abs(bucket_total[k] / n - tgt) <= 0.02


def test_report_lineup_exposures(capsys):
    selected = select_lineups(lineups, player_dict, targets, 20)

