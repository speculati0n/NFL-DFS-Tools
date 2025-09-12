import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from stack_metrics import analyze_lineup


def test_analyze_lineup_identifies_stacks_and_counts():
    player_dict = {
        "QB_A": {"Position": "QB", "Team": "A", "Opponent": "B"},
        "WR_A1": {"Position": "WR", "Team": "A", "Opponent": "B"},
        "WR_A2": {"Position": "WR", "Team": "A", "Opponent": "B"},
        "WR_B1": {"Position": "WR", "Team": "B", "Opponent": "A"},
        "TE_B1": {"Position": "TE", "Team": "B", "Opponent": "A"},
        "RB_A1": {"Position": "RB", "Team": "A", "Opponent": "B"},
    }
    lineup = list(player_dict.keys())

    result = analyze_lineup(lineup, player_dict)

    assert result["presence"]["QB+WR"] == 1
    assert result["presence"]["QB+WR+OppWR"] == 1
    assert result["presence"]["RB+WR same-team"] == 1
    assert result["counts"]["QB+WR"] == 2
    assert result["counts"]["QB+WR+OppWR"] == 2
    assert result["bucket"] == "QB+WR+OppWR"


def test_analyze_lineup_no_stack_bucket():
    player_dict = {
        "QB_A": {"Position": "QB", "Team": "A", "Opponent": "B"},
        "WR_C1": {"Position": "WR", "Team": "C", "Opponent": "D"},
        "RB_E1": {"Position": "RB", "Team": "E", "Opponent": "F"},
        "TE_G1": {"Position": "TE", "Team": "G", "Opponent": "H"},
    }
    lineup = list(player_dict.keys())

    result = analyze_lineup(lineup, player_dict)

    assert result["bucket"] == "No Stack"
