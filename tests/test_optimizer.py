import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from nfl_optimizer import NFL_Optimizer


def test_optimizer_handles_players_removed_from_player_dict():
    opt = NFL_Optimizer(site="dk", num_lineups=1, num_uniques=1)

    # choose an existing opponent team to ensure qb vs dst rule executes
    opponent_team = next(iter(opt.players_by_team.keys()))

    dummy_player = {
        "Fpts": 10,
        "Position": "QB",
        "ID": 0,
        "Salary": 5000,
        "Name": "Dummy QB",
        "Matchup": f"AAA @ {opponent_team}",
        "Team": "AAA",
        "Opponent": opponent_team,
        "Ownership": 0,
        "Ceiling": 15,
        "StdDev": 3,
    }

    key = ("dummy qb", "QB", "AAA")
    opt.player_dict[key] = dummy_player
    if "AAA" not in opt.players_by_team:
        opt.players_by_team["AAA"] = {"QB": [], "RB": [], "WR": [], "TE": [], "DST": []}
    opt.players_by_team["AAA"]["QB"].append(dummy_player)

    # Re-run assertion to remove players without ids
    opt.assertPlayerDict()

    # Ensure player removed from players_by_team
    assert dummy_player not in opt.players_by_team["AAA"]["QB"]

    # Should not raise KeyError
    opt.optimize()
