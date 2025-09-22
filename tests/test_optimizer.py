import sys
import os
import pytest
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from nfl_optimizer import NFL_Optimizer, Player


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


def test_optimizer_skips_stack_when_opponent_missing():
    opt = NFL_Optimizer(site="dk", num_lineups=1, num_uniques=1)

    team = next(iter(opt.players_by_team.keys()))
    opp_team = opt.players_by_team[team]["QB"][0]["Opponent"]

    if opp_team in opt.players_by_team:
        del opt.players_by_team[opp_team]

    # Should not raise KeyError when opponent team is missing
    opt.optimize()


def test_select_slot_players_promotes_latest_start():
    opt = NFL_Optimizer(site="dk", num_lineups=0, num_uniques=1)

    early = datetime.datetime(2025, 9, 21, 13, 0)
    late = datetime.datetime(2025, 9, 21, 16, 25)

    players = [
        Player(name="QB", pos="QB", team="AAA", salary=7000, proj=20, start_time=early),
        Player(name="RB1", pos="RB", team="AAA", salary=6800, proj=18, start_time=early),
        Player(name="RB2", pos="RB", team="AAA", salary=6600, proj=17, start_time=early),
        Player(name="WR Late", pos="WR", team="AAA", salary=7200, proj=19, start_time=late),
        Player(name="WR2", pos="WR", team="AAA", salary=7100, proj=18, start_time=early),
        Player(name="WR3", pos="WR", team="AAA", salary=7000, proj=17, start_time=early),
        Player(name="WR Flex", pos="WR", team="AAA", salary=6900, proj=16, start_time=early),
        Player(name="TE", pos="TE", team="AAA", salary=5000, proj=12, start_time=early),
        Player(name="DST", pos="DST", team="AAA", salary=3000, proj=8, start_time=early),
    ]

    slots = opt.select_slot_players(players)
    assert slots["FLEX"].name == "WR Late"
    assert slots["WR1"].name == "WR Flex"
