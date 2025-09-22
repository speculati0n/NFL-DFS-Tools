import os
import sys
import csv
import pytest
import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from nfl_optimizer import NFL_Optimizer
from lineup_writer_patch import HEADER, Player as WriterPlayer, _extract_slots


TARGET_HEADER = [
    "QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST",
    "Salary","Fpts Proj","Fpts Used","Fpts Act","Ceiling",
    "Own. Sum","Own. Product","STDDEV","Players vs DST","Stack"
]


def test_output_includes_players_vs_dst_column():
    opt = NFL_Optimizer(site="dk", num_lineups=1, num_uniques=1)
    opt.optimize()
    path, _ = opt.output()
    with open(path) as f:
        rows = list(csv.reader(f))

    assert rows[1][8] and not rows[1][8].isdigit()
    assert rows[1][17].isdigit()
    assert rows[1][18] != ""
    # salary equals sum of player salaries
    id_to_salary = {int(v["ID"]): v["Salary"] for v in opt.player_dict.values()}
    player_ids = [int(x.split("(")[-1].rstrip(")")) for x in rows[1][:9]]
    expected_salary = sum(id_to_salary[i] for i in player_ids)
    assert float(rows[1][9]) == expected_salary


def test_writer_raises_without_dst():
    opt = NFL_Optimizer(site="dk", num_lineups=1, num_uniques=1)
    opt.optimize()
    lineup, fpts_used = opt.lineups[0]
    lineup = [p for p in lineup if opt.player_dict[p]["Position"] != "DST"]
    opt.lineups[0] = (lineup, fpts_used)
    with pytest.raises(AssertionError):
        opt.output()


def test_extract_slots_moves_latest_start_to_flex():
    early = datetime.datetime(2025, 9, 21, 13, 0)
    late = datetime.datetime(2025, 9, 21, 16, 25)

    players = [
        WriterPlayer(name="QB", pos="QB", team="AAA", salary=7000, proj=20, start_time=early),
        WriterPlayer(name="RB1", pos="RB", team="AAA", salary=6800, proj=18, start_time=early),
        WriterPlayer(name="RB2", pos="RB", team="AAA", salary=6600, proj=17, start_time=early),
        WriterPlayer(name="WR Late", pos="WR", team="AAA", salary=7200, proj=19, start_time=late),
        WriterPlayer(name="WR2", pos="WR", team="AAA", salary=7100, proj=18, start_time=early),
        WriterPlayer(name="WR3", pos="WR", team="AAA", salary=7000, proj=17, start_time=early),
        WriterPlayer(name="WR Flex", pos="WR", team="AAA", salary=6900, proj=16, start_time=early),
        WriterPlayer(name="TE", pos="TE", team="AAA", salary=5000, proj=12, start_time=early),
        WriterPlayer(name="DST", pos="DST", team="AAA", salary=3000, proj=8, start_time=early),
    ]

    slots = _extract_slots(players)
    assert slots["FLEX"].name == "WR Late"
    assert slots["WR1"].name == "WR Flex"
