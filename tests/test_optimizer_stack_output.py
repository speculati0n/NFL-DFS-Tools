import os
import sys
import csv
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from nfl_optimizer import NFL_Optimizer


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
    assert rows[0] == TARGET_HEADER
    assert len(rows[1]) == len(TARGET_HEADER)
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
