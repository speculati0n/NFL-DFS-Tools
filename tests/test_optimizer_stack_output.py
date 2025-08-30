import os
import sys
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from nfl_optimizer import NFL_Optimizer


def test_output_includes_players_vs_dst_column():
    opt = NFL_Optimizer(site="dk", num_lineups=1, num_uniques=1)
    opt.optimize()
    path, _ = opt.output()
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0][-2] == "Players vs DST"
    assert rows[0][-1] == "Stack"
    assert rows[1][-2].isdigit()
    assert rows[1][-1] != ""
