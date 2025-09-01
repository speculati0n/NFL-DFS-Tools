import os
import sys

# Ensure src modules are importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from dfs_rl.utils.data import load_week_folder
from dfs_rl.arena import run_tournament
from backtesting.backtester import backtest_week


def test_arena_includes_projection_columns():
    week_dir = os.path.join("data", "historical", "2019", "2019-09-22")
    bundle = load_week_folder(week_dir)
    pool = bundle["projections"]
    df = run_tournament(pool, n_lineups_per_agent=1, train_pg=False)
    assert "projections_proj" in df.columns
    assert "projections_actpts" in df.columns


def test_backtester_includes_projection_columns():
    week_dir = os.path.join("data", "historical", "2019", "2019-09-22")
    res = backtest_week(week_dir, n_lineups_per_agent=1)
    gen = res["generated"]
    assert "projections_proj" in gen.columns
    assert "projections_actpts" in gen.columns
