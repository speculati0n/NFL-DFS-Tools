import os
import random
import pytest

from dfs.constraints import (
    Player,
    Lineup,
    action_mask_for_slot,
    validate_lineup,
    DEFAULT_SALARY_CAP,
)
from dfs_rl.arena import run_tournament
from dfs_rl.utils.data import load_week_folder

@pytest.fixture(scope="module")
def sample_generated_lineups():
    week = "2019-09-22"
    bundle = load_week_folder(week)
    pool = bundle["projections"]
    df = run_tournament(pool, n_lineups_per_agent=5, train_pg=False, min_salary_pct=0.99)

    # map name to Player
    mapping = {}
    for idx, row in pool.iterrows():
        p = Player(
            id=str(idx),
            name=row["name"],
            pos=row["pos"],
            team=row.get("team"),
            opp=row.get("opp"),
            salary=int(row["salary"]),
            proj=float(row.get("projections_proj", 0.0)),
        )
        mapping[p.name] = p

    lineups = []
    for _, row in df.iterrows():
        lu = Lineup(
            QB=mapping[row["QB"]],
            RB1=mapping[row["RB1"]],
            RB2=mapping[row["RB2"]],
            WR1=mapping[row["WR1"]],
            WR2=mapping[row["WR2"]],
            WR3=mapping[row["WR3"]],
            TE=mapping[row["TE"]],
            FLEX=mapping[row["FLEX"]],
            DST=mapping[row["DST"]],
        )
        lineups.append(lu)
    return lineups


def test_min_spend_in_backtester(sample_generated_lineups):
    for lu in sample_generated_lineups:
        assert validate_lineup(lu, cap=DEFAULT_SALARY_CAP, min_pct=0.99)


def test_action_mask_has_valid_option():
    # Construct small player pool with feasible lineup
    pool_by_pos = {
        "QB": [Player("1", "QB1", "QB", None, None, 6000, 10.0)],
        "RB": [
            Player("2", "RB1", "RB", None, None, 8000, 10.0),
            Player("3", "RB2", "RB", None, None, 7500, 10.0),
            Player("4", "RB3", "RB", None, None, 5000, 10.0),
        ],
        "WR": [
            Player("5", "WR1", "WR", None, None, 7000, 10.0),
            Player("6", "WR2", "WR", None, None, 6500, 10.0),
            Player("7", "WR3", "WR", None, None, 6000, 10.0),
        ],
        "TE": [Player("8", "TE1", "TE", None, None, 3500, 10.0)],
        "DST": [Player("9", "DST1", "DST", None, None, 500, 5.0)],
    }
    lineup = Lineup(QB=pool_by_pos["QB"][0])
    used_ids = {pool_by_pos["QB"][0].id}
    mask = action_mask_for_slot("RB1", lineup, pool_by_pos, used_ids, cap=DEFAULT_SALARY_CAP, min_pct=0.99)
    assert any(mask.values())
