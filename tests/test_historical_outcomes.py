import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from dfs_rl.utils.historical_outcomes import (
    attach_historical_outcomes,
    load_outcomes_for_date,
    ROSTER_SLOTS,
)


def test_attach_historical_outcomes_handles_existing_columns():
    base_dir = os.path.join("data", "historical")
    date_like = "2019-09-22"
    hist = load_outcomes_for_date(base_dir, date_like)
    assert not hist.empty

    lineup_cols = ROSTER_SLOTS + ["contest_id"]
    generated = hist[lineup_cols].head(1).copy()

    first = attach_historical_outcomes(generated, date_like, base_dir)
    second = attach_historical_outcomes(first, date_like, base_dir)

    assert second["contest_rank"].iloc[0] == first["contest_rank"].iloc[0]


def test_attach_historical_outcomes_preserves_existing_when_no_match():
    base_dir = os.path.join("data", "historical")
    date_like = "2019-09-08"
    lineup = {s: f"{s}_player" for s in ROSTER_SLOTS}
    lineup.update({"contest_rank": 7, "amount_won": 0.0})
    generated = pd.DataFrame([lineup])

    out = attach_historical_outcomes(generated, date_like, base_dir)
    assert out["contest_rank"].iloc[0] == 7
    assert out["matches_found"].iloc[0] == 0
