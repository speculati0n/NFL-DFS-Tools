import sys, os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from nfl_gpp_simulator import NFL_GPP_Simulator
import nfl_gpp_simulator


def test_lamar_jackson_gets_id_without_mismatch(monkeypatch, capsys):
    # Avoid heavy optimizer/correlation work in tests
    monkeypatch.setattr(NFL_GPP_Simulator, "get_optimal", lambda self: None)
    monkeypatch.setattr(NFL_GPP_Simulator, "load_correlation_rules", lambda self: None)

    sim = NFL_GPP_Simulator(
        site="dk",
        field_size=10,
        num_iterations=1,
        use_contest_data=False,
        use_lineup_input=False,
    )

    captured = capsys.readouterr()
    assert "lamar jackson name mismatch" not in captured.out.lower()

    key = ("lamar jackson", str(["QB"]), "BAL")
    assert key in sim.player_dict
    assert sim.player_dict[key]["ID"] not in ("", None, 0)


def test_matchups_populated_and_lineups_generate(monkeypatch):
    # Skip heavy optimizer/correlation setup
    monkeypatch.setattr(NFL_GPP_Simulator, "get_optimal", lambda self: None)
    monkeypatch.setattr(NFL_GPP_Simulator, "load_correlation_rules", lambda self: None)

    class DummyPool:
        def __init__(self, *args, **kwargs):
            self._processes = 1
            self._state = "RUN"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def starmap(self, func, iterable):
            return [func(*args) for args in iterable]

        def close(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(nfl_gpp_simulator.mp, "Pool", lambda *a, **k: DummyPool())
    monkeypatch.setattr(NFL_GPP_Simulator, "sort_lineup_by_start_time", lambda self, lu: lu)

    sim = NFL_GPP_Simulator(
        site="dk",
        field_size=1,
        num_iterations=1,
        use_contest_data=False,
        use_lineup_input=False,
    )

    # All players should have a populated matchup derived from projections team/opp
    assert all(p.get("Matchup") for p in sim.player_dict.values())

    # Reduce randomness for deterministic test
    sim.pct_field_using_stacks = 0
    sim.pct_field_double_stacks = 0
    sim.optimal_score = 200
    sim.generate_field_lineups()
    assert len(sim.field_lineups) > 0
