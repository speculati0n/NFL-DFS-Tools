import sys, os
import pytest
import csv
import datetime

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


def test_sort_lineup_swaps_latest_start_into_flex():
    sim = NFL_GPP_Simulator.__new__(NFL_GPP_Simulator)
    sim.position_map = NFL_GPP_Simulator.position_map
    sim.player_dict = {}
    sim.game_info = {}
    sim.player_start_times = {}

    base_time = datetime.datetime(2025, 9, 21, 13, 0)
    late_time = datetime.datetime(2025, 9, 21, 16, 25)

    def register(pid, pos, matchup, start_time):
        record = {
            "ID": str(pid),
            "Position": [pos],
            "Matchup": matchup,
            "StartTime": start_time,
        }
        sim.player_dict[(f"player{pid}", str([pos]), "TEAM")] = record
        sim.player_start_times[str(pid)] = start_time
        sim.game_info[matchup] = start_time

    register(1, "DST", "AAA@BBB", base_time)
    register(2, "QB", "AAA@BBB", base_time)
    register(3, "RB", "AAA@BBB", base_time)
    register(4, "RB", "AAA@BBB", base_time)
    register(5, "WR", "AAA@BBB", late_time)
    register(6, "WR", "AAA@BBB", base_time)
    register(7, "WR", "AAA@BBB", base_time)
    register(8, "TE", "AAA@BBB", base_time)
    register(9, "WR", "AAA@BBB", base_time)

    lineup = [
        "1",  # DST
        "2",  # QB
        "3",  # RB1
        "4",  # RB2
        "5",  # WR1 (late)
        "6",  # WR2
        "7",  # WR3
        "8",  # TE
        "9",  # FLEX (early WR)
    ]

    sorted_lineup = sim.sort_lineup_by_start_time(list(lineup))
    assert sorted_lineup[8] == "5"
    assert sorted_lineup[4] == "9"


def test_output_includes_stack_columns(monkeypatch):
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

    called = {"flag": False}
    original_analyze = nfl_gpp_simulator.analyze_lineup

    def wrapped_analyze_lineup(lineup, player_dict):
        assert all("Opponent" in p for p in player_dict.values())

        called["flag"] = True
        return original_analyze(lineup, player_dict)

    monkeypatch.setattr(nfl_gpp_simulator, "analyze_lineup", wrapped_analyze_lineup)

    sim = NFL_GPP_Simulator(
        site="dk",
        field_size=1,
        num_iterations=1,
        use_contest_data=False,
        use_lineup_input=False,
    )


    sim.output()

    assert called["flag"]

    lineups_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "output",
        f"dk_gpp_sim_lineups_{sim.field_size}_{sim.num_iterations}.csv",
    )

    with open(lineups_path) as f:
        rows = list(csv.reader(f))

    header = rows[0]
    data = rows[1]
    stack1_idx = header.index("Stack1 Type")
    stack2_idx = header.index("Stack2 Type")

    assert "Stack2 Type" in header
