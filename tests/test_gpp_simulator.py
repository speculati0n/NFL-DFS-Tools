import sys, os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from nfl_gpp_simulator import NFL_GPP_Simulator


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

