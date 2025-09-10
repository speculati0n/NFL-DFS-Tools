import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import nfl_showdown_simulator
from nfl_showdown_simulator import NFL_Showdown_Simulator, _normalize_name


def test_generational_suffixes_match(monkeypatch):
    # Avoid heavy optimizer/correlation work in tests
    monkeypatch.setattr(NFL_Showdown_Simulator, "get_optimal", lambda self: None)
    monkeypatch.setattr(NFL_Showdown_Simulator, "load_correlation_rules", lambda self: None)

    original_load_config = NFL_Showdown_Simulator.load_config

    def patched_load_config(self):
        original_load_config(self)
        self.config.setdefault("allow_def_vs_qb_cpt", False)

    monkeypatch.setattr(NFL_Showdown_Simulator, "load_config", patched_load_config)

    data_dir = os.path.join(os.path.dirname(__file__), "data")

    def fake_get_data_path(site, filename):
        return os.path.join(data_dir, filename)

    monkeypatch.setattr(nfl_showdown_simulator, "get_data_path", fake_get_data_path)

    sim = NFL_Showdown_Simulator(
        site="dk",
        field_size=1,
        num_iterations=1,
        use_contest_data=False,
        use_lineup_input=False,
    )

    players = [
        ("Calvin Austin III", "PIT", "1"),
        ("Travis Etienne Jr.", "JAX", "2"),
    ]

    for name, team, pid in players:
        norm = _normalize_name(name)
        key = (norm, "FLEX", team)
        assert key in sim.player_dict
        assert sim.player_dict[key]["ID"] == pid
        # also ensure CPT entry uses same ID
        key_cpt = (norm, "CPT", team)
        assert key_cpt in sim.player_dict
        assert sim.player_dict[key_cpt]["ID"] == pid
