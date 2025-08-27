import sys, os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import csv
from pathlib import Path

import nfl_gpp_simulator as ngs
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


def test_stacks_exclude_unmatched_qb(monkeypatch, tmp_path):
    monkeypatch.setattr(NFL_GPP_Simulator, "get_optimal", lambda self: None)
    monkeypatch.setattr(NFL_GPP_Simulator, "load_correlation_rules", lambda self: None)

    src_dir = Path(__file__).resolve().parent.parent / "dk_data"
    player_src = src_dir / "player_ids.csv"

    player_dst = tmp_path / "player_ids.csv"
    with open(player_src, newline="", encoding="utf-8-sig") as inp, open(
        player_dst, "w", newline="", encoding="utf-8-sig"
    ) as out:
        reader = csv.DictReader(inp)
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["displayname"] != "Lamar Jackson":
                writer.writerow(row)

    original_get_data_path = ngs.get_data_path

    def fake_get_data_path(site, filename):
        if filename == "player_ids.csv":
            return str(player_dst)
        return original_get_data_path(site, filename)

    monkeypatch.setattr(ngs, "get_data_path", fake_get_data_path)

    sim = NFL_GPP_Simulator(
        site="dk",
        field_size=10,
        num_iterations=1,
        use_contest_data=False,
        use_lineup_input=False,
    )

    assert "BAL" not in sim.stacks_dict
