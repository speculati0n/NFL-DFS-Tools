"""Run the classic GPP simulator after applying portfolio diversity guards."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

from src.anti_cannibalizer import Candidate, DiversityRules, diversify_portfolio

try:  # Simulator class name differs between historical versions
    from src.nfl_gpp_simulator import GppSimulator as _Simulator  # type: ignore
except ImportError:  # pragma: no cover
    from src.nfl_gpp_simulator import NFL_GPP_Simulator as _Simulator  # type: ignore

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")
CONFIG_PATHS = [
    os.path.join(ROOT_DIR, "config.json"),
    os.path.join(ROOT_DIR, "sample.config.json"),
]

PRIORITY_CSVS = [
    "optimized_lineups_diverse.csv",
    "optimized_lineups.csv",
    "saved_lineups.csv",
]


def _load_config() -> Dict:
    for path in CONFIG_PATHS:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    return {}


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _build_rules(div_cfg: Dict, lineup_target: int) -> DiversityRules:
    rules = DiversityRules(lineup_count=lineup_target)
    if "max_shared_players" in div_cfg and div_cfg["max_shared_players"] is not None:
        try:
            rules.max_shared_players = int(div_cfg["max_shared_players"])
        except Exception:
            pass
    if "min_jaccard_distance" in div_cfg and div_cfg["min_jaccard_distance"] is not None:
        try:
            rules.min_jaccard_distance = float(div_cfg["min_jaccard_distance"])
        except Exception:
            pass
    if "per_player_cap" in div_cfg:
        rules.per_player_cap = (
            float(div_cfg["per_player_cap"]) if div_cfg["per_player_cap"] is not None else None
        )
    if "per_team_cap" in div_cfg:
        rules.per_team_cap = (
            float(div_cfg["per_team_cap"]) if div_cfg["per_team_cap"] is not None else None
        )
    if "min_stack_mix" in div_cfg and div_cfg["min_stack_mix"]:
        rules.min_stack_mix = {str(k): float(v) for k, v in div_cfg["min_stack_mix"].items()}
    if "max_stack_mix" in div_cfg and div_cfg["max_stack_mix"]:
        rules.max_stack_mix = {str(k): float(v) for k, v in div_cfg["max_stack_mix"].items()}
    return rules


def _find_source_csv() -> Optional[str]:
    for name in PRIORITY_CSVS:
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(path):
            return path
    return None


def _load_lineups(path: str) -> List[List[str]]:
    lineups: List[List[str]] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for row in reader:
            players = [cell.strip() for cell in row[1:10] if cell.strip()]
            if players:
                lineups.append(players)
    return lineups


def _write_csv(lineups: Iterable[List[str]], path: str, include_index: bool = True) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if include_index:
            writer.writerow(["Lineup"] + [str(i) for i in range(1, 10)])
            for idx, lineup in enumerate(lineups, start=1):
                writer.writerow([idx] + list(lineup))
        else:
            writer.writerow(["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"])
            for lineup in lineups:
                writer.writerow(list(lineup))


def main() -> None:
    config = _load_config()
    diversity_cfg = config.get("diversity", {}) or {}

    source_csv = _find_source_csv()
    if not source_csv:
        raise FileNotFoundError("No lineup CSV found in output/. Run the optimizer first.")

    raw_lineups = _load_lineups(source_csv)
    if not raw_lineups:
        raise ValueError(f"No lineups parsed from {source_csv}")

    requested = int(config.get("num_sim_lineups", len(raw_lineups)))
    requested = max(1, min(requested, len(raw_lineups)))

    rules = _build_rules(diversity_cfg, requested)
    candidates = [Candidate(players=lu) for lu in raw_lineups]
    portfolio = diversify_portfolio(candidates, rules)

    diversified_lineups = portfolio.lineups
    diversified_path = os.path.join(OUTPUT_DIR, "simulator_diverse_lineups.csv")
    _write_csv(diversified_lineups, diversified_path, include_index=True)

    site = str(config.get("site", "dk"))
    uploads_site_dir = os.path.join(UPLOAD_DIR, site)
    os.makedirs(uploads_site_dir, exist_ok=True)
    tournament_path = os.path.join(uploads_site_dir, "tournament_lineups.csv")
    _write_csv(diversified_lineups, tournament_path, include_index=False)

    field_size = int(config.get("field_size", max(len(diversified_lineups), 1)))
    num_iterations = int(config.get("num_iterations", 1000))
    use_contest_data = _as_bool(config.get("use_contest_data"), False)
    use_lineup_input = _as_bool(config.get("use_lineup_input"), True)
    profile = config.get("profile")
    pool_factor = float(config.get("pool_factor", 5.0))

    simulator = _Simulator(
        site,
        field_size,
        num_iterations,
        use_contest_data,
        use_lineup_input,
        profile=profile,
        pool_factor=pool_factor,
    )
    simulator.generate_field_lineups()
    simulator.run_tournament_simulation()
    sim_outputs = simulator.output()

    audit = {
        "rules": asdict(rules),
        "metrics": portfolio.metrics,
        "selected_count": len(diversified_lineups),
        "initial_candidate_count": len(raw_lineups),
        "taken_indices": portfolio.taken_idx,
        "reasons_rejected": portfolio.reasons_rejected,
        "source_csv": os.path.relpath(source_csv, ROOT_DIR),
        "diversified_path": os.path.relpath(diversified_path, ROOT_DIR),
        "tournament_path": os.path.relpath(tournament_path, ROOT_DIR),
        "simulator_outputs": sim_outputs,
    }
    audit_path = os.path.join(OUTPUT_DIR, "simulator_diversity_audit.json")
    with open(audit_path, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)

    print(f"Diversified simulator lineups saved to {diversified_path}")
    print(f"Tournament input written to {tournament_path}")
    print(f"Simulator outputs: {sim_outputs}")
    print(f"Audit saved to {audit_path}")


if __name__ == "__main__":
    main()
