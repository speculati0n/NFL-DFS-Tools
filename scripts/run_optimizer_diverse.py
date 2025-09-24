"""Generate a diversified lineup portfolio using the existing optimiser."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

from src.anti_cannibalizer import Candidate, DiversityRules, diversify_portfolio

try:  # Optimiser class name varies between forks.
    from src.nfl_optimizer import NFLClassicOptimizer as _Optimizer  # type: ignore
except ImportError:  # pragma: no cover - fall back to legacy name
    from src.nfl_optimizer import NFL_Optimizer as _Optimizer  # type: ignore

try:
    from src.stack_metrics import analyze_lineup
except Exception:  # pragma: no cover - stack detection optional when metadata missing
    analyze_lineup = None  # type: ignore


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
CONFIG_PATHS = [
    os.path.join(ROOT_DIR, "config.json"),
    os.path.join(ROOT_DIR, "sample.config.json"),
]


def _load_config() -> Dict:
    for path in CONFIG_PATHS:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    return {}


def _format_player(rec: Dict, site: str) -> str:
    name = str(rec.get("Name") or rec.get("name") or rec.get("PlayerName") or "").strip()
    pid = rec.get("ID") or rec.get("Id") or rec.get("PlayerID")
    if pid in ("", 0, "0"):
        pid = None
    if pid is not None:
        pid = str(pid).strip()
    if not name:
        name = pid or "Unknown"
    if site.lower() == "dk":
        return f"{name} ({pid})" if pid else name
    if site.lower() == "fd":
        return f"{pid}:{name}" if pid else name
    return f"{name} ({pid})" if pid else name


def _ownership(rec: Dict) -> Optional[float]:
    for key in ("Own%", "Own", "Ownership", "ProjOwn", "ProjOwn%", "ProjectedOwn", "projected_ownership"):
        if key in rec and rec[key] not in (None, ""):
            try:
                val = float(str(rec[key]).replace("%", ""))
            except Exception:
                continue
            if math.isnan(val):
                continue
            if val < 0:
                val = 0.0
            if val > 1.0:
                val = val / 100.0
            return max(min(val, 1.0), 0.0)
    return None


def _team(rec: Dict) -> Optional[str]:
    for key in ("TeamAbbrev", "Team", "team", "TeamAbbreviation"):
        val = rec.get(key)
        if val:
            return str(val).upper().strip()
    return None


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _determine_lineup_target(config: Dict, diversity_cfg: Dict) -> int:
    raw = diversity_cfg.get("lineup_count") or config.get("num_lineups")
    try:
        val = int(raw)
    except Exception:
        val = DiversityRules().lineup_count
    return max(val, 1)


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


def _write_candidate_csv(optimizer: _Optimizer, site: str) -> List[List[str]]:
    lineup_keys: List[List[str]] = []
    candidate_path = os.path.join(OUTPUT_DIR, "optimized_lineups.csv")
    with open(candidate_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Lineup"] + [str(i) for i in range(1, 10)])
        for idx, (players, _) in enumerate(optimizer.lineups):
            try:
                ordered = optimizer.sort_lineup(list(players))
            except Exception:
                ordered = list(players)
            display = []
            for key in ordered:
                rec = optimizer.player_dict.get(key, {})
                display.append(_format_player(rec, site))
            lineup_keys.append(ordered)
            writer.writerow([idx + 1] + display)
    return lineup_keys


def _build_candidates(
    optimizer: _Optimizer,
    lineup_keys: List[List[str]],
) -> List[Candidate]:
    candidate_path = os.path.join(OUTPUT_DIR, "optimized_lineups.csv")
    candidates: List[Candidate] = []
    with open(candidate_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        headers = next(reader, None)
        for idx, row in enumerate(reader):
            players = [cell.strip() for cell in row[1:10] if cell.strip()]
            keys = lineup_keys[idx] if idx < len(lineup_keys) else []
            owners: List[Optional[float]] = []
            teams: List[Optional[str]] = []
            if keys:
                for key in keys:
                    rec = optimizer.player_dict.get(key, {})
                    owners.append(_ownership(rec))
                    teams.append(_team(rec))
            if len(owners) < len(players):
                owners.extend([None] * (len(players) - len(owners)))
            if len(teams) < len(players):
                teams.extend([None] * (len(players) - len(teams)))
            stack_label = None
            if analyze_lineup and keys:
                try:
                    metrics = analyze_lineup(keys, optimizer.player_dict)
                except TypeError:
                    metrics = analyze_lineup(keys)  # type: ignore[call-arg]
                except Exception:
                    metrics = None
                if isinstance(metrics, dict):
                    stack_label = metrics.get("bucket") or metrics.get("label")
            candidates.append(
                Candidate(
                    players=players,
                    owners=owners,
                    teams=teams,
                    stack_label=stack_label,
                )
            )
    return candidates


def _write_diversified_csv(lineups: Iterable[List[str]], filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Lineup"] + [str(i) for i in range(1, 10)])
        for idx, lineup in enumerate(lineups, start=1):
            writer.writerow([idx] + list(lineup))
    return path


def main() -> None:
    _ensure_output_dir()
    config = _load_config()
    diversity_cfg = config.get("diversity", {}) or {}

    site = str(config.get("site", "dk"))
    num_uniques = int(config.get("num_uniques", 1))
    lineup_target = _determine_lineup_target(config, diversity_cfg)
    pool_factor = float(diversity_cfg.get("pool_factor", 4.0))

    optimizer = _Optimizer(
        site=site,
        num_lineups=lineup_target,
        num_uniques=num_uniques,
        profile=config.get("profile"),
        pool_factor=max(pool_factor, 1.0),
    )

    desired_final = optimizer.num_lineups or lineup_target
    if desired_final <= 0:
        desired_final = lineup_target
    pool_size = max(int(math.ceil(desired_final * pool_factor)), desired_final)
    optimizer.num_lineups = pool_size
    if desired_final:
        ratio = pool_size / max(desired_final, 1)
        adjust = pool_factor / max(ratio, 1.0)
        optimizer.pool_factor = max(adjust, 1.0)

    optimizer.optimize()

    lineup_keys = _write_candidate_csv(optimizer, site)
    candidates = _build_candidates(optimizer, lineup_keys)

    rules = _build_rules(diversity_cfg, desired_final)
    portfolio = diversify_portfolio(candidates, rules)

    diversified_path = _write_diversified_csv(portfolio.lineups, "optimized_lineups_diverse.csv")

    audit = {
        "rules": asdict(rules),
        "metrics": portfolio.metrics,
        "selected_count": len(portfolio.lineups),
        "initial_candidate_count": len(candidates),
        "taken_indices": portfolio.taken_idx,
        "reasons_rejected": portfolio.reasons_rejected,
        "candidate_pool_path": os.path.join("output", "optimized_lineups.csv"),
        "diversified_path": os.path.relpath(diversified_path, ROOT_DIR),
        "pool_size": pool_size,
        "requested_lineups": desired_final,
        "pool_factor": pool_factor,
    }
    audit_path = os.path.join(OUTPUT_DIR, "diversity_audit.json")
    with open(audit_path, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)

    print(f"Wrote diversified lineups to {diversified_path}")
    print(f"Audit saved to {audit_path}")


if __name__ == "__main__":
    main()
