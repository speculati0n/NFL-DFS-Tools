"""Portfolio diversity guards for NFL DFS lineups.

This module adds a light-weight selection layer that can be used on top of
existing optimisation/simulation flows.  Given an oversized pool of candidate
lineups, we greedily construct a diversified portfolio subject to overlap,
exposure and stack mix rules.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence

try:  # ``stack_metrics`` lives in the repo; guard for standalone usage/tests.
    from . import stack_metrics  # type: ignore
except Exception:  # pragma: no cover - fallback when used outside the package
    stack_metrics = None  # type: ignore


@dataclass
class DiversityRules:
    """Configuration for pruning a candidate lineup pool."""

    max_shared_players: int = 6
    min_jaccard_distance: float = 0.20
    per_player_cap: Optional[float] = 0.45
    per_team_cap: Optional[float] = 0.40
    min_stack_mix: Optional[Dict[str, float]] = None
    max_stack_mix: Optional[Dict[str, float]] = None
    lineup_count: int = 20


@dataclass
class Candidate:
    """Lineup candidate produced by the optimiser or an upstream process."""

    players: List[str]
    owners: Optional[List[Optional[float]]] = None
    teams: Optional[List[Optional[str]]] = None
    stack_label: Optional[str] = None
    score: float = 0.0


@dataclass
class PortfolioResult:
    """Result of the greedy diversification pass."""

    lineups: List[List[str]]
    taken_idx: List[int]
    metrics: Dict[str, float]
    reasons_rejected: Dict[str, int]


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Return the Jaccard similarity of two lineups."""

    set_a = set(a)
    set_b = set(b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def hamming_shared(a: Sequence[str], b: Sequence[str]) -> int:
    """Count of shared players between two unordered lineups."""

    return len(set(a) & set(b))


def lineup_score_ownership_product(owners: Optional[Sequence[Optional[float]]]) -> float:
    """Return a scarcity score using the product of ownerships.

    Ownership values are expected on the 0..1 scale.  Values outside this range
    are coerced by dividing by 100 if they look like percentages and by clamping
    into the [1e-6, 1] interval before taking the product.
    """

    if not owners:
        return 0.0

    product = 1.0
    has_value = False
    for raw in owners:
        if raw is None:
            continue
        try:
            val = float(raw)
        except Exception:
            continue
        if math.isnan(val):
            continue
        if val < 0:
            val = 0.0
        if val > 1.0:
            val = val / 100.0
        val = max(min(val, 1.0), 1e-6)
        product *= val
        has_value = True
    if not has_value:
        return 0.0
    return -math.log(product)


def _infer_stack_label(players: Sequence[str], supplied: Optional[str]) -> str:
    """Return a stack label using ``stack_metrics`` when available."""

    if supplied:
        return supplied
    if stack_metrics is None:
        return "No Stack"
    try:
        metrics = stack_metrics.analyze_lineup(players)  # type: ignore[arg-type]
    except TypeError:
        # Older signature expects the player dictionary as the second argument;
        # fall back to an empty dict which yields the "No Stack" bucket.
        try:
            metrics = stack_metrics.analyze_lineup(players, {})  # type: ignore[arg-type]
        except Exception:
            return "No Stack"
    except Exception:
        return "No Stack"
    label = None
    if isinstance(metrics, dict):
        label = metrics.get("label") or metrics.get("bucket")
    return label or "No Stack"


def _cap_count(cap: Optional[float], total: int) -> Optional[int]:
    if cap is None:
        return None
    try:
        limit = int(math.floor(float(cap) * total))
    except Exception:
        return None
    return max(limit, 0)


def _pairwise_jaccard_mean(lineup_sets: List[set]) -> float:
    if len(lineup_sets) < 2:
        return 0.0
    total = 0.0
    count = 0
    for a, b in combinations(lineup_sets, 2):
        union = a | b
        if not union:
            continue
        total += len(a & b) / len(union)
        count += 1
    return total / count if count else 0.0


def diversify_portfolio(candidates: List[Candidate], rules: DiversityRules) -> PortfolioResult:
    """Greedily select a diversified subset of the candidate lineups."""

    if rules.lineup_count < 0:
        rules.lineup_count = 0

    processed: List[Dict[str, object]] = []
    for idx, cand in enumerate(candidates):
        players = list(cand.players or [])
        owners = list(cand.owners or [])
        teams = list(cand.teams or [])
        label = _infer_stack_label(players, cand.stack_label)
        score = cand.score
        if score == 0:
            score = lineup_score_ownership_product(owners)
        team_set = {str(t).upper() for t in teams if t}
        processed.append(
            {
                "index": idx,
                "players": players,
                "owners": owners,
                "teams": teams,
                "team_set": team_set,
                "label": label or "No Stack",
                "score": float(score),
            }
        )

    processed.sort(key=lambda item: (-item["score"], item["index"]))  # type: ignore[index]

    max_shared = rules.max_shared_players if rules.max_shared_players is not None else None
    jaccard_threshold = None
    if rules.min_jaccard_distance is not None:
        try:
            jaccard_threshold = 1.0 - float(rules.min_jaccard_distance)
        except Exception:
            jaccard_threshold = None
    player_cap_limit = _cap_count(rules.per_player_cap, rules.lineup_count)
    team_cap_limit = _cap_count(rules.per_team_cap, rules.lineup_count)

    selected_lineups: List[List[str]] = []
    selected_sets: List[set] = []
    taken_indices: List[int] = []
    taken_set = set()
    player_counts: Counter = Counter()
    team_counts: Counter = Counter()
    stack_counts: Counter = Counter()
    reasons = defaultdict(int)

    def _attempt_take(entry: Dict[str, object]) -> bool:
        if len(selected_lineups) >= rules.lineup_count > 0:
            return False
        players = entry["players"]  # type: ignore[assignment]
        label = entry["label"]  # type: ignore[assignment]
        team_set = entry["team_set"]  # type: ignore[assignment]
        idx = entry["index"]  # type: ignore[assignment]

        unique_players = set(players)
        if player_cap_limit is not None and player_cap_limit >= 0:
            for player in unique_players:
                if player_counts[player] >= player_cap_limit:
                    key = f"player_cap:{player}"
                    reasons[key] += 1
                    return False
        if team_cap_limit is not None and team_cap_limit >= 0 and team_set:
            for team in team_set:
                if team_counts[team] >= team_cap_limit:
                    key = f"team_cap:{team}"
                    reasons[key] += 1
                    return False

        if rules.max_stack_mix and label in rules.max_stack_mix:
            try:
                allowed = math.floor(float(rules.max_stack_mix[label]) * rules.lineup_count)
            except Exception:
                allowed = None
            if allowed is not None and allowed >= 0 and stack_counts[label] >= allowed:
                key = f"stack_max:{label}"
                reasons[key] += 1
                return False

        if max_shared is not None or jaccard_threshold is not None:
            for existing in selected_sets:
                if max_shared is not None and max_shared >= 0:
                    shared = len(existing & unique_players)
                    if shared > max_shared:
                        key = f"overlap:{shared}"
                        reasons[key] += 1
                        return False
                if jaccard_threshold is not None:
                    union = existing | unique_players
                    if not union:
                        continue
                    jac = len(existing & unique_players) / len(union)
                    if jac > jaccard_threshold:
                        key = f"jaccard:{jac:.2f}"
                        reasons[key] += 1
                        return False

        selected_lineups.append(list(players))
        selected_sets.append(unique_players)
        taken_indices.append(int(idx))
        taken_set.add(int(idx))
        for player in unique_players:
            player_counts[player] += 1
        for team in team_set:
            team_counts[team] += 1
        stack_counts[label] += 1
        return True

    for entry in processed:
        if len(selected_lineups) >= rules.lineup_count > 0:
            break
        _attempt_take(entry)

    if rules.min_stack_mix:
        for label, frac in rules.min_stack_mix.items():
            try:
                required = int(math.ceil(float(frac) * rules.lineup_count))
            except Exception:
                required = 0
            required = max(0, min(required, rules.lineup_count))
            guard = 0
            while stack_counts.get(label, 0) < required and len(selected_lineups) < rules.lineup_count:
                added = False
                for entry in processed:
                    if entry["label"] != label:  # type: ignore[index]
                        continue
                    if entry["index"] in taken_set:  # type: ignore[operator]
                        continue
                    if _attempt_take(entry):
                        added = True
                        break
                if not added:
                    break
                guard += 1
                if guard > len(processed):
                    break
            if stack_counts.get(label, 0) < required:
                deficit = required - stack_counts.get(label, 0)
                if deficit > 0:
                    reasons[f"stack_min_unmet:{label}"] += deficit

    metrics: Dict[str, float] = {
        "pairwise_jaccard_mean": _pairwise_jaccard_mean(selected_sets),
        "player_HHI": sum((count / len(selected_lineups)) ** 2 for count in player_counts.values())
        if selected_lineups
        else 0.0,
        "selected_lineups": float(len(selected_lineups)),
        "requested_lineups": float(rules.lineup_count),
    }

    return PortfolioResult(
        lineups=selected_lineups,
        taken_idx=taken_indices,
        metrics=metrics,
        reasons_rejected=dict(reasons),
    )
