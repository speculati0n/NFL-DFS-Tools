#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified stack analysis utilities.

- Presence: fraction of lineups containing >=1 instance of a stack.
- Multiplicity: average count per lineup for that stack.
- Bucket: exclusive stack label per lineup via precedence ladder.

This module computes *pair* stacks and *composite* stacks so the same keys
exist across presence/multiplicity and bucket outputs. That prevents the
"presence=0, multiplicity=0, but bucket>0" mismatch.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter

__all__ = [
    "analyze_lineup",
    "compute_presence_and_counts",
    "compute_features",
    "classify_bucket",
]

# --- Lineup access helpers ---

SLOTS_ORDER = ["QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST"]

def _get(lineup: Dict[str, Any], base: str, field: str) -> Any:
    # supports QB_name / QB_team and common variants
    return lineup.get(f"{base}_{field}") or lineup.get(f"{base}{field.capitalize()}") or lineup.get(f"{base}{field}")

def _collect_players(lineup: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Normalize lineup into per-slot dicts with fields: name,pos,team,opp,slot.
    Robust to varied column casings.
    """
    players: Dict[str, Dict[str, Any]] = {}
    for slot in SLOTS_ORDER:
        name = _get(lineup, slot, "name")
        pos  = _get(lineup, slot, "pos")
        team = _get(lineup, slot, "team")
        opp  = _get(lineup, slot, "opp")
        if name is None and pos is None and team is None and opp is None:
            continue
        players[slot] = {
            "slot": slot,
            "name": name,
            "pos": (pos or "").upper(),
            "team": team,
            "opp": opp,
        }
    return players

def _underlying_pos(p: Dict[str,Any]) -> str:
    """Treat FLEX as the player's true position if provided."""
    return (p.get("pos") or "").upper()

def _lists_by_pos(players: Dict[str,Dict[str,Any]]) -> Tuple[Dict[str,Any], List[Dict[str,Any]], List[Dict[str,Any]], List[Dict[str,Any]], Dict[str,Any]]:
    qb = players.get("QB")
    rbs, wrs, tes = [], [], []
    for slot in SLOTS_ORDER:
        if slot not in players or slot in ("QB","DST"):
            continue
        p = players[slot]
        pos = _underlying_pos(p)
        if pos == "RB":
            rbs.append(p)
        elif pos == "WR":
            wrs.append(p)
        elif pos == "TE":
            tes.append(p)
    dst = players.get("DST")
    return qb, rbs, wrs, tes, dst

# --- Counting primitives ---

def _count_qb_plus(qb: Dict[str,Any], arr: List[Dict[str,Any]]) -> int:
    if not qb:
        return 0
    qb_team = qb.get("team")
    return sum(1 for p in arr if p.get("team") == qb_team)

def _count_qb_opp(qb: Dict[str,Any], arr: List[Dict[str,Any]]) -> int:
    if not qb:
        return 0
    opp = qb.get("opp")
    return sum(1 for p in arr if p.get("team") == opp)

def _count_same_team_pairs(a: List[Dict[str,Any]], b: List[Dict[str,Any]]) -> int:
    team_to_a = Counter(p.get("team") for p in a)
    team_to_b = Counter(p.get("team") for p in b)
    total = 0
    for t, ca in team_to_a.items():
        if t in team_to_b:
            total += ca * team_to_b[t]
    return total

def _count_vs_pairs(a: List[Dict[str,Any]], b: List[Dict[str,Any]]) -> int:
    total = 0
    for pa in a:
        ta, oa = pa.get("team"), pa.get("opp")
        for pb in b:
            tb, ob = pb.get("team"), pb.get("opp")
            if ta and tb and (ta == ob or oa == tb):
                total += 1
    return total

def _any_vs_dst(offense: List[Dict[str,Any]], dst: Dict[str,Any]) -> int:
    if not dst:
        return 0
    dteam = dst.get("team")
    dopp  = dst.get("opp")
    cnt = 0
    for p in offense:
        if p.get("pos") == "DST":
            continue
        if p.get("opp") == dteam or p.get("team") == dopp:
            cnt += 1
    return cnt

# --- Compatibility helpers ---

def _lineup_to_dict(lineup: List[Any], player_dict: Dict[Any, Dict[str,Any]]) -> Dict[str,Any]:
    """Convert a list-style lineup into the slot-based dict format.

    Each element of ``lineup`` is a key into ``player_dict`` which provides
    ``Position``, ``Team``, ``Opponent`` and optionally ``Name``.
    """

    res: Dict[str, Any] = {}
    rb = wr = 0
    flex_used = False
    for key in lineup:
        info = player_dict[key]
        pos = (info.get("Position") or "").upper()
        team = info.get("Team")
        opp = info.get("Opponent")
        name = info.get("Name") or str(key)

        if pos == "QB":
            slot = "QB"
        elif pos == "DST":
            slot = "DST"
        elif pos == "RB":
            rb += 1
            if rb == 1:
                slot = "RB1"
            elif rb == 2:
                slot = "RB2"
            elif not flex_used:
                slot = "FLEX"
                flex_used = True
            else:
                slot = f"RB{rb}"
        elif pos == "WR":
            wr += 1
            if wr <= 3:
                slot = f"WR{wr}"
            elif not flex_used:
                slot = "FLEX"
                flex_used = True
            else:
                slot = f"WR{wr}"
        elif pos == "TE":
            if "TE_name" not in res:
                slot = "TE"
            elif not flex_used:
                slot = "FLEX"
                flex_used = True
            else:
                slot = "TE2"
        else:
            if not flex_used:
                slot = "FLEX"
                flex_used = True
            else:
                slot = f"P{len(res)}"

        res[f"{slot}_name"] = name
        res[f"{slot}_pos"] = pos
        res[f"{slot}_team"] = team
        res[f"{slot}_opp"] = opp

    return res

# --- Public computations ---

def compute_presence_and_counts(lineup: Dict[str,Any]) -> Tuple[Dict[str,int], Dict[str,int]]:
    """
    Returns:
      flags:  dict[str,int] -> 1 if present else 0  (for presence)
      counts: dict[str,int] -> number of occurrences (for multiplicity)
    Includes *composite* stacks so keys align with buckets.
    """
    players = _collect_players(lineup)
    qb, rbs, wrs, tes, dst = _lists_by_pos(players)

    counts: Dict[str,int] = defaultdict(int)

    # QB pairs
    qb_wr = _count_qb_plus(qb, wrs)
    qb_te = _count_qb_plus(qb, tes)
    qb_opp_wr = _count_qb_opp(qb, wrs)
    qb_opp_rb = _count_qb_opp(qb, rbs)

    counts["QB+WR"] = qb_wr
    counts["QB+TE"] = qb_te
    counts["QB+OppWR"] = qb_opp_wr
    counts["QB+OppRB"] = qb_opp_rb

    # Cross-team minis
    counts["RB vs OppWR"] = _count_vs_pairs(rbs, wrs)
    counts["WR vs OppWR"] = _count_vs_pairs(wrs, wrs)
    counts["TE vs OppWR"] = _count_vs_pairs(tes, wrs)
    counts["RB+WR same-team"] = _count_same_team_pairs(rbs, wrs)

    # Composite stacks (KEYS must match bucket labels)
    counts["QB+WR+OppWR"] = (qb_wr > 0 and qb_opp_wr > 0) * max(qb_wr * qb_opp_wr, 1 if (qb_wr > 0 and qb_opp_wr > 0) else 0)
    counts["QB+WR+TE"]    = (qb_wr > 0 and qb_te > 0) * 1
    counts["QB+TE+OppWR"] = (qb_te > 0 and qb_opp_wr > 0) * max(qb_te * qb_opp_wr, 1 if (qb_te > 0 and qb_opp_wr > 0) else 0)

    # Feature-style counts
    offense = rbs + wrs + tes
    counts["Any vs DST (per player)"] = _any_vs_dst(offense, players.get("DST"))

    flags: Dict[str,int] = {k: int(v > 0) for k, v in counts.items()}
    return flags, counts

def compute_features(lineup: Dict[str,Any]) -> Dict[str,int]:
    """Non-stack features used by rewards/filters."""
    players = _collect_players(lineup)
    _, rbs, wrs, tes, _ = _lists_by_pos(players)

    flex = players.get("FLEX")
    flex_pos = _underlying_pos(flex) if flex else ""
    flags, _ = compute_presence_and_counts(lineup)

    feats = {
        "flex_pos": flex_pos,
        "flex_is_wr": int(flex_pos == "WR"),
        "flex_is_rb": int(flex_pos == "RB"),
        "flex_is_te": int(flex_pos == "TE"),
        "feat_double_te": int(len(tes) >= 2),
        "feat_any_vs_dst": flags.get("Any vs DST (per player)", 0),
        "has_qb_wr": flags.get("QB+WR", 0),
        "has_qb_te": flags.get("QB+TE", 0),
        "has_qb_oppwr": flags.get("QB+OppWR", 0),
        "has_qb_wr_te": flags.get("QB+WR+TE", 0),
        "has_qb_wr_oppwr": flags.get("QB+WR+OppWR", 0),
        "has_qb_te_oppwr": flags.get("QB+TE+OppWR", 0),
    }
    return feats

# --- Bucketing ---

BUCKET_ORDER = [
    "QB+WR+TE",
    "QB+WR+OppWR",
    "QB+TE+OppWR",
    "QB+WR",
    "QB+TE",
    "RB+WR same-team",
    "RB vs OppWR",
    "WR vs OppWR",
    "TE vs OppWR",
    "QB+OppRB",
    "No Stack"
]

def classify_bucket(flags: Dict[str,int]) -> str:
    """
    Choose one exclusive label using a precedence ladder.
    Uses the *same keys* produced in compute_presence_and_counts.
    """
    if flags.get("QB+WR+TE", 0):
        return "QB+WR+TE"
    if flags.get("QB+WR+OppWR", 0):
        return "QB+WR+OppWR"
    if flags.get("QB+TE+OppWR", 0):
        return "QB+TE+OppWR"
    if flags.get("QB+WR", 0):
        return "QB+WR"
    if flags.get("QB+TE", 0):
        return "QB+TE"
    for k in ("RB+WR same-team", "RB vs OppWR", "WR vs OppWR", "TE vs OppWR", "QB+OppRB"):
        if flags.get(k, 0):
            return k
    return "No Stack"

# --- Top-level convenience ---

def analyze_lineup(
    lineup: Dict[str, Any] | List[Any],
    player_dict: Dict[Any, Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Analyze a single lineup.

    ``lineup`` may be a dict with slot fields (QB_team, WR1_team, ...)
    or a list of player keys paired with ``player_dict``. The return value
    is backward compatible with older helpers while also exposing the new
    unified structures.
    """

    if not isinstance(lineup, dict):
        if player_dict is None:
            raise ValueError("player_dict required when lineup is a list")
        lineup = _lineup_to_dict(list(lineup), player_dict)

    flags, counts = compute_presence_and_counts(lineup)
    bucket = classify_bucket(flags)
    bucket_onehot = {bucket: 1.0}

    return {
        "presence": flags,
        "counts": counts,            # legacy name
        "multiplicity": counts,
        "bucket": bucket,            # legacy name (label)
        "bucket_label": bucket,
        "bucket_onehot": bucket_onehot,
    }
