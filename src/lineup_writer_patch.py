#!/usr/bin/env python3
# src/lineup_writer_patch.py
# Robust CSV writer for lineups with enforced DST presence and unique headers.
from dataclasses import dataclass
from typing import List, Optional, Callable, Iterable
import csv, os, re

@dataclass
class Player:
    name: str
    pos: str
    team: str
    salary: float
    proj: float
    act: float = 0.0
    ceil: float = 0.0
    own: float = 0.0
    stddev: float = 0.0

def _norm_pos(p: str) -> str:
    p = (p or "").upper().strip()
    p = re.sub(r"[\[\]\"']", "", p)
    if p in ("D", "DEF", "DS", "D/ST", "DST"):
        return "DST"
    return p

def _slot_str(p: Optional[Player]) -> str:
    return "" if p is None else f"{p.name}"

def _extract_slots(players: Iterable[Player]):
    # Normalize positions up front
    normed = []
    for p in players:
        p.pos = _norm_pos(p.pos)
        normed.append(p)

    by_pos = {"QB": [], "RB": [], "WR": [], "TE": [], "DST": []}
    for p in normed:
        pos = _norm_pos(p.pos)
        if pos in by_pos:
            by_pos[pos].append(p)

    qb  = by_pos["QB"][0]  if by_pos["QB"]  else None
    dst = by_pos["DST"][0] if by_pos["DST"] else None

    # deterministic order for slotting
    prio = {"QB":0,"RB":1,"WR":2,"TE":3}
    skill = [p for p in normed if _norm_pos(p.pos) != "DST"]
    skill.sort(key=lambda x: (prio.get(_norm_pos(x.pos), 9), -float(x.salary or 0.0), x.name or ""))

    rb = [p for p in skill if _norm_pos(p.pos) == "RB"][:2]
    wr = [p for p in skill if _norm_pos(p.pos) == "WR"][:3]
    te = [p for p in skill if _norm_pos(p.pos) == "TE"][:1]

    used_ids = {id(x) for x in ([qb, dst] + rb + wr + te) if x is not None}
    flex = next((p for p in skill if id(p) not in used_ids), None)

    return {
        "QB": qb, "RB1": rb[0] if len(rb)>0 else None, "RB2": rb[1] if len(rb)>1 else None,
        "WR1": wr[0] if len(wr)>0 else None, "WR2": wr[1] if len(wr)>1 else None, "WR3": wr[2] if len(wr)>2 else None,
        "TE": te[0] if len(te)>0 else None, "FLEX": flex, "DST": dst
    }

def _own_sum(players: Iterable[Player]) -> float:
    return sum(float(getattr(p, "own", 0.0) or 0.0) for p in players)

def _own_prod(players: Iterable[Player]) -> float:
    prod = 1.0
    for p in players:
        prod *= max(1e-6, (float(getattr(p, "own", 0.0) or 0.0) / 100.0))
    return prod

def _sum(players: Iterable[Player], attr: str) -> float:
    return sum(float(getattr(p, attr) or 0.0) for p in players)

HEADER = [
    "QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST",
    "Salary","Fpts Proj","Fpts Used","Fpts Act","Ceiling",
    "Own. Sum","Own. Product","STDDEV","Players vs DST","Stack"
]

def write_lineup_csv(
    lineups: List[List[Player]],
    out_path: str,
    stddev_fn: Optional[Callable[[List[Player]], float]] = None,
    players_vs_dst_fn: Optional[Callable[[List[Player]], int]] = None,
    stack_str_fn: Optional[Callable[[List[Player]], str]] = None,
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i, lineup in enumerate(lineups, start=1):
            slots = _extract_slots(lineup)
            nine = [slots["QB"], slots["RB1"], slots["RB2"], slots["WR1"], slots["WR2"], slots["WR3"], slots["TE"], slots["FLEX"], slots["DST"]]
            if not all(p is not None for p in nine):
                missing = [k for k, v in slots.items() if v is None]
                raise AssertionError(f"[lineup {i}] Incomplete lineup or missing DST. Missing: {missing}. Lineup POS: {[getattr(p,'pos',None) for p in lineup]}")

            total_salary = _sum(nine, "salary")
            total_proj   = _sum(nine, "proj")

            # --- Begin: hard guard on salary bounds (independent of optimizer) ---
            cap_env   = os.environ.get("OPT_CAP")
            floor_env = os.environ.get("OPT_FLOOR")
            try:
                cap   = int(cap_env)   if cap_env   is not None else 50000
                floor = int(floor_env) if floor_env is not None else 45000
            except Exception:
                cap, floor = 50000, 45000  # DK defaults if env missing

            if not (floor - 1e-6 <= total_salary <= cap + 1e-6):
                raise AssertionError(
                    f"[lineup {i}] Salary {total_salary} out of bounds "
                    f"(cap={cap}, floor={floor}). "
                    f"QB={_slot_str(slots['QB'])}, RB1={_slot_str(slots['RB1'])}, RB2={_slot_str(slots['RB2'])}, "
                    f"WR1={_slot_str(slots['WR1'])}, WR2={_slot_str(slots['WR2'])}, WR3={_slot_str(slots['WR3'])}, "
                    f"TE={_slot_str(slots['TE'])}, FLEX={_slot_str(slots['FLEX'])}, DST={_slot_str(slots['DST'])}"
                )
            # --- End: hard guard on salary bounds ---

            total_act    = _sum(nine, "act")
            total_ceil   = _sum(nine, "ceil")
            own_sum      = _own_sum(nine)
            own_prod     = _own_prod(nine)
            stddev       = float(stddev_fn(nine)) if stddev_fn else 0.0
            p_vs_dst     = int(players_vs_dst_fn(nine)) if players_vs_dst_fn else 0
            stack_str    = stack_str_fn(nine) if stack_str_fn else ""
            row = [
                _slot_str(slots["QB"]), _slot_str(slots["RB1"]), _slot_str(slots["RB2"]),
                _slot_str(slots["WR1"]), _slot_str(slots["WR2"]), _slot_str(slots["WR3"]),
                _slot_str(slots["TE"]), _slot_str(slots["FLEX"]), _slot_str(slots["DST"]),
                round(total_salary,2), round(total_proj,2), round(total_proj,2), round(total_act,2), round(total_ceil,2),
                round(own_sum,6), own_prod, round(stddev,6), p_vs_dst, stack_str
            ]
            assert len(row) == len(HEADER), f"[lineup {i}] Row has {len(row)} fields; expected {len(HEADER)}"
            w.writerow(row)

        # Optional: add a simple footer with optimal and floor if provided via environment
        opt = os.environ.get("OPTIMAL_FPTS")
        floor = os.environ.get("MIN_FPTS_FLOOR")
        if opt or floor:
            w.writerow([])
            w.writerow(["# Optimal FPTS", opt or "", "# Min FPTS Floor", floor or ""])
