from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional, Set, Tuple
import math

# DraftKings NFL defaults
DEFAULT_SALARY_CAP = 50000
DEFAULT_MIN_SPEND_PCT = 0.99  # 99% of cap => $49,500 by default

# Valid positions for FLEX on DK NFL
FLEX_POS = {"RB", "WR", "TE"}

@dataclass(frozen=True)
class Player:
    id: str
    name: str
    pos: str
    team: Optional[str]
    opp: Optional[str]
    salary: int
    proj: float

@dataclass
class Lineup:
    # positions must match your app’s column order
    QB: Optional[Player] = None
    RB1: Optional[Player] = None
    RB2: Optional[Player] = None
    WR1: Optional[Player] = None
    WR2: Optional[Player] = None
    WR3: Optional[Player] = None
    TE: Optional[Player] = None
    FLEX: Optional[Player] = None
    DST: Optional[Player] = None

    def slots(self) -> List[str]:
        return ["QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST"]

    def players(self) -> List[Player]:
        return [getattr(self, s) for s in self.slots() if getattr(self, s) is not None]

    def salary(self) -> int:
        return sum(p.salary for p in self.players())

    def projection(self) -> float:
        return sum(float(p.proj) for p in self.players())

def sanitize_salary(x) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0

def lineup_valid_positions(L: Lineup) -> bool:
    # Ensure positions match slot rules (FLEX must be RB/WR/TE; DST must be DST)
    for slot in ["QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST"]:
        p = getattr(L, slot)
        if p is None:
            return False
        if slot == "QB" and p.pos != "QB": return False
        if slot in {"RB1","RB2"} and p.pos != "RB": return False
        if slot in {"WR1","WR2","WR3"} and p.pos != "WR": return False
        if slot == "TE" and p.pos != "TE": return False
        if slot == "FLEX" and p.pos not in FLEX_POS: return False
        if slot == "DST" and p.pos != "DST": return False
    # no duplicate players
    ids = [p.id for p in L.players()]
    return len(ids) == len(set(ids))

def lineup_meets_salary(L: Lineup, cap: int, min_pct: float) -> bool:
    s = L.salary()
    return (s <= cap) and (s >= math.floor(cap * min_pct))

def validate_lineup(L: Lineup, cap: int = DEFAULT_SALARY_CAP, min_pct: float = DEFAULT_MIN_SPEND_PCT) -> bool:
    return lineup_valid_positions(L) and lineup_meets_salary(L, cap, min_pct)

def cheapest_feasible_cost(remaining_slots: List[str], pool_by_pos: Dict[str, List[Player]], used_ids: Set[str]) -> int:
    """Lower bound cost needed to fill remaining slots (greedy min)."""
    cost = 0
    for slot in remaining_slots:
        if slot == "FLEX":
            # pick the minimum among RB/WR/TE not used
            cands = [p for pos in FLEX_POS for p in pool_by_pos.get(pos, []) if p.id not in used_ids]
        else:
            want = "DST" if slot == "DST" else slot.rstrip("123")  # RB1->RB, WR3->WR
            cands = [p for p in pool_by_pos.get(want, []) if p.id not in used_ids]
        if not cands:
            return 10**9  # impossible
        cost += min(p.salary for p in cands)
    return cost

def action_mask_for_slot(
    slot: str,
    current: Lineup,
    pool_by_pos: Dict[str, List[Player]],
    used_ids: Set[str],
    cap: int = DEFAULT_SALARY_CAP,
    min_pct: float = DEFAULT_MIN_SPEND_PCT,
) -> Dict[str, bool]:
    """
    Return a dict {player_id: allowed?} for the given slot, forbidding choices that
    (a) violate position rules, (b) duplicate a player, or
    (c) make it impossible to end within [min_pct * cap, cap].
    """
    already = current.salary()
    remaining_slots = [s for s in current.slots() if getattr(current, s) is None and s != slot]
    # candidates for this slot
    if slot == "FLEX":
        cand_list = [p for pos in FLEX_POS for p in pool_by_pos.get(pos, [])]
    else:
        want = "DST" if slot == "DST" else slot.rstrip("123")
        cand_list = pool_by_pos.get(want, [])
    out = {}
    for p in cand_list:
        if p.id in used_ids:
            out[p.id] = False
            continue
        # optimistic feasibility check:
        # 1) salary cannot exceed cap immediately
        if already + p.salary > cap:
            out[p.id] = False
            continue
        # 2) with the cheapest remaining, can we still reach min spend and stay under cap?
        new_used = set(used_ids)
        new_used.add(p.id)
        lower_bound = cheapest_feasible_cost(remaining_slots, pool_by_pos, new_used)
        min_needed = max(0, math.floor(cap * min_pct) - (already + p.salary))
        max_room   = cap - (already + p.salary)
        out[p.id] = (lower_bound <= max_room) and (min_needed <= max_room)
    return out

def repair_to_min_salary(
    L: Lineup,
    pool_by_pos: Dict[str, List[Player]],
    cap: int = DEFAULT_SALARY_CAP,
    min_pct: float = DEFAULT_MIN_SPEND_PCT,
    max_iters: int = 200,
) -> Lineup:
    """
    Greedy upgrade: while salary < min, try replacing a slot with a higher-salary,
    higher-projection player that keeps lineup <= cap. Stops when stuck or satisfied.
    """
    def upgrade_candidates(slot: str, current_ids: Set[str]) -> List[Player]:
        if slot == "FLEX":
            cands = [p for pos in FLEX_POS for p in pool_by_pos.get(pos, [])]
        else:
            want = "DST" if slot == "DST" else slot.rstrip("123")
            cands = pool_by_pos.get(want, [])
        return [p for p in cands if p.id not in current_ids]

    it = 0
    while it < max_iters and L.salary() < math.floor(cap * min_pct):
        it += 1
        used = {p.id for p in L.players()}
        best_delta = 0.0
        best_slot: Optional[str] = None
        best_player: Optional[Player] = None
        for slot in L.slots():
            cur = getattr(L, slot)
            for cand in upgrade_candidates(slot, used - ({cur.id} if cur else set())):
                if cand.salary <= (cur.salary if cur else 0):  # must be an upgrade in spend
                    continue
                new_sal = L.salary() - (cur.salary if cur else 0) + cand.salary
                if new_sal > cap:
                    continue
                delta_proj = cand.proj - (cur.proj if cur else 0.0)
                if delta_proj > best_delta:
                    best_delta, best_slot, best_player = delta_proj, slot, cand
        if best_player is None:
            break
        setattr(L, best_slot, best_player)
    return L
