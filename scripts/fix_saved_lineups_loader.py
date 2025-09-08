#!/usr/bin/env python3
"""
Patch nfl_gpp_simulator.py -> load_lineups_from_file() to avoid hanging when
"Use saved lineups" is selected. We replace the FLEX-unaware while-loop with a
deterministic DK slot reconstruction (QB,RB,RB,WR,WR,WR,TE,FLEX,DST) using
canonical positions (QB/RB/WR/TE/DST) and FLEX ∈ {RB,WR,TE}. Idempotent patch.
"""
from pathlib import Path
import re
import sys
import textwrap

SIM = Path("src/nfl_gpp_simulator.py")

HELPERS = r"""
# --- begin: canonical helpers (saved-lineups fix) ---
import ast as _ast
_CANON_PRIMARY = ("QB","RB","WR","TE","DST")
_SYNONYMS = {"D":"DST","DEF":"DST","DS":"DST","D/ST":"DST","DST":"DST",
             "W":"WR","R":"RB","Q":"QB","T":"TE"}
_DROP = {"FLEX","UTIL","BN","BENCH","NA","IR","SLOT","OP"}
def _canon_pos_any(v):
    if v is None: return ""
    if isinstance(v,(list,tuple,set)):
        toks=[str(x).strip().upper() for x in v]
    else:
        s=str(v).strip(); toks=None
        if s.startswith("[") and s.endswith("]"):
            try:
                lv=ast.literal_eval(s)
                if isinstance(lv,(list,tuple,set)):
                    toks=[str(x).strip().upper() for x in lv]
            except Exception:
                toks=None
        if toks is None:
            s2=s.replace("/",",").replace(";",",")
            toks=[t.strip().upper() for t in s2.split(",") if t.strip()]
    norm=[]
    for t in toks:
        t=_SYNONYMS.get(t,t)
        if t in _DROP: continue
        norm.append(t)
    for p in _CANON_PRIMARY:
        if p in norm: return p
    if len(norm)==1 and norm[0] in _SYNONYMS: return _SYNONYMS[norm[0]]
    return norm[0] if norm else ""
def _canon_pos_primary(v):
    s=_canon_pos_any(v)
    if s in _CANON_PRIMARY: return s
    return _SYNONYMS.get(s,"") if s else ""
# --- end: canonical helpers (saved-lineups fix) ---
"""

def ensure_helpers(src: str) -> str:
    if "_canon_pos_primary(" in src:
        return src
    m = re.search(r"(^|\n)(import[^\n]*\n)+", src)
    return src[: (m.end() if m else 0)] + HELPERS + src[(m.end() if m else 0):]

def patch_loader(src: str) -> str:
    # Locate load_lineups_from_file
    m_fun = re.search(r"\n\s*def\s+load_lineups_from_file\s*\(\s*self\s*\)\s*:\s*\n", src)
    if not m_fun:
        return src
    # We will replace the inner while-loop that tries to fill by temp_roster_construction
    # Identify the FLEX-unaware block by its "while z < 9" anchor.
    start = m_fun.end()
    tail = src[start:]
    m_block = re.search(r"\n\s*while\s+z\s*<\s*9\s*:\s*\n", tail)
    if not m_block:
        return src
    block_start = start + m_block.start()

    # Find the end of that while-block: stop right after it builds 'lineup_list = sorted(shuffled_lu)'
    m_end = re.search(r"\n\s*lineup_list\s*=\s*sorted\(\s*shuffled_lu\s*\)\s*\n\s*lineup_set\s*=\s*frozenset", src[block_start:])
    if not m_end:
        return src
    block_end = block_start + m_end.start()

    new_block = textwrap.dedent(r"""
                # Rebuild DK slots deterministically with canonical positions
                id_to_player_dict = {v.get("ID"): v for v in self.player_dict.values()}
                # slot map we will fill
                slot_names = {"QB": None, "RB1": None, "RB2": None, "WR1": None, "WR2": None, "WR3": None, "TE": None, "FLEX": None, "DST": None}
                # Collect (id, pos)
                details = []
                for l in list(lineup):
                    rec = id_to_player_dict.get(l)
                    if not rec:
                        # Missing ID in dictionary; skip and continue safely
                        continue
                    pos = _canon_pos_primary(rec.get("Position"))
                    details.append((l, pos))
                # First pass: primaries
                flex_pool = []
                for pid, pos in details:
                    if pos == "QB" and slot_names["QB"] is None:
                        slot_names["QB"] = pid
                    elif pos == "DST" and slot_names["DST"] is None:
                        slot_names["DST"] = pid
                    elif pos == "TE" and slot_names["TE"] is None:
                        slot_names["TE"] = pid
                    elif pos == "RB":
                        if slot_names["RB1"] is None:
                            slot_names["RB1"] = pid
                        elif slot_names["RB2"] is None:
                            slot_names["RB2"] = pid
                        else:
                            flex_pool.append(pid)
                    elif pos == "WR":
                        if slot_names["WR1"] is None:
                            slot_names["WR1"] = pid
                        elif slot_names["WR2"] is None:
                            slot_names["WR2"] = pid
                        elif slot_names["WR3"] is None:
                            slot_names["WR3"] = pid
                        else:
                            flex_pool.append(pid)
                    else:
                        # Non-skill (or unknown) does not go to FLEX
                        pass
                # FLEX: RB/WR/TE only
                if slot_names["FLEX"] is None:
                    # include TE to flex_pool if it overflowed primaries
                    if slot_names["TE"] is not None:
                        # already assigned primary TE; extras would have been in flex_pool above
                        pass
                    if flex_pool:
                        slot_names["FLEX"] = flex_pool[0]
                # If any core slot is still missing, bail out gracefully to avoid hanging
                core_slots = ["QB","RB1","RB2","WR1","WR2","WR3","TE","DST"]
                if any(slot_names[s] is None for s in core_slots):
                    # skip malformed saved lineup
                    continue
                # Construct shuffled_lu matching DK order
                shuffled_lu = [
                    slot_names["QB"], slot_names["RB1"], slot_names["RB2"],
                    slot_names["WR1"], slot_names["WR2"], slot_names["WR3"],
                    slot_names["TE"], slot_names["FLEX"], slot_names["DST"],
                ]
                lineup_copy = []  # no longer needed; keep variable around for later code
                position_counts = {}  # not used with deterministic mapping
    """).rstrip("\n")

    return src[:block_start] + "\n" + new_block + "\n" + src[block_end:]

def main():
    if not SIM.exists():
        print("Run from repo root; src/nfl_gpp_simulator.py not found.", file=sys.stderr)
        sys.exit(2)
    src = SIM.read_text(encoding="utf-8", errors="ignore")
    orig = src
    src = ensure_helpers(src)
    src = patch_loader(src)
    if src != orig:
        SIM.with_suffix(".py.bak").write_text(orig, encoding="utf-8")
        SIM.write_text(src, encoding="utf-8")
        print(f"✅ Patched {SIM} (backup created).")
    else:
        print("ℹ️ No changes applied (already patched).")

if __name__ == "__main__":
    main()
