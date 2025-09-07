#!/usr/bin/env python3
"""
Fix simulator lineup validity & export mapping:

1) Make `is_valid_for_position()` treat single-string positions correctly,
   accept list-strings (e.g. "['WR','FLEX']"), and canonicalize to QB/RB/WR/TE/DST.

2) In `NFL_GPP_Simulator.output()` (DK branch), rebuild the nine slots from
   canonical positions (QB,RB1,RB2,WR1,WR2,WR3,TE,FLEX,DST) instead of relying
   on whatever order `x['Lineup']` happens to be in.

Idempotent; creates a .bak once.
"""
from pathlib import Path
import re
import sys
import textwrap

SIM = Path("src/nfl_gpp_simulator.py")

HELPERS = r"""
# --- begin: canonical position helpers (sim fix) ---
import ast as _ast

_CANON_PRIMARY = ("QB","RB","WR","TE","DST")
_SYNONYMS = {"D":"DST","DEF":"DST","DS":"DST","D/ST":"DST","DST":"DST",
             "W":"WR","R":"RB","Q":"QB","T":"TE"}
_DROP = {"FLEX","UTIL","BN","BENCH","NA","IR","SLOT","OP"}

def _canon_pos_any(v):
    if v is None:
        return ""
    # list/tuple
    if isinstance(v, (list, tuple, set)):
        toks = [str(x).strip().upper() for x in v]
    else:
        s = str(v).strip()
        toks = None
        if s.startswith("[") and s.endswith("]"):
            try:
                lv = _ast.literal_eval(s)
                if isinstance(lv, (list, tuple, set)):
                    toks = [str(x).strip().upper() for x in lv]
            except Exception:
                toks = None
        if toks is None:
            s2 = s.replace("/", ",").replace(";", ",")
            toks = [t.strip().upper() for t in s2.split(",") if t.strip()]

    norm = []
    for t in toks:
        t = _SYNONYMS.get(t, t)
        if t in _DROP:
            continue
        norm.append(t)

    for p in _CANON_PRIMARY:
        if p in norm:
            return p
    if len(norm) == 1 and norm[0] in _SYNONYMS:
        return _SYNONYMS[norm[0]]
    return norm[0] if norm else ""

def _canon_pos_primary(v):
    s = _canon_pos_any(v)
    if s in _CANON_PRIMARY:
        return s
    return _SYNONYMS.get(s, "") if s else ""
# --- end: canonical position helpers (sim fix) ---
"""

def ensure_helpers(src: str) -> str:
    if "_canon_pos_primary(" in src:
        return src
    # insert after imports
    m = re.search(r"(^|\n)(import[^\n]*\n)+", src)
    insert_at = m.end() if m else 0
    return src[:insert_at] + HELPERS + src[insert_at:]

def patch_is_valid(src: str) -> str:
    m = re.search(r"\n\s*def\s+is_valid_for_position\s*\(self,\s*player,\s*position_idx\)\s*:\s*\n", src)
    if not m:
        return src
    start = m.start()
    tail = src[m.end():]
    m_next = re.search(r"\n\s*def\s+\w+\s*\(", tail)
    end = m.end() + (m_next.start() if m_next else 0)
    new_body = textwrap.dedent("""
        def is_valid_for_position(self, player, position_idx):
            # Accept both single-string and iterable positions; canonicalize
            raw = self.get_player_attribute(player, "Position")
            try:
                if isinstance(raw, (list, tuple, set)):
                    pos_list = list(raw)
                else:
                    pos_list = [raw]
            except Exception:
                pos_list = [raw]
            pos_list = [(_canon_pos_primary(p) if p is not None else "") for p in pos_list]
            pos_list = [p for p in pos_list if p]
            return any(p in self.position_map[position_idx] for p in pos_list)
    """).strip("\n")
    return src[:start] + "\n" + new_body + "\n" + src[end:]

def patch_output_slots(src: str) -> str:
    # Replace the DK "player_parts = [...] ; player_parts.append(...)" block
    anchor = "                # Build player name/id pairs"
    start = src.find(anchor)
    if start == -1:
        return src
    end = src.find("                if self.use_contest_data:", start)
    if end == -1:
        return src
    new_block = textwrap.dedent("""
                # Build player name/id pairs (deterministic slot mapping)
                # Reconstruct DK slots from canonical positions
                slot_names = {"QB": None, "RB1": None, "RB2": None, "WR1": None, "WR2": None, "WR3": None, "TE": None, "FLEX": None, "DST": None}
                details = []
                for i, pid in enumerate(x["Lineup"]):
                    name = lu_names[i]
                    pos_raw = None
                    for k, v in self.player_dict.items():
                        if v.get("ID") == pid:
                            pos_raw = v.get("Position")
                            break
                    pos = _canon_pos_primary(pos_raw)
                    details.append((pid, name, pos))
                flex_pool = []
                for pid, name, pos in details:
                    label = f"{name.replace('#','-')} ({pid})"
                    if pos == "QB" and slot_names["QB"] is None:
                        slot_names["QB"] = label
                    elif pos == "DST" and slot_names["DST"] is None:
                        slot_names["DST"] = label
                    elif pos == "TE" and slot_names["TE"] is None:
                        slot_names["TE"] = label
                    elif pos == "RB":
                        if slot_names["RB1"] is None:
                            slot_names["RB1"] = label
                        elif slot_names["RB2"] is None:
                            slot_names["RB2"] = label
                        else:
                            flex_pool.append(label)
                    elif pos == "WR":
                        if slot_names["WR1"] is None:
                            slot_names["WR1"] = label
                        elif slot_names["WR2"] is None:
                            slot_names["WR2"] = label
                        elif slot_names["WR3"] is None:
                            slot_names["WR3"] = label
                        else:
                            flex_pool.append(label)
                    else:
                        flex_pool.append(label)
                if slot_names["FLEX"] is None and flex_pool:
                    slot_names["FLEX"] = flex_pool[0]
                player_parts = [
                    slot_names.get("QB",""),
                    slot_names.get("RB1",""),
                    slot_names.get("RB2",""),
                    slot_names.get("WR1",""),
                    slot_names.get("WR2",""),
                    slot_names.get("WR3",""),
                    slot_names.get("TE",""),
                    slot_names.get("FLEX",""),
                    slot_names.get("DST",""),
                ]
    """)
    return src[:start] + new_block + src[end:]

def main():
    if not SIM.exists():
        print("Could not find src/nfl_gpp_simulator.py. Run from repo root.", file=sys.stderr)
        sys.exit(2)

    src = SIM.read_text(encoding="utf-8", errors="ignore")
    orig = src

    src = ensure_helpers(src)
    src = patch_is_valid(src)
    src = patch_output_slots(src)

    if src != orig:
        SIM.with_suffix(".py.bak").write_text(orig, encoding="utf-8")
        SIM.write_text(src, encoding="utf-8")
        print(f"✅ Patched {SIM} (backup at {SIM.with_suffix('.py.bak').name})")
    else:
        print("ℹ️ No changes applied (already patched).")

if __name__ == "__main__":
    main()
