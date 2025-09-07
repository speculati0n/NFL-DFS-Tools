#!/usr/bin/env python3
"""
Patch nfl_gpp_simulator.py to:
  1) Build pos_matrix from canonical positions and map FLEX -> RB/WR/TE.
  2) Add a _scalar() helper and use it wherever salary/proj add from numpy picks.
  3) In output(), only allow RB/WR/TE to fill FLEX slot.
Idempotent; makes a .bak once.
"""
from pathlib import Path
import re
import sys
import textwrap

SIM = Path("src/nfl_gpp_simulator.py")

HELPERS = r"""
# --- begin: canonical pos + scalar helpers (sim) ---
import ast as _ast
_CANON_PRIMARY = ("QB","RB","WR","TE","DST")
_SYNONYMS = {"D":"DST","DEF":"DST","DS":"DST","D/ST":"DST","DST":"DST",
             "W":"WR","R":"RB","Q":"QB","T":"TE"}
_DROP = {"FLEX","UTIL","BN","BENCH","NA","IR","SLOT","OP"}

def _canon_pos_any(v):
    if v is None: return ""
    if isinstance(v, (list,tuple,set)):
        toks = [str(x).strip().upper() for x in v]
    else:
        s = str(v).strip()
        toks = None
        if s.startswith("[") and s.endswith("]"):
            try:
                lv = _ast.literal_eval(s)
                if isinstance(lv, (list,tuple,set)):
                    toks = [str(x).strip().upper() for x in lv]
            except Exception:
                toks = None
        if toks is None:
            s2 = s.replace("/", ",").replace(";", ",")
            toks = [t.strip().upper() for t in s2.split(",") if t.strip()]
    norm = []
    for t in toks:
        t = _SYNONYMS.get(t, t)
        if t in _DROP: continue
        norm.append(t)
    for p in _CANON_PRIMARY:
        if p in norm: return p
    if len(norm)==1 and norm[0] in _SYNONYMS: return _SYNONYMS[norm[0]]
    return norm[0] if norm else ""

def _canon_pos_primary(v):
    s = _canon_pos_any(v)
    if s in _CANON_PRIMARY: return s
    return _SYNONYMS.get(s, "") if s else ""

def _scalar(x):
    try:
        return float(x.item())
    except Exception:
        try:
            return float(x[0])
        except Exception:
            return float(x)
# --- end: canonical pos + scalar helpers (sim) ---
"""

def ensure_helpers(src: str) -> str:
    if "_scalar(" in src and "_canon_pos_primary(" in src:
        return src
    m = re.search(r"(^|\n)(import[^\n]*\n)+", src)
    return src[: (m.end() if m else 0)] + HELPERS + src[(m.end() if m else 0):]

def patch_pos_matrix(src: str) -> str:
    # Replace the loop that builds `positions.append(np.array(pos_list))`
    # around lines building ids/ownership/salaries/teams/opponents/matchups.
    pattern = re.compile(
        r"""
        pos_list\s*=\s*\[\]\s*\n
        \s*for\s+pos\s+in\s+temp_roster_construction:\s*\n
        \s*if\s+pos\s+in\s+self\.player_dict\[k\]\["Position"\]:\s*\n
        \s*pos_list\.append\(1\)\s*\n
        \s*else:\s*\n
        \s*pos_list\.append\(0\)\s*\n
        \s*positions\.append\(\s*np\.array\(pos_list\)\s*\)\s*
        """,
        re.VERBOSE,
    )
    if not pattern.search(src):
        return src
    replacement = textwrap.dedent(
        """
        pos_list = []
        p = _canon_pos_primary(self.player_dict[k].get("Position"))
        for slot in temp_roster_construction:
            if slot == "FLEX":
                pos_list.append(1 if p in ("RB","WR","TE") else 0)
            else:
                pos_list.append(1 if p == slot else 0)
        positions.append(np.array(pos_list, dtype=int))
        """
    ).strip("\n")
    return pattern.sub(replacement, src, count=1)

def patch_scalar_adds(src: str) -> str:
    # Replace all `salary += salaries[choice_idx]` and `proj += projections[choice_idx]`
    src = re.sub(r"salary\s*\+=\s*salaries\[choice_idx\]", "salary += _scalar(salaries[choice_idx])", src)
    src = re.sub(r"proj\s*\+=\s*projections\[choice_idx\]", "proj += _scalar(projections[choice_idx])", src)
    # a few places set `salary = 0` then add again; those are fine.
    return src

def patch_output_flex(src: str) -> str:
    # In the DK export block we previously inserted, restrict FLEX pool to RB/WR/TE only.
    # Replace any line that appends to flex_pool in the `else:` case to guard by pos.
    src = re.sub(
        r"\n(\s*)else:\s*\n\1\s*flex_pool\.append\(label\)",
        (
            "\n\1else:\n"
            "\1    # Only allow skill players to be considered for FLEX\n"
            "\1    if pos in (\"RB\",\"WR\",\"TE\"):\n"
            "\1        flex_pool.append(label)"
        ),
        src,
    )
    return src

def main():
    if not SIM.exists():
        print("Run from repo root; src/nfl_gpp_simulator.py not found.", file=sys.stderr)
        sys.exit(2)
    src = SIM.read_text(encoding="utf-8", errors="ignore")
    orig = src
    src = ensure_helpers(src)
    src = patch_pos_matrix(src)
    src = patch_scalar_adds(src)
    src = patch_output_flex(src)
    if src != orig:
        SIM.with_suffix(".py.bak").write_text(orig, encoding="utf-8")
        SIM.write_text(src, encoding="utf-8")
        print(f"✅ Patched {SIM} (backup created).")
    else:
        print("ℹ️ No changes applied (already patched).")

if __name__ == "__main__":
    main()
