#!/usr/bin/env python3
"""
Patch nfl_gpp_simulator.py to:
  1) Canonicalize positions (handles 'W' -> 'WR', D/DEF/D/ST -> DST, list-strings).
  2) Harden get_corr_value() to use canonical positions and default to 0.0 on unknowns.
  3) Add a gentle fallback in field-lineup generation to reduce "find failed ..." loops.

Idempotent; creates a .bak once.
"""
from pathlib import Path
import re
import sys

SIM = Path("src/nfl_gpp_simulator.py")

HELPERS = r"""
# --- begin: canonical position helpers (sim) ---
import ast as _ast

_CANON_PRIMARY = ("QB","RB","WR","TE","DST")
_SYNONYMS = {"D":"DST","DEF":"DST","DS":"DST","D/ST":"DST","DST":"DST","W":"WR","R":"RB","Q":"QB","T":"TE"}
_DROP = {"FLEX","UTIL","BN","BENCH","NA","IR","SLOT","OP"}

def _canon_pos_any(v):
    \"\"\"
    Accepts:
      - 'WR' / 'DST'
      - 'W'/'R'/'Q'/'T'/'D' single-letter
      - "['WR','FLEX']" (stringified list)
      - 'WR/FLEX' or 'WR, FLEX'
      - ['WR','FLEX'] (actual list)
    Returns canonical single: QB/RB/WR/TE/DST or ''.
    \"\"\"
    if v is None:
        return ""
    # list-like?
    tok_list = None
    if isinstance(v, (list, tuple)):
        tok_list = [str(x).strip().upper() for x in v]
    else:
        s = str(v).strip()
        if (s.startswith("[") and s.endswith("]")):
            try:
                lv = _ast.literal_eval(s)
                if isinstance(lv, (list, tuple)):
                    tok_list = [str(x).strip().upper() for x in lv]
            except Exception:
                tok_list = None
        if tok_list is None:
            s2 = s.replace("/", ",").replace(";", ",")
            tok_list = [t.strip().upper() for t in s2.split(",") if t.strip()]

    norm = []
    for t in tok_list:
        t = _SYNONYMS.get(t, t)
        if t in _DROP:
            continue
        norm.append(t)

    for p in _CANON_PRIMARY:
        if p in norm:
            return p
    # single-letter alias -> map again
    if len(norm) == 1 and norm[0] in _SYNONYMS:
        return _SYNONYMS[norm[0]]
    return norm[0] if norm else ""

def _canon_pos_primary(v):
    s = _canon_pos_any(v)
    if s in _CANON_PRIMARY:
        return s
    return _SYNONYMS.get(s, "") if s else ""
# --- end: canonical position helpers (sim) ---
"""

GET_CORR_PATCH = r"""
    # --- begin: robust correlation getter ---
    def get_corr_value(self, p_i, p_j, position_correlations=None):
        \"\"\"
        Return correlation between two players based on their primary positions.
        Works with either a flat dict (pos->val) or nested dict (pos->pos->val).
        Defaults to 0.0 if missing.
        \"\"\"
        try:
            # p_i / p_j may be dict-like rows or Player objects
            pos_i = getattr(p_i, "pos", None) or getattr(p_i, "Position", None) or (p_i.get("Position") if isinstance(p_i, dict) else None)
            pos_j = getattr(p_j, "pos", None) or getattr(p_j, "Position", None) or (p_j.get("Position") if isinstance(p_j, dict) else None)
        except Exception:
            pos_i = pos_j = None
        pi = _canon_pos_primary(pos_i)
        pj = _canon_pos_primary(pos_j)
        if not pi or not pj:
            return 0.0

        if position_correlations is None:
            try:
                position_correlations = self.position_correlations
            except Exception:
                position_correlations = {}

        try:
            # nested dict first: corr[pi][pj]
            val = position_correlations[pi][pj]
        except Exception:
            try:
                # flat dict fallback: corr[pi]
                val = position_correlations[pi]
            except Exception:
                val = 0.0
        try:
            return float(val)
        except Exception:
            return 0.0
    # --- end: robust correlation getter ---
"""

FALLBACK_INSERT = r"""
        # --- begin: lineup generation fallback tuning ---
        # If we repeatedly fail to select the first/non-stack player, relax the strictness a bit.
        # This prevents 5000-iteration timeouts on small or multi-eligible pools.
        if "fail_counter" not in locals():
            fail_counter = 0
        fail_counter += 1
        if fail_counter in (500, 1000, 2000, 3000, 4000):
            try:
                # widen pool: allow any skill position as seed (WR/RB/TE), keep QB/DST out
                widened = [p for p in players if _canon_pos_primary(getattr(p, "pos", None) or getattr(p, "Position", None)) in ("WR","RB","TE")]
                if widened:
                    first_player_pool = widened
                # as a last resort after 3000 attempts, allow stack leniency by one
                if fail_counter >= 3000 and hasattr(self, "stack_rules"):
                    self.stack_rules = max(0, int(self.stack_rules) - 1)
            except Exception:
                pass
        # --- end: lineup generation fallback tuning ---
"""

def patch_helpers(src: str) -> str:
    if "_canon_pos_primary(" in src:
        return src
    m = re.search(r"(^|\n)(import[^\n]*\n)+", src)
    insert_at = m.end() if m else 0
    return src[:insert_at] + HELPERS + src[insert_at:]

def patch_get_corr(src: str) -> str:
    # Replace existing get_corr_value definition or insert a new one on the class
    if "def get_corr_value(" in src and "robust correlation getter" in src:
        return src  # already patched
    if re.search(r"\ndef\s+get_corr_value\s*\([^) ]*\)\s*:\s*.*?\n\s*#\s*end\s*:?get_corr_value.*?\n", src, flags=re.DOTALL):
        # replace body of the function
        src = re.sub(
            r"\ndef\s+get_corr_value\s*\([^)]*\)\s*:\s*.*?\n\s*#\s*end\s*:?get_corr_value.*?\n",
            "\n" + GET_CORR_PATCH + "\n",
            src,
            flags=re.DOTALL
        )
        # If the end marker wasn't there, do a simpler replace up to next def
        if "robust correlation getter" not in src:
            src = re.sub(
                r"\ndef\s+get_corr_value\s*\([^)]*\)\s*:\s*.*?(?=\n\s*def\s+\w+\s*\(|\Z)",
                "\n" + GET_CORR_PATCH + "\n",
                src,
                flags=re.DOTALL
            )
        return src
    # Insert near other helper methods (after class def)
    m = re.search(r"\nclass\s+NFL_GPP_Simulator\([^)]*\):\s*\n", src)
    if m:
        insert_at = m.end()
        return src[:insert_at] + GET_CORR_PATCH + src[insert_at:]
    return src

def patch_fallback(src: str) -> str:
    # Heuristic: insert fallback in the loop that logs "find failed on nonstack and first player selection"
    if "lineup generation fallback tuning" in src:
        return src
    m = re.search(r"find failed on nonstack and first player selection", src)
    if not m:
        return src
    # insert fallback right after the logging line
    line_start = src.rfind("\n", 0, m.start()) + 1
    line_end = src.find("\n", m.end())
    insert_at = line_end if line_end != -1 else m.end()
    return src[:insert_at] + FALLBACK_INSERT + src[insert_at:]

def main():
    if not SIM.exists():
        print("Could not find src/nfl_gpp_simulator.py (run from repo root)", file=sys.stderr)
        sys.exit(2)
    src = SIM.read_text(encoding="utf-8")
    orig = src

    src = patch_helpers(src)
    src = patch_get_corr(src)
    src = patch_fallback(src)

    if src != orig:
        bak = SIM.with_suffix(".py.bak")
        bak.write_text(orig, encoding="utf-8")
        SIM.write_text(src, encoding="utf-8")
        print(f"✅ Patched {SIM} (backup at {bak})")
    else:
        print("ℹ️ No changes applied (already patched or patterns not found).")

if __name__ == "__main__":
    main()
