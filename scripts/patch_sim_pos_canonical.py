#!/usr/bin/env python3
# scripts/patch_sim_pos_canonical.py
#
# Normalize Position in nfl_gpp_simulator.py:
#  - parse list-like strings (e.g., "['WR','FLEX']") and lists/tuples
#  - map D/DEF/DS/D/ST -> DST
#  - remove FLEX/UTIL and keep primary among QB/RB/WR/TE/DST
#  - apply to players_df and player_dict before any guards
#
# Idempotent: won't duplicate blocks; creates a .bak backup.

from pathlib import Path
import re, sys

SIM = Path("src/nfl_gpp_simulator.py")

HELPERS = r"""
# --- begin: position canonicalization helpers ---
import ast as _ast

_CANON_PRIMARY = ("QB","RB","WR","TE","DST")
_SYNONYMS = {"D":"DST","DEF":"DST","DS":"DST","D/ST":"DST","DST":"DST"}
_DROP = {"FLEX","UTIL","BN","BENCH","NA","IR","SLOT","OP"}

def _canon_pos_any(v):
    """
    Accepts:
      - 'WR' / 'DST'
      - "['WR','FLEX']" (stringified list)
      - 'WR/FLEX' or 'WR, FLEX'
      - ['WR','FLEX'] (actual list)
    Returns canonical single: QB/RB/WR/TE/DST or ''.
    """
    if v is None:
        return ""
    # Already a clean string?
    if isinstance(v, str):
        s = v.strip()
        # stringified list?
        if (s.startswith("[") and s.endswith("]")) or ("," in s) or ("/" in s):
            try:
                # Try literal_eval first
                lv = _ast.literal_eval(s)
                if not isinstance(lv, (list, tuple)):
                    # fall through to splitting
                    raise ValueError
                toks = [str(x).strip().upper() for x in lv]
            except Exception:
                s2 = s.replace("[","").replace("]","").replace("'","").replace("\"","")
                s2 = s2.replace("/", ",")
                toks = [t.strip().upper() for t in s2.split(",") if t.strip()]
        else:
            toks = [s.upper()]
    else:
        try:
            # list/tuple-like
            toks = [str(x).strip().upper() for x in v]
        except Exception:
            toks = [str(v).strip().upper()]

    # Map synonyms and drop non-positions
    norm = []
    for t in toks:
        t = _SYNONYMS.get(t, t)
        if t in _DROP:
            continue
        norm.append(t)

    # Choose primary in priority order
    for p in _CANON_PRIMARY:
        if p in norm:
            return p
    # As a fallback, a single mapped token if any
    return norm[0] if norm else ""
# --- end: position canonicalization helpers ---
"""

# Insert helpers after imports if not present
def ensure_helpers(src: str) -> str:
    if "_canon_pos_any(" in src:
        return src
    m = re.search(r"(^|\n)(import[^\n]*\n)+", src)
    insert_at = m.end() if m else 0
    return src[:insert_at] + HELPERS + src[insert_at:]

# Normalize players_df right after load_projections()
PROJ_NORM = r"""
        # --- Begin: canonicalize positions in players_df ---
        try:
            if hasattr(self, "players_df") and self.players_df is not None:
                for col in ("pos","Pos","position","Position"):
                    if col in self.players_df.columns:
                        self.players_df[col] = self.players_df[col].apply(_canon_pos_any)
                        # standardize to 'Position'
                        if col != "Position":
                            self.players_df["Position"] = self.players_df[col]
                # common team fixes
                for tcol in ("team","Team","TeamAbbrev","teamAbbrev"):
                    if tcol in self.players_df.columns:
                        self.players_df[tcol] = (
                            self.players_df[tcol].astype(str).str.upper().str.strip().replace({"LA":"LAR"})
                        )
                        self.players_df["TeamAbbrev"] = self.players_df[tcol]
        except Exception:
            pass
        # --- End: canonicalize positions in players_df ---
"""

def inject_proj_norm(src: str) -> str:
    if PROJ_NORM in src:
        return src
    m = re.search(r"(self\.load_projections\([^\)]*\)\s*\n)", src)
    if not m:
        return src
    return src[:m.end()] + PROJ_NORM + src[m.end():]

# Normalize player_dict entries and re-run DST backfill on canonical positions
DICT_NORM = r"""
        # --- Begin: canonicalize positions in player_dict & DST backfill ---
        try:
            # 1) Canonicalize Position / TeamAbbrev
            for _k, _rec in list(self.player_dict.items()):
                # bring Position to canonical
                raw_pos = _rec.get("Position") or _rec.get("pos")
                _rec["Position"] = _canon_pos_any(raw_pos)
                # team abbrev to upper
                if "TeamAbbrev" in _rec:
                    _rec["TeamAbbrev"] = str(_rec["TeamAbbrev"]).upper().strip()
                elif "Team" in _rec:
                    _rec["TeamAbbrev"] = str(_rec["Team"]).upper().strip()

            # 2) Backfill DST IDs strictly by canonical 'DST'
            try:
                from player_ids_flex import load_player_ids_flex
                pid_path = getattr(self, "player_ids_path", "data/player_ids.csv")
                _pid_df = load_player_ids_flex(pid_path)
            except Exception:
                _pid_df = None

            def _dst_id_by_team_lookup(team):
                team = str(team or "").upper().strip()
                if _pid_df is None or not team:
                    return None
                try:
                    row = _pid_df[
                        (_pid_df["Position"]=="DST") &
                        (_pid_df["TeamAbbrev"].astype(str).str.upper()==team)
                    ].iloc[0]
                    return int(row["ID"])
                except Exception:
                    return None

            for _k, _rec in list(self.player_dict.items()):
                if _rec.get("Position") == "DST":
                    _id_raw = _rec.get("ID", 0)
                    try:
                        _id_ok = int(str(_id_raw).replace(",",""))
                    except Exception:
                        _id_ok = 0
                    if not _id_ok:
                        pid = _dst_id_by_team_lookup(_rec.get("TeamAbbrev"))
                        if pid:
                            _rec["ID"] = pid

            # 3) Post-normalization position counts for diagnostics
            pos_counts = {}
            for _rec in self.player_dict.values():
                p = str(_rec.get("Position","")).upper()
                pos_counts[p] = pos_counts.get(p, 0) + 1
            # Guard: must have at least one canonical DST
            if pos_counts.get("DST",0) <= 0:
                raise AssertionError(f"Simulator: no DST players after position canonicalization. POS counts={pos_counts}")
        except Exception:
            raise
        # --- End: canonicalize positions in player_dict & DST backfill ---
"""

def inject_dict_norm(src: str) -> str:
    # Place this inside get_optimal(), at the start, before objective build
    m = re.search(r"(def\s+get_optimal\s*\(self[^\)]*\)\s*:\s*\n)", src)
    if not m or DICT_NORM in src:
        return src
    return src[:m.end()] + DICT_NORM + src[m.end():]

def main():
    if not SIM.exists():
        print("Could not find src/nfl_gpp_simulator.py. Run from repo root.", file=sys.stderr)
        sys.exit(2)

    src = SIM.read_text(encoding="utf-8")
    orig = src

    src = ensure_helpers(src)
    src = inject_proj_norm(src)
    src = inject_dict_norm(src)

    if src != orig:
        bak = SIM.with_suffix(".py.bak")
        bak.write_text(orig, encoding="utf-8")
        SIM.write_text(src, encoding="utf-8")
        print(f"✅ Patched {SIM} (backup at {bak})")
    else:
        print("ℹ️ No changes applied (already patched).")

if __name__ == "__main__":
    main()

