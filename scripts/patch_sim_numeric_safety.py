#!/usr/bin/env python3
# scripts/patch_sim_numeric_safety.py
# Patches src/nfl_gpp_simulator.py to:
#  - add _norm_pos and _sf (safe-float) helpers
#  - normalize projections table pos/team on ingest
#  - coerce numeric fields on self.player_dict before get_optimal builds the LP
#  - backfill DST IDs by team if missing (re-uses player_ids_flex)
#  - add robust guards with diagnostics
#
# Safe to run multiple times; writes a .bak backup on first change.

import re
from pathlib import Path

ROOT = Path.cwd()
SIM = ROOT / "src" / "nfl_gpp_simulator.py"

HELPERS = r"""
def _norm_pos(p):
    p = str(p or "").upper().strip()
    return "DST" if p in ("D","DEF","DS","D/ST") else p

def _sf(x, default=0.0):
    try:
        if x is None: return float(default)
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none"): return float(default)
        return float(s)
    except Exception:
        return float(default)
"""

# Inject helpers right after imports
def ensure_helpers(src: str) -> str:
    if "_norm_pos(" in src and "_sf(" in src:
        return src
    m = re.search(r"(\nfrom\s+[^\n]+\n|^import\s+[^\n]+\n)+", src, flags=re.MULTILINE)
    block = HELPERS
    if m:
        return src[:m.end()] + block + src[m.end():]
    else:
        return block + "\n" + src

# Normalize projections df on ingest in __init__
PROJ_NORM_BLOCK = r"""
        # --- Begin: normalize projections table (pos/team) ---
        try:
            if hasattr(self, "players_df") and self.players_df is not None:
                if "pos" in self.players_df.columns:
                    self.players_df["pos"] = self.players_df["pos"].apply(_norm_pos)
                if "Position" in self.players_df.columns:
                    self.players_df["Position"] = self.players_df["Position"].apply(_norm_pos)
                if "team" in self.players_df.columns:
                    self.players_df["team"] = (
                        self.players_df["team"].astype(str).str.upper().str.strip().replace({"LA":"LAR"})
                    )
        except Exception:
            pass
        # --- End: normalize projections table ---
"""

def inject_proj_norm(src: str) -> str:
    # place after load_projections() call inside __init__
    m = re.search(r"(self\.load_projections\([^\)]*\)\s*\n)", src)
    if m and PROJ_NORM_BLOCK not in src:
        return src[:m.end()] + PROJ_NORM_BLOCK + src[m.end():]
    return src

# Add numeric coercion + DST ID backfill + guard at start of get_optimal()
CLEANUP_BLOCK = r"""
        # --- Begin: clean numeric fields + DST backfill before building LP ---
        try:
            # 1) normalize Position/TeamAbbrev keys inside player_dict
            for _k, _rec in list(self.player_dict.items()):
                # unify Position with DST canonicalization
                if "Position" in _rec:
                    _rec["Position"] = _norm_pos(_rec.get("Position"))
                elif "pos" in _rec:
                    _rec["Position"] = _norm_pos(_rec.get("pos"))
                # team to uppercase key TeamAbbrev
                if "TeamAbbrev" not in _rec:
                    team_guess = _rec.get("TeamAbbrev") or _rec.get("Team") or _rec.get("TeamAbbreviation") or ""
                    _rec["TeamAbbrev"] = str(team_guess or "").upper()

            # 2) backfill DST IDs by team if missing
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
                    _id_raw = _rec.get("ID")
                    try:
                        _id_ok = int(str(_id_raw).replace(",",""))
                    except Exception:
                        _id_ok = 0
                    if not _id_ok:
                        t = _rec.get("TeamAbbrev")
                        if not t and hasattr(self, "players_df") and {"name","team"}.issubset(set(self.players_df.columns)):
                            try:
                                nm = str(_rec.get("Name","")).strip().lower()
                                t = str(self.players_df.loc[self.players_df["name"].str.lower()==nm, "team"].iloc[0]).upper()
                            except Exception:
                                t = ""
                        pid = _dst_id_by_team_lookup(t)
                        if pid:
                            _rec["ID"] = pid

            # 3) coerce numerics used by the LP
            # Try to pick projection from common keys, default 0.0
            PROJ_KEYS = ("Projection","projections_proj","proj","fpts_proj","projected_points","fpts","points")
            SAL_KEYS  = ("Salary","salary","sal","cost","dk_salary")
            OWN_KEYS  = ("own","ownership","proj_own","ownership_proj")
            CEIL_KEYS = ("ceil","ceiling","fpts_ceil","projection_ceil")
            STD_KEYS  = ("stddev","stdev","sd","projection_std","fpts_std")

            for _k, _rec in list(self.player_dict.items()):
                # projection
                _p = None
                for kk in PROJ_KEYS:
                    if kk in _rec and _rec[kk] not in (None, ""):
                        _p = _rec[kk]; break
                if _p is None and hasattr(self, "players_df") and "name" in self.players_df.columns:
                    try:
                        nm = str(_rec.get("Name","")).strip().lower()
                        _row = self.players_df.loc[self.players_df["name"].str.lower()==nm]
                        if not _row.empty:
                            for alt in PROJ_KEYS:
                                if alt in _row.columns:
                                    _p = _row[alt].iloc[0]; break
                    except Exception:
                        pass
                _rec["Projection"] = _sf(_p, 0.0)

                # salary
                _s = None
                for kk in SAL_KEYS:
                    if kk in _rec and _rec[kk] not in (None, ""):
                        _s = _rec[kk]; break
                _rec["Salary"] = _sf(_s, 0.0)

                # ownership (optional)
                _o = None
                for kk in OWN_KEYS:
                    if kk in _rec and _rec[kk] not in (None, ""):
                        _o = _rec[kk]; break
                _rec["Own"] = _sf(_o, 0.0)

                # ceiling/stddev (optional)
                _c = None
                for kk in CEIL_KEYS:
                    if kk in _rec and _rec[kk] not in (None, ""):
                        _c = _rec[kk]; break
                _rec["Ceil"] = _sf(_c, 0.0)

                _sd = None
                for kk in STD_KEYS:
                    if kk in _rec and _rec[kk] not in (None, ""):
                        _sd = _rec[kk]; break
                _rec["STDDEV"] = _sf(_sd, 0.0)

            # 4) diagnostics + guard: ensure no None leaked into Projection
            bad = [(_k, _rec.get("Name",""), _rec.get("Position","")) for _k,_rec in self.player_dict.items()
                   if not isinstance(_rec.get("Projection"), (int,float))]
            if bad:
                raise AssertionError(f"Simulator: non-numeric Projection for {len(bad)} players (e.g., {bad[:3]}).")
            # ensure we actually have at least one DST in the pool
            pos_counts = {}
            for _rec in self.player_dict.values():
                p = str(_rec.get("Position","")).upper()
                pos_counts[p] = pos_counts.get(p, 0) + 1
            if pos_counts.get("DST",0) <= 0:
                raise AssertionError(f"Simulator: no DST players after normalization/backfill. POS counts: {pos_counts}")
        except Exception as _e:
            # Surface rich context to Streamlit; the main try will bubble this up
            raise
        # --- End: clean numeric fields + DST backfill ---
"""

def inject_cleanup_get_optimal(src: str) -> str:
    # Insert at the start of get_optimal() body
    m = re.search(r"(def\s+get_optimal\s*\(self[^\)]*\)\s*:\s*\n)", src)
    if m and CLEANUP_BLOCK not in src:
        return src[:m.end()] + CLEANUP_BLOCK + src[m.end():]
    return src

def main():
    if not SIM.exists():
        raise SystemExit("Could not find src/nfl_gpp_simulator.py. Run from repo root.")
    src = SIM.read_text(encoding="utf-8")
    orig = src
    src = ensure_helpers(src)
    src = inject_proj_norm(src)
    src = inject_cleanup_get_optimal(src)
    if src != orig:
        bak = SIM.with_suffix(".py.bak")
        bak.write_text(orig, encoding="utf-8")
        SIM.write_text(src, encoding="utf-8")
        print(f"✅ Patched {SIM} (backup at {bak})")
    else:
        print("ℹ️ No changes applied (file may already be patched).")
if __name__ == "__main__":
    main()
