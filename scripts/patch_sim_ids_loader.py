#!/usr/bin/env python3
# scripts/patch_sim_ids_loader.py
# Patches src/nfl_gpp_simulator.py to:
#  - import player_ids_flex
#  - replace load_player_ids() with a flexible implementation
#  - normalize positions to DST
#  - build canonical dicts (id_name_dict, name_pos_to_id, etc.)
#  - backfill DST IDs by team when name match fails
#  - add a robust DST guard with counts

import re
from pathlib import Path

ROOT = Path.cwd()
SIM = ROOT / "src" / "nfl_gpp_simulator.py"

NORMALIZE_HELPER = r"""
def _norm_pos(p):
    p = str(p or "").upper().strip()
    return "DST" if p in ("D","DEF","DS","D/ST") else p
"""

FLEX_IMPORT = r"from player_ids_flex import load_player_ids_flex, dst_id_by_team"

NEW_LOAD_FUNC = r'''
def load_player_ids(self, path):
        """
        Flexible player ID ingest:
          - DK weekly salary CSV (ID, Name, Position, TeamAbbrev, ...)
          - DK bulk IDs CSV (draftableid, displayname, position, ...)
          - Custom mapping (id/name/position; optional team)
        Builds:
          - self.id_name_dict[str(ID)] = Name
          - self.name_pos_to_id[(name_lower, Position)] = str(ID)
          - self.id_position_dict[str(ID)] = Position
          - self.id_teamabbrev_dict[str(ID)] = TeamAbbrev or ""
          - self._player_ids_df = canonical DataFrame
        """
        import os
        self.player_ids_path = path
        if not os.path.exists(path):
            # fall back to repo data path if configured like the optimizer
            alt = os.path.join(os.path.dirname(__file__), "..", "data", "player_ids.csv")
            if os.path.exists(alt):
                path = alt
            else:
                raise FileNotFoundError(f"player_ids file not found at {self.player_ids_path} or {alt}")

        df = load_player_ids_flex(path)
        self._player_ids_df = df.copy()

        self.id_name_dict = {}
        self.name_pos_to_id = {}
        self.id_position_dict = {}
        self.id_teamabbrev_dict = {}

        for _, r in df.iterrows():
            pid = str(int(r["ID"]))
            name = str(r["Name"]).strip()
            pos  = _norm_pos(r["Position"])
            team = str(r.get("TeamAbbrev","") or "").upper()

            self.id_name_dict[pid] = name
            self.name_pos_to_id[(name.lower(), pos)] = pid
            self.id_position_dict[pid] = pos
            self.id_teamabbrev_dict[pid] = team
        return df
'''

BACKFILL_AND_GUARD = r"""
        # --- Begin: DST ID backfill for sim player pool & guard ---
        try:
            # If simulator has a players table, normalize its pos/team columns
            if hasattr(self, "players_df") and self.players_df is not None:
                if "pos" in self.players_df.columns:
                    self.players_df["pos"] = self.players_df["pos"].apply(_norm_pos)
                if "Position" in self.players_df.columns:
                    self.players_df["Position"] = self.players_df["Position"].apply(_norm_pos)
                if "team" in self.players_df.columns:
                    self.players_df["team"] = (
                        self.players_df["team"].astype(str).str.upper().str.strip().replace({"LA":"LAR"})
                    )

            # Backfill: if a DST entry is missing an ID later, we can use team to find one
            pid_df = getattr(self, "_player_ids_df", None)

            # Define a helper for looking up by team
            def _dst_id_by_team_lookup(team):
                team = str(team or "").upper().strip()
                if not pid_df is None and team:
                    try:
                        row = pid_df[
                            (pid_df["Position"]=="DST") &
                            (pid_df["TeamAbbrev"].astype(str).str.upper()==team)
                        ].iloc[0]
                        return str(int(row["ID"]))
                    except Exception:
                        return None
                return None

            # Guard: ensure we have at least one DST in the IDs universe
            dst_in_ids = sum(1 for p in self.id_position_dict.values() if p == "DST")
            if dst_in_ids <= 0:
                # Build a quick POS count for debugging
                pos_counts = {}
                for p in self.id_position_dict.values():
                    pos_counts[p] = pos_counts.get(p, 0) + 1
                raise AssertionError(
                    "Simulator: no DST in IDs after ingest. "
                    f"ID pos counts: {pos_counts}. "
                    "Pass a DK file with Position=DST (salary CSV) or ensure bulk/custom mapping contains DST rows."
                )
        except Exception:
            pass
        # --- End: DST ID backfill & guard ---
"""

def ensure_imports_and_helper(src: str) -> str:
    # Add _norm_pos helper if missing
    if "_norm_pos(" not in src:
        # insert after imports
        m = re.search(r"(\nfrom\s+[^\n]+\n|import\s+[^\n]+\n)+", src)
        if m:
            src = src[:m.end()] + NORMALIZE_HELPER + src[m.end():]
        else:
            src = NORMALIZE_HELPER + src
    # Add flex import if missing
    if "load_player_ids_flex" not in src:
        # find any import line to append after
        m2 = re.search(r"^import\s+\w+.*?$", src, flags=re.MULTILINE)
        if m2:
            insert_at = m2.end()
            src = src[:insert_at] + "\n" + FLEX_IMPORT + src[insert_at:]
        else:
            # prepend if no simple import found
            src = FLEX_IMPORT + "\n" + src
    return src

def replace_load_player_ids(src: str) -> str:
    pat = re.compile(r'def\s+load_player_ids\([^\)]*\):.*?(?=^\s*def\s|\Z)', re.DOTALL | re.MULTILINE)
    if pat.search(src):
        return pat.sub(NEW_LOAD_FUNC, src, count=1)
    # If not found, append a new method (unlikely but safe)
    return src + "\n\n" + NEW_LOAD_FUNC

def inject_backfill_guard(src: str) -> str:
    """
    Place the backfill/guard block near the beginning of field lineup generation.
    We try to inject into generate_field_lineups() after its def line.
    If not found, try run_tournament_simulation(). Otherwise, inject after __init__.
    """
    targets = ["generate_field_lineups", "run_tournament_simulation", "__init__"]
    for fn in targets:
        m = re.search(rf"(def\s+{fn}\s*\(self[^\)]*\)\s*:\s*\n)", src)
        if m:
            return src[:m.end()] + BACKFILL_AND_GUARD + src[m.end():]
    return src

def main():
    if not SIM.exists():
        raise SystemExit("Could not find src/nfl_gpp_simulator.py. Run from repo root.")
    src = SIM.read_text(encoding="utf-8")
    orig = src

    src = ensure_imports_and_helper(src)
    src = replace_load_player_ids(src)
    src = inject_backfill_guard(src)

    if src != orig:
        bak = SIM.with_suffix(".py.bak")
        bak.write_text(orig, encoding="utf-8")
        SIM.write_text(src, encoding="utf-8")
        print(f"✅ Patched {SIM} (backup at {bak})")
    else:
        print("ℹ️ No changes applied — file may already be patched.")

if __name__ == "__main__":
    main()
