#!/usr/bin/env python3
# scripts/patch_dst_pipeline.py
# Patches src/nfl_optimizer.py to:
#  1) Normalize Position in self.player_dict (pos/Position -> Position) with DST mapping
#  2) Backfill DST IDs by team when name-based match failed
#  3) Move/replace the DST guard to run AFTER normalization/backfill and check both keys

import re
from pathlib import Path

ROOT = Path.cwd()
OPT = ROOT / "src" / "nfl_optimizer.py"

NORMALIZE_HELPER = r"""
def _normalize_pos_key_and_value(rec):
    # Ensure dict has 'Position' key with normalized value (D/DEF/DS/D/ST->DST).
    def _norm(p):
        p = str(p or "").strip().upper()
        return "DST" if p in ("D","DEF","DS","D/ST") else p
    # Source may have 'pos' or 'Position'
    if "Position" in rec:
        rec["Position"] = _norm(rec.get("Position"))
    elif "pos" in rec:
        rec["Position"] = _norm(rec.get("pos"))
    else:
        # Nothing present; leave as is
        pass
    return rec
"""

BACKFILL_BLOCK = r"""
        # --- Begin: DST ID backfill & normalization ---
        try:
            # Normalize all player_dict records to have Position with DST canonicalized
            for _k, _rec in list(self.player_dict.items()):
                _normalize_pos_key_and_value(_rec)
                # copy team if missing into a consistent key
                if "TeamAbbrev" not in _rec:
                    team_guess = _rec.get("TeamAbbrev") or _rec.get("Team") or _rec.get("TeamAbbreviation") or ""
                    _rec["TeamAbbrev"] = str(team_guess or "").upper()

            # Prepare a player-ids dataframe for team-based DST lookup
            pid_df = getattr(self, "_player_ids_df", None)
            if pid_df is None:
                try:
                    from player_ids_flex import load_player_ids_flex
                    # Try the path recorded during load
                    pid_path = getattr(self, "player_ids_path", "data/player_ids.csv")
                    pid_df = load_player_ids_flex(pid_path)
                    self._player_ids_df = pid_df.copy()
                except Exception:
                    pid_df = None

            # Helper: lookup DST ID by team
            def _dst_id_by_team(team):
                team = str(team or "").upper().strip()
                if not pid_df is None:
                    try:
                        row = pid_df[
                            (pid_df["Position"]=="DST") &
                            (pid_df["TeamAbbrev"].astype(str).str.upper()==team)
                        ].iloc[0]
                        return int(row["ID"])
                    except Exception:
                        return None
                return None

            # Backfill IDs for DST when missing
            for _k, _rec in list(self.player_dict.items()):
                if _rec.get("Position") == "DST":
                    # recognize missing/zero IDs
                    _id_raw = _rec.get("ID")
                    try:
                        _id_int = int(str(_id_raw).replace(",",""))
                    except Exception:
                        _id_int = 0
                    if not _id_int:
                        team = str(_rec.get("TeamAbbrev") or _rec.get("Team") or _rec.get("TeamAbbreviation") or "").upper()
                        if not team and hasattr(self, "players_df") and "name" in self.players_df.columns and "team" in self.players_df.columns:
                            try:
                                nm = str(_rec.get("Name","" )).strip().lower()
                                team = str(self.players_df.loc[self.players_df["name"].str.lower()==nm, "team"].iloc[0]).upper()
                            except Exception:
                                team = ""
                        if team:
                            pid = _dst_id_by_team(team)
                            if pid:
                                _rec["ID"] = int(pid)
        except Exception:
            pass
        # --- End: DST ID backfill & normalization ---
"""

DST_GUARD = r"""
        # --- Begin: DST pool guard AFTER normalization/backfill ---
        try:
            pos_counts = {}
            def _getpos(v):
                if "Position" in v and v["Position"]:
                    return str(v["Position"]).upper()
                if "pos" in v and v["pos"]:
                    return str(v["pos"]).upper()
                return ""
            for _v in self.player_dict.values():
                p = _getpos(_v)
                pos_counts[p] = pos_counts.get(p, 0) + 1
            if (pos_counts.get("DST", 0) or 0) <= 0:
                raise AssertionError(
                    "No DST candidates after ingest & ID match. "
                    f"Counts seen: {pos_counts}. "
                    "Fix: ensure projections have pos='DST' and player_ids file has DST rows with TeamAbbrev; team backfill is applied."
                )
        except Exception as _e:
            raise
        # --- End: DST pool guard ---
"""

def ensure_imports(src: str) -> str:
    # make sure our helper is present once
    if "_normalize_pos_key_and_value" not in src:
        # insert helper after the last import block
        m = re.search(r"(\nfrom\s+[^\n]+\n|import\s+[^\n]+\n)+", src)
        if m:
            idx = m.end()
            src = src[:idx] + NORMALIZE_HELPER + src[idx:]
        else:
            src = NORMALIZE_HELPER + src
    return src

def inject_backfill_after_player_dict(src: str) -> str:
    """
    Insert the backfill block right after player_dict is finished being built,
    but before assertPlayerDict()/pruning/variable building.
    We search for a few anchors; if they fail, we fall back to inject at start of optimize().
    """
    # Try to place before a typical sanity/validation step on player_dict
    anchors = [
        r"\n\s*#\s*assertPlayerDict",          # comment marker if present
        r"\n\s*self\.assertPlayerDict\(",      # actual method call
        r"\n\s*#\s*Setup our linear programming equation",  # before LP setup
        r"\n\s*#\s*We will use PuLP as our solver"          # also before LP setup
    ]
    for pat in anchors:
        m = re.search(pat, src)
        if m:
            return src[:m.start()] + BACKFILL_BLOCK + src[m.start():]
    # Fallback: inject near start of optimize()
    m = re.search(r"(def\s+optimize\(\s*self[^\)]*\)\s*:\s*\n)", src)
    if m:
        insert_at = m.end()
        return src[:insert_at] + BACKFILL_BLOCK + src[insert_at:]
    return src

def replace_or_move_guard(src: str) -> str:
    """
    Remove any existing 'DST guard' placed too early and insert the robust guard after backfill.
    We detect a prior guard by searching 'No DST candidates after ingest' string.
    """
    # Remove existing guard occurrences
    src = re.sub(
        r"\n\s*#\s*Guard: Ensure DST.*?raise AssertionError\([^\)]+\)\s*\n",
        "\n",
        src,
        flags=re.DOTALL
    )
    # Now insert our guard after backfill (search for end marker we just injected)
    marker = r"# --- End: DST ID backfill & normalization ---"
    m = re.search(re.escape(marker), src)
    if m:
        return src[:m.end()] + DST_GUARD + src[m.end():]
    # Fallback: put guard at end of optimize() header if marker not found
    m2 = re.search(r"(def\s+optimize\(\s*self[^\)]*\)\s*:\s*\n)", src)
    if m2:
        return src[:m2.end()] + DST_GUARD + src[m2.end():]
    return src

def main():
    if not OPT.exists():
        raise SystemExit("Could not find src/nfl_optimizer.py. Run from repo root.")
    src = OPT.read_text(encoding="utf-8")
    orig = src

    # Ensure our helper exists
    src = ensure_imports(src)
    # Inject backfill/normalization after player_dict is built
    src = inject_backfill_after_player_dict(src)
    # Replace/move guard to after backfill and make it robust
    src = replace_or_move_guard(src)

    if src != orig:
        backup = OPT.with_suffix(".py.bak")
        backup.write_text(orig, encoding="utf-8")
        OPT.write_text(src, encoding="utf-8")
        print(f"✅ Patched {OPT} (backup at {backup})")
    else:
        print("ℹ️ No changes applied; file may already contain these patches")

if __name__ == "__main__":
    main()
