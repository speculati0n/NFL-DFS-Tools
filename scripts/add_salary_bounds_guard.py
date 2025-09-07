#!/usr/bin/env python3
# scripts/add_salary_bounds_guard.py
#
# Patches src/nfl_optimizer.py to:
#  1) Log active site/cap/floor before lineup loop (Streamlit if available).
#  2) After each solve, compute det_salary/det_proj from self.player_dict[players].
#  3) Assert salary is within [min_floor, cap] with tiny epsilon.
#  4) Save lineup tuples with deterministic projection (det_proj).
#
# Safe to run multiple times; it won't double-insert (uses a marker).

from pathlib import Path
import re
import sys

OPT = Path("src/nfl_optimizer.py")
MARKER_GUARD = "# --- Salary bounds guard ---"
MARKER_INFO  = "# --- Optimizer cap/floor info ---"

def insert_info_block(src: str) -> str:
    """Insert a one-liner info block right before the lineup loop, after '# Crunch!'."""
    if MARKER_INFO in src:
        return src
    # Find "# Crunch!" then the first 'for i in range(' after it
    m_crunch = re.search(r"\n\s*#\s*Crunch!\s*\n", src)
    if not m_crunch:
        return src
    start = m_crunch.end()
    info_block = f"""
        {MARKER_INFO}
        try:
            import streamlit as st
            st.info(
                f"Optimizer: site={{self.site}}, cap={{50000 if self.site=='dk' else 60000}}, "
                f"min_floor={{self.min_lineup_salary or (45000 if self.site=='dk' else 55000)}}"
            )
        except Exception:
            pass
    """
    return src[:start] + info_block + src[start:]

def insert_salary_guard_after_players(src: str) -> str:
    """Right after 'players = [k for k,v in lp_variables.items() if v.value() == 1]' insert the guard."""
    if MARKER_GUARD in src:
        return src

    # robust pattern for the players list creation
    patt_players = re.compile(
        r"(?P<lead>\n\s*)players\s*=\s*\[\s*k\s+for\s+k,\s*v\s+in\s+lp_variables\.items\(\)\s+if\s+v\.value\(\)\s*==\s*1\s*\]\s*\n",
        re.MULTILINE
    )
    m = patt_players.search(src)
    if not m:
        # fallback: try a slightly looser match
        patt_players_loose = re.compile(r"\n\s*players\s*=\s*\[.*?v\.value\(\)\s*==\s*1.*?\]\s*\n", re.DOTALL)
        m = patt_players_loose.search(src)
    if not m:
        return src  # cannot find, leave untouched

    lead = m.groupdict().get("lead", "\n        ")
    guard_block = f"""{lead}{MARKER_GUARD}
{lead}# Deterministic projection & salary for the chosen players
{lead}det_proj   = sum(self.player_dict[key]["Fpts"]   for key in players)
{lead}det_salary = sum(self.player_dict[key]["Salary"] for key in players)
{lead}
{lead}# Active cap/floor (match LP constraints)
{lead}max_salary = 50000 if self.site == "dk" else 60000
{lead}min_salary = self.min_lineup_salary if self.min_lineup_salary else (45000 if self.site == "dk" else 55000)
{lead}
{lead}# Enforce at runtime (tiny epsilon for float safety)
{lead}if det_salary > max_salary + 1e-6 or det_salary < min_salary - 1e-6:
{lead}    raise AssertionError(
{lead}        f"Lineup salary {{det_salary}} out of bounds "
{lead}        f"(site={{self.site}}, cap={{max_salary}}, floor={{min_salary}})."
{lead}    )
"""

    insert_at = m.end()
    src = src[:insert_at] + guard_block + src[insert_at:]
    return src

def replace_lineup_append_to_det_proj(src: str) -> str:
    """
    Replace:
        fpts_used = self.problem.objective.value()
        self.lineups.append((players, fpts_used))
    with:
        self.lineups.append((players, det_proj))
    If already deterministic, leave as-is.
    """
    # If already appending det_proj, nothing to do
    if re.search(r"self\.lineups\.append\(\s*\(players\s*,\s*det_proj\s*\)\s*\)", src):
        return src

    # Nuke any immediate 'fpts_used = ...' followed by append with fpts_used
    src = re.sub(
        r"\n\s*fpts_used\s*=\s*self\.problem\.objective\.value\(\)\s*\n\s*self\.lineups\.append\(\s*\(players\s*,\s*fpts_used\s*\)\s*\)\s*",
        "\n        self.lineups.append((players, det_proj))\n",
        src
    )
    # If there was only the append with fpts_used
    src = re.sub(
        r"\n\s*self\.lineups\.append\(\s*\(players\s*,\s*fpts_used\s*\)\s*\)\s*",
        "\n        self.lineups.append((players, det_proj))\n",
        src
    )
    return src

def main():
    if not OPT.exists():
        print("Could not find src/nfl_optimizer.py. Run from repo root.", file=sys.stderr)
        sys.exit(2)

    src = OPT.read_text(encoding="utf-8")
    orig = src

    src = insert_info_block(src)
    src = insert_salary_guard_after_players(src)
    src = replace_lineup_append_to_det_proj(src)

    if src != orig:
        bak = OPT.with_suffix(".py.bak")
        bak.write_text(orig, encoding="utf-8")
        OPT.write_text(src, encoding="utf-8")
        print(f"✅ Patched {OPT} (backup at {bak})")
    else:
        print("ℹ️ No changes applied (markers present or patterns not found).")

if __name__ == "__main__":
    main()
