#!/usr/bin/env python3
"""
Patch the inner get_corr_value inside run_simulation_for_game() to use canonical
positions (QB/RB/WR/TE/DST) and default to 0.0 when a correlation key is missing.
Idempotent: backs up original file once as .bak.
"""
from pathlib import Path
import re
import sys

SIM = Path("src/nfl_gpp_simulator.py")

def main():
    if not SIM.exists():
        print("Could not find src/nfl_gpp_simulator.py. Run from repo root.", file=sys.stderr)
        sys.exit(2)

    src = SIM.read_text(encoding="utf-8", errors="ignore")

    # sanity: we rely on the canonical helpers already present in your file
    if "_canon_pos_primary(" not in src:
        print("Missing _canon_pos_primary helper; please run the prior position-canonicalization patch first.", file=sys.stderr)
        sys.exit(2)

    m_run = re.search(r"\n\s*@staticmethod\s*\n\s*def\s+run_simulation_for_game\s*\(", src)
    if not m_run:
        print("run_simulation_for_game() not found.", file=sys.stderr)
        sys.exit(2)

    m_inner = re.search(r"\n(?P<indent>\s*)def\s+get_corr_value\s*\(\s*player1\s*,\s*player2\s*\)\s*:\s*\n", src[m_run.start():])
    if not m_inner:
        print("Inner get_corr_value not found inside run_simulation_for_game().", file=sys.stderr)
        sys.exit(2)

    indent = m_inner.group("indent")
    inner_start = m_run.start() + m_inner.start()

    m_build = re.search(rf"\n{indent}def\s+build_covariance_matrix\s*\(", src[inner_start:])
    if not m_build:
        print("build_covariance_matrix() not found after inner get_corr_value.", file=sys.stderr)
        sys.exit(2)
    inner_end = inner_start + m_build.start()

    new_inner = f"""
\n{indent}def get_corr_value(player1, player2):
{indent}    \"\"\"Robust correlation: canonical positions + safe defaults.\"\"\" 
{indent}    # Player-specific override first
{indent}    try:
{indent}        pc = player1.get("Player Correlations", {{}})
{indent}        if player2.get("Name") in pc:
{indent}            return float(pc[player2["Name"]])
{indent}    except Exception:
{indent}        pass
{indent}
{indent}    # Canonicalize primary positions
{indent}    try:
{indent}        pos1 = _canon_pos_primary(player1.get("Position"))
{indent}    except Exception:
{indent}        pos1 = ""
{indent}    try:
{indent}        pos2 = _canon_pos_primary(player2.get("Position"))
{indent}    except Exception:
{indent}        pos2 = ""
{indent}    if not pos1 or not pos2:
{indent}        return 0.0
{indent}
{indent}    # Base correlations by position (fallback)
{indent}    position_correlations = {{
{indent}        "QB": -0.5,
{indent}        "RB": -0.2,
{indent}        "WR":  0.1,
{indent}        "TE": -0.2,
{indent}        "K":  -0.5,
{indent}        "DST": -0.5,
{indent}    }}
{indent}
{indent}    # Same team & same canonical pos -> base table
{indent}    try:
{indent}        same_team = player1.get("Team") == player2.get("Team")
{indent}    except Exception:
{indent}        same_team = False
{indent}
{indent}    if same_team and pos1 == pos2:
{indent}        return float(position_correlations.get(pos1, 0.0))
{indent}
{indent}    # Else use player1["Correlations"] with canonical keys
{indent}    try:
{indent}        corr = player1.get("Correlations", {{}})
{indent}        key = (f"Opp {{pos2}}") if not same_team else pos2
{indent}        return float(corr.get(key, 0.0))
{indent}    except Exception:
{indent}        return 0.0
"""

    patched = src[:inner_start] + new_inner + src[inner_end:]
    if patched == src:
        print("No changes applied (already patched).")
        return

    SIM.with_suffix(".py.bak").write_text(src, encoding="utf-8")
    SIM.write_text(patched, encoding="utf-8")
    print(f"✅ Patched inner get_corr_value in {SIM}")

if __name__ == "__main__":
    main()
