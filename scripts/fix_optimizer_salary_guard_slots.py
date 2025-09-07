#!/usr/bin/env python3
# scripts/fix_optimizer_salary_guard_slots.py
#
# Update the optimizer’s runtime salary guard to use the *slotted nine*:
#   players_ids -> get_players() -> select_slot_players() -> sum 9
#
# Idempotent: won’t double-insert. Backs up original as .bak.

from pathlib import Path
import re
import sys

OPT = Path("src/nfl_optimizer.py")
MARKER = "# --- Salary bounds guard ---"

def patch_guard(src: str) -> str:
    if MARKER not in src:
        # Guard not present (older file) — nothing to patch here.
        return src

    # Replace the block starting at the guard marker through the next lineup append
    # with a version that uses get_players() + select_slot_players()
    pattern = re.compile(
        r"(?P<prefix>\n\s*)# --- Salary bounds guard ---.*?"
        r"self\.lineups\.append\(\(players,\s*det_proj\)\)\s*",
        re.DOTALL
    )

    def repl(m):
        lead = m.group("prefix")
        block = f"""{lead}{MARKER}
{lead}# Convert chosen variable IDs -> Player objects, then slot to nine
{lead}players_ids = players
{lead}players_objs = self.get_players(players_ids)
{lead}slots = self.select_slot_players(players_objs)
{lead}nine = [slots[\"QB\"], slots[\"RB1\"], slots[\"RB2\"], slots[\"WR1\"], slots[\"WR2\"], slots[\"WR3\"], slots[\"TE\"], slots[\"FLEX\"], slots[\"DST\"]]
{lead}
{lead}# Deterministic totals based on the *nine* that will be exported
{lead}det_proj   = sum(p.proj   for p in nine)
{lead}det_salary = sum(p.salary for p in nine)
{lead}
{lead}# Active cap/floor (match LP constraints)
{lead}max_salary = 50000 if self.site == \"dk\" else 60000
{lead}min_salary = self.min_lineup_salary if self.min_lineup_salary else (45000 if self.site == \"dk\" else 55000)
{lead}
{lead}# Enforce at runtime (tiny epsilon for float safety)
{lead}if det_salary > max_salary + 1e-6 or det_salary < min_salary - 1e-6:
{lead}    raise AssertionError(
{lead}        f\"Lineup salary {{det_salary}} out of bounds \"
{lead}        f\"(site={{self.site}}, cap={{max_salary}}, floor={{min_salary}}). \"
{lead}        f\"QB={{nine[0].name}}, RB1={{nine[1].name}}, RB2={{nine[2].name}}, \"
{lead}        f\"WR1={{nine[3].name}}, WR2={{nine[4].name}}, WR3={{nine[5].name}}, \"
{lead}        f\"TE={{nine[6].name}}, FLEX={{nine[7].name}}, DST={{nine[8].name}}\"
{lead}    )
{lead}
{lead}# Store lineup with deterministic projection (same metric as constraints)
{lead}self.lineups.append((players, det_proj))
"""
        return block

    new_src, n = pattern.subn(repl, src, count=1)
    return new_src if n else src

def main():
    if not OPT.exists():
        print("Could not find src/nfl_optimizer.py. Run from repo root.", file=sys.stderr)
        sys.exit(2)

    src = OPT.read_text(encoding="utf-8")
    orig = src
    src = patch_guard(src)

    if src != orig:
        bak = OPT.with_suffix(".py.bak")
        bak.write_text(orig, encoding="utf-8")
        OPT.write_text(src, encoding="utf-8")
        print(f"✅ Patched {OPT} (backup at {bak})")
    else:
        print("ℹ️ No changes applied (guard not found or already patched).")

if __name__ == "__main__":
    main()
