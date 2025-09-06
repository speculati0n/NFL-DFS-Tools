#!/usr/bin/env python3
# scripts/apply_dst_fixes.py
# Apply in-place fixes to src/nfl_optimizer.py for DST normalization and robust sorting.
import re, sys
from pathlib import Path

def patch_file(path: Path):
    src = path.read_text(encoding="utf-8")
    orig = src

    # 1) Fix _to_player mapping: normalize D/DEF -> DST (do NOT flip DST->D)
    src = re.sub(
        r'pos\s*=\s*data\["Position"\]\s*\n\s*if\s+pos\s*==\s*"DST":\s*\n\s*pos\s*=\s*"D"',
        'pos = data["Position"]\n            if pos in ("D", "DEF"):\n                pos = "DST"',
        src
    )

    # 2) Normalize Position across player_dict before sorting/output()
    src = re.sub(
        r'(\n\s*sorted_lineups\s*=\s*\[\])',
        '\n        # Normalize Position field in player_dict (D/DEF -> DST) before ordering\n'
        '        for _k, _rec in self.player_dict.items():\n'
        '            p = str(_rec.get("Position","" )).upper()\n'
        '            if p in ("D","DEF"):\n'
        '                _rec["Position"] = "DST"\n'
        r'\1',
        src, count=1
    )

    # 3) Replace sort_lineup with robust version that treats D/DEF as DST
    sort_pat = re.compile(
        r'def\s+sort_lineup\(self,\s*lineup\):.*?(?=^\s*def\s|\Z)',
        re.DOTALL | re.MULTILINE
    )
    new_sort = r'''

def sort_lineup(self, lineup):
        copy_lineup = copy.deepcopy(lineup)
        positional_order = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
        final_lineup = []

        def _norm(p):
            p = str(p or "").upper().strip()
            return "DST" if p in ("D","DEF") else p

        # Sort players based on their positional order
        for position in positional_order:
            if position != "FLEX":
                eligible_players = [
                    player
                    for player in copy_lineup
                    if _norm(self.player_dict[player]["Position"]) == position
                ]
                if eligible_players:
                    eligible_player = eligible_players[0]
                    final_lineup.append(eligible_player)
                    copy_lineup.remove(eligible_player)
                else:
                    final_lineup.append(None)
            else:
                eligible_players = [
                    player
                    for player in copy_lineup
                    if _norm(self.player_dict[player]["Position"]) in ["RB", "WR", "TE"]
                ]
                if eligible_players:
                    eligible_player = eligible_players[0]
                    final_lineup.append(eligible_player)
                    copy_lineup.remove(eligible_player)
                else:
                    final_lineup.append(None)

        final_lineup = [p for p in final_lineup if p is not None]
        return final_lineup
'''.lstrip("\n")

    if sort_pat.search(src):
        src = sort_pat.sub(new_sort, src, count=1)
    else:
        print("WARNING: sort_lineup not found; skipping replacement.")

    if src != orig:
        backup = path.with_suffix(".py.bak")
        backup.write_text(orig, encoding="utf-8")
        path.write_text(src, encoding="utf-8")
        print(f"✅ Patched {path} (backup at {backup})")
    else:
        print("No changes applied (already patched?)")

def main():
    repo_root = Path.cwd()
    p = repo_root / "src" / "nfl_optimizer.py"
    if not p.exists():
        print("Could not locate src/nfl_optimizer.py. Run this from repo root.")
        sys.exit(1)
    patch_file(p)

if __name__ == "__main__":
    main()
