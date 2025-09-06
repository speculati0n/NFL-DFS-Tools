#!/usr/bin/env python3
# scripts/apply_ingest_dst_fix.py
# Patches src/nfl_optimizer.py to fix DST ingest + ID matching and guard against empty DST pool.
import re
from pathlib import Path

TARGET = Path("src") / "nfl_optimizer.py"

def patch():
    path = TARGET if TARGET.exists() else Path(__file__).resolve().parents[1] / "src" / "nfl_optimizer.py"
    src = path.read_text(encoding="utf-8")
    orig = src

    # --- 1) In DK branch of load_player_ids, normalize position to DST ---
    # Insert right after: position = row["position"]
    src = re.sub(
        r'(def\s+load_player_ids\([^\)]*\):\s*\n\s*with open\(path[^\n]+\n\s*reader\s*=\s*csv\.DictReader[^\n]+\n\s*for row in reader:\s*\n\s*if self\.site\s*==\s*"dk":\s*\n\s*position\s*=\s*row\["position"\]\s*\n)',
        r'\1            if position in ("D","DEF"):\n                position = "DST"\n',
        src
    )

    # --- 2) In load_projections, use normalized 'position' in projection_minimum filter ---
    # Replace the filter that references row["pos"] with one that uses 'position'
    src = re.sub(
        r'(\n\s*if\s*\(\s*float\(row\["projections_proj"\]\)\s*<\s*self\.projection_minimum\s*\)\s*and\s*row\["pos"\]\s*!=\s*"DST"\s*\):\s*\n\s*continue\s*\n)',
        '\n                if (float(row["projections_proj"]) < self.projection_minimum) and position != "DST":\n                    continue\n',
        src
    )

    # --- 3) Add early guard in optimize(): ensure there is at least one DST candidate ---
    src = re.sub(
        r'(def\s+optimize\(\s*self[^\)]*\)\s*:\s*\n)',
        r'\1        # Guard: Ensure DST pool exists before building variables\n'
        r'        if not any(v.get("Position") == "DST" for v in self.player_dict.values()):\n'
        r'            raise AssertionError("No DST candidates after ingest & ID match. Check projections pos (D/DEF->DST) and player_ids mapping.")\n',
        src, count=1
    )

    if src != orig:
        backup = path.with_suffix(".py.bak")
        backup.write_text(orig, encoding="utf-8")
        path.write_text(src, encoding="utf-8")
        print(f"✅ Patched {path} (backup at {backup})")
    else:
        print("ℹ️ No changes applied (already patched?)")

if __name__ == "__main__":
    patch()
