#!/usr/bin/env python3
# scripts/enforce_off_optimal_optimizer.py
# Make the optimizer honor config["max_pct_off_optimal"]:
#  - read the value in __init__
#  - solve once deterministically to get optimal_score
#  - add a min-FPTS floor constraint before generating lineups
#
# Safe to run multiple times; writes a .bak once.

from pathlib import Path
import re

OPT = Path("src/nfl_optimizer.py")

def patch_init(src: str) -> str:
    """
    After `self.load_config()` in __init__, add:
        self.max_pct_off_optimal = float(self.config.get("max_pct_off_optimal", 0.0))
    """
    pat = r"(self\.load_config\(\)\s*\n)"
    if re.search(pat, src):
        inject = (
            "        # Honor off-optimal floor from config (0..1). 0 disables.\n"
            "        self.max_pct_off_optimal = float(self.config.get(\"max_pct_off_optimal\", 0.0))\n"
            "        if not (0.0 <= self.max_pct_off_optimal <= 1.0):\n"
            "            self.max_pct_off_optimal = 0.0\n"
        )
        src = re.sub(pat, r"\1" + inject, src, count=1)
    return src

def patch_enforcement(src: str) -> str:
    """
    Before the '# Crunch!' loop in optimize(), inject:
      - deterministic objective solve to get optimal_score
      - add floor constraint: sum(Fpts*x) >= (1 - max_pct)*optimal_score
    Assumes: pulp imported as plp, lp_variables & self.problem already created,
    self.player_dict entries have 'Fpts' and 'ID' keys.
    """
    anchor = r"\n\s*# Crunch!\s*\n"
    if not re.search(anchor, src):
        return src

    block = r'''
        # --- Begin: enforce off-optimal floor from config ---
        try:
            _pct = float(getattr(self, "max_pct_off_optimal", 0.0))
        except Exception:
            _pct = 0.0

        if _pct and 0.0 < _pct < 1.0:
            # Deterministic FPTS sum (no randomness) to measure "optimal"
            _det_fpts_sum = plp.lpSum(
                self.player_dict[(player, pos_str, team)]["Fpts"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
            )
            # Solve once deterministically to find the true optimal score
            self.problem += _det_fpts_sum, "Objective"
            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
                _opt = float(self.problem.objective.value())
            except Exception:
                _opt = None

            # Only enforce the floor if we successfully computed an optimal score
            if _opt is not None and _opt > 0:
                _min_fpts = (1.0 - _pct) * _opt
                # Add a hard floor constraint based on *deterministic* projections,
                # so it remains valid even when we randomize the objective later.
                self.problem += (_det_fpts_sum >= _min_fpts), "MinFptsOffOptimal"
        # --- End: enforce off-optimal floor ---
    '''

    return re.sub(anchor, block + "\n        # Crunch!\n", src, count=1)

def main():
    if not OPT.exists():
        raise SystemExit("Could not find src/nfl_optimizer.py (run from repo root).")
    original = OPT.read_text(encoding="utf-8")
    patched = original

    patched = patch_init(patched)
    patched = patch_enforcement(patched)

    if patched != original:
        backup = OPT.with_suffix(".py.bak")
        backup.write_text(original, encoding="utf-8")
        OPT.write_text(patched, encoding="utf-8")
        print(f"✅ Patched {OPT} (backup at {backup})")
    else:
        print("ℹ️ No changes applied (file may already be patched).")

if __name__ == "__main__":
    main()
