#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
risk_audit.py

Provides:
1) Risk table (shared by Optimizer & Simulator):
   - Inputs: fantasyyear_consistency, fantasyyear_upside, fantasyyear_duds
   - Derived: sigma_base, sigma_eff, r_plus, r_minus (+ shrink factor)
   - Context: name, pos, team, proj, floor, ceiling

2) Optimizer jitter audit WITH selection impact:
   - Tracks, per player (ID), across lineup-build iterations:
       nudged_up, nudged_down, nudged_total
       nudged_up_selected, nudged_up_not_selected
       nudged_down_selected, nudged_down_not_selected
       selected_total, selection_rate
       selected_when_up_rate, selected_when_down_rate
       avg_nudge_pts (signed), avg_abs_nudge_pts
       avg_nudge_pts_selected, avg_abs_nudge_pts_selected
   - Uses a per-iteration staging buffer so we can join the jitter sign with
     the solve result (selected vs not).

Outputs written by callers:
  - risk_table_optimizer.csv / risk_table_simulator.csv
  - risk_jitter_optimizer.csv / risk_jitter_simulator.csv
"""

from typing import Dict, List, Optional, Iterable, Union
import os
import pandas as pd

# ---------- Utilities ----------

def _to_float(x):
    try:
        if x is None: return None
        s = str(x).strip()
        if s == "": return None
        return float(s)
    except Exception:
        return None

def _ensure_dir(d: Optional[str]) -> str:
    if not d:
        return "."
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d

# ---------- Risk table (shared) ----------

RISK_NUM_COLS = [
    "proj","floor","ceiling","consistency","upside","duds",
    "sigma_base","sigma_eff","r_plus","r_minus","sigma_shrink_factor",
]

RISK_DISPLAY_ORDER = [
    "name","pos","team",
    "proj","floor","ceiling",
    "consistency","upside","duds",
    "sigma_base","sigma_eff","sigma_shrink_factor",
    "r_plus","r_minus",
]

# ---------- Jitter table (optimizer) ----------

JITTER_DISPLAY_ORDER = [
    "name","pos","team","proj",
    "sigma_base","sigma_eff","r_plus","r_minus",
    "nudged_up","nudged_down","nudged_total",
    "nudged_up_selected","nudged_up_not_selected",
    "nudged_down_selected","nudged_down_not_selected",
    "selected_total","selection_rate",
    "selected_when_up_rate","selected_when_down_rate",
    "avg_nudge_pts","avg_abs_nudge_pts",
    "avg_nudge_pts_selected","avg_abs_nudge_pts_selected",
]

class RiskAuditAccumulator:
    """
    Accumulates:
      - Risk rows (shared)
      - Optimizer jitter rows WITH selection impact

    Keys jitter by unique player_id to avoid name collisions.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir

        # Risk rows (simple list of dicts)
        self._risk_rows: List[Dict] = []

        # Jitter persistent store: player_id -> dict of aggregates
        self._jitter: Dict[Union[int,str], Dict] = {}

        # Per-iteration staging: player_id -> last noise (points)
        self._iter_noise: Dict[Union[int,str], float] = {}

    # ----- Risk rows -----

    def add_risk_row(self, *,
                     name: str,
                     pos: str,
                     team: str,
                     proj: float,
                     floor: Optional[float],
                     ceiling: Optional[float],
                     consistency: Optional[float],
                     upside: Optional[float],
                     duds: Optional[float],
                     sigma_base: float,
                     sigma_eff: float,
                     r_plus: float,
                     r_minus: float) -> None:
        self._risk_rows.append({
            "name": name,
            "pos": (pos or "").upper(),
            "team": team,
            "proj": _to_float(proj),
            "floor": _to_float(floor),
            "ceiling": _to_float(ceiling),
            "consistency": _to_float(consistency),  # may be 0..1 or 0..100; shown as-is
            "upside": _to_float(upside),
            "duds": _to_float(duds),
            "sigma_base": _to_float(sigma_base),
            "sigma_eff": _to_float(sigma_eff),
            "r_plus": _to_float(r_plus),
            "r_minus": _to_float(r_minus),
            "sigma_shrink_factor": (
                (_to_float(sigma_eff) / _to_float(sigma_base))
                if (_to_float(sigma_base) and _to_float(sigma_eff)) else None
            ),
        })

    def build_risk_table(self) -> pd.DataFrame:
        df = pd.DataFrame(self._risk_rows)
        if df.empty:
            return pd.DataFrame(columns=RISK_DISPLAY_ORDER)
        for c in RISK_NUM_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ["name","pos","team"]:
            if c not in df.columns:
                df[c] = ""
        df = df.sort_values(["pos","team","name"], kind="stable").reset_index(drop=True)
        cols = [c for c in RISK_DISPLAY_ORDER if c in df.columns] + [c for c in df.columns if c not in RISK_DISPLAY_ORDER]
        return df[cols]

    def save_risk_table(self, filename: str) -> Optional[str]:
        df = self.build_risk_table()
        if df.empty: return None
        out_dir = _ensure_dir(self.output_dir)
        path = os.path.join(out_dir, filename)
        try:
            df.to_csv(path, index=False)
            return path
        except Exception:
            return None

    # ----- Optimizer jitter with selection -----

    def start_iteration(self) -> None:
        """Call at the start of a lineup build iteration (before drawing jitter)."""
        self._iter_noise = {}

    def record_jitter_sample(self, *,
                             player_id: Union[int,str],
                             name: str,
                             pos: str,
                             team: str,
                             proj: Optional[float],
                             sigma_base: Optional[float],
                             sigma_eff: Optional[float],
                             r_plus: Optional[float],
                             r_minus: Optional[float],
                             noise_points: float) -> None:
        """
        Record the jitter draw for this iteration. Selection impact is joined later.
        """
        pid = player_id
        # Stage noise so we can join with selection after solve:
        self._iter_noise[pid] = float(noise_points)

        # Ensure persistent entry exists and carries reference fields
        ent = self._jitter.get(pid)
        if ent is None:
            ent = {
                "player_id": pid,
                "name": str(name),
                "pos": (pos or "").upper(),
                "team": str(team),
                "proj": _to_float(proj),
                "sigma_base": _to_float(sigma_base),
                "sigma_eff": _to_float(sigma_eff),
                "r_plus": _to_float(r_plus),
                "r_minus": _to_float(r_minus),

                # counts
                "nudged_up": 0,
                "nudged_down": 0,
                "nudged_total": 0,
                "nudged_up_selected": 0,
                "nudged_up_not_selected": 0,
                "nudged_down_selected": 0,
                "nudged_down_not_selected": 0,
                "selected_total": 0,

                # sums for averages
                "sum_nudge": 0.0,
                "sum_abs_nudge": 0.0,
                "sum_nudge_selected": 0.0,
                "sum_abs_nudge_selected": 0.0,
            }
            self._jitter[pid] = ent

    def finalize_iteration(self, selected_player_ids: Iterable[Union[int,str]]) -> None:
        """
        Call AFTER solving one lineup. Joins staged noise with selection.
        `selected_player_ids` is the set/list of IDs included in the solved lineup.
        """
        sel = set(selected_player_ids or [])
        for pid, noise in self._iter_noise.items():
            ent = self._jitter.get(pid)
            if ent is None:
                # Shouldn't happen; guard anyway
                continue

            is_up = noise >= 0.0
            was_selected = pid in sel

            # Counts
            if is_up:
                ent["nudged_up"] += 1
                if was_selected:
                    ent["nudged_up_selected"] += 1
                else:
                    ent["nudged_up_not_selected"] += 1
            else:
                ent["nudged_down"] += 1
                if was_selected:
                    ent["nudged_down_selected"] += 1
                else:
                    ent["nudged_down_not_selected"] += 1

            ent["nudged_total"] += 1
            if was_selected:
                ent["selected_total"] += 1

            # Sums for averages
            ent["sum_nudge"] += float(noise)
            ent["sum_abs_nudge"] += abs(float(noise))
            if was_selected:
                ent["sum_nudge_selected"] += float(noise)
                ent["sum_abs_nudge_selected"] += abs(float(noise))

        # Clear staging for next iteration
        self._iter_noise = {}

    def build_jitter_table(self) -> pd.DataFrame:
        if not self._jitter:
            return pd.DataFrame(columns=JITTER_DISPLAY_ORDER)
        df = pd.DataFrame(list(self._jitter.values()))
        if df.empty:
            return pd.DataFrame(columns=JITTER_DISPLAY_ORDER)

        # Averages & rates
        with pd.option_context('mode.use_inf_as_na', True):
            tot = df["nudged_total"].replace({0: pd.NA})
            up  = df["nudged_up"].replace({0: pd.NA})
            dn  = df["nudged_down"].replace({0: pd.NA})

            df["avg_nudge_pts"] = df["sum_nudge"] / tot
            df["avg_abs_nudge_pts"] = df["sum_abs_nudge"] / tot

            df["avg_nudge_pts_selected"] = df["sum_nudge_selected"] / df["selected_total"].replace({0: pd.NA})
            df["avg_abs_nudge_pts_selected"] = df["sum_abs_nudge_selected"] / df["selected_total"].replace({0: pd.NA})

            df["selection_rate"] = df["selected_total"] / df["nudged_total"].replace({0: pd.NA})
            df["selected_when_up_rate"] = df["nudged_up_selected"] / up
            df["selected_when_down_rate"] = df["nudged_down_selected"] / dn

        # Cleanup & order
        drop_cols = ["sum_nudge","sum_abs_nudge","sum_nudge_selected","sum_abs_nudge_selected","player_id"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        df = df.sort_values(["pos","team","name"], kind="stable").reset_index(drop=True)
        cols = [c for c in JITTER_DISPLAY_ORDER if c in df.columns] + [c for c in df.columns if c not in JITTER_DISPLAY_ORDER]
        return df[cols]

    def save_jitter_table(self, filename: str) -> Optional[str]:
        df = self.build_jitter_table()
        if df.empty: return None
        out_dir = _ensure_dir(self.output_dir)
        path = os.path.join(out_dir, filename)
        try:
            df.to_csv(path, index=False)
            return path
        except Exception:
            return None
