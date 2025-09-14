import csv
import ast as _ast

# --- begin: position canonicalization helpers ---
_CANON_PRIMARY = ("QB", "RB", "WR", "TE", "DST")
_SYNONYMS = {"D": "DST", "DEF": "DST", "DS": "DST", "D/ST": "DST", "DST": "DST", "W": "WR", "R": "RB", "Q": "QB", "T": "TE"}
_DROP = {"FLEX", "UTIL", "BN", "BENCH", "NA", "IR", "SLOT", "OP"}

def _canon_pos_any(v):
    """Return canonical position QB/RB/WR/TE/DST from various inputs."""
    if v is None:
        return ""
    # Already a clean string?
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("[") and s.endswith("]")) or ("," in s) or ("/" in s):
            try:
                lv = _ast.literal_eval(s)
                if not isinstance(lv, (list, tuple)):
                    raise ValueError
                toks = [str(x).strip().upper() for x in lv]
            except Exception:
                s2 = s.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
                s2 = s2.replace("/", ",")
                toks = [t.strip().upper() for t in s2.split(",") if t.strip()]
        else:
            toks = [s.upper()]
    else:
        try:
            toks = [str(x).strip().upper() for x in v]
        except Exception:
            toks = [str(v).strip().upper()]

    norm = []
    for t in toks:
        t = _SYNONYMS.get(t, t)
        if t in _DROP:
            continue
        norm.append(t)

    for p in _CANON_PRIMARY:
        if p in norm:
            return p
    return norm[0] if norm else ""


def _canon_pos_primary(v):
    s = _canon_pos_any(v)
    if s in _CANON_PRIMARY:
        return s
    return _SYNONYMS.get(s, '') if s else ''
# --- end: position canonicalization helpers ---

def _scalar(x):
    try:
        return float(x.item())
    except Exception:
        try:
            return float(x[0])
        except Exception:
            return float(x)

def _sf(x, default=0.0):
    try:
        if x is None: return float(default)
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none"): return float(default)
        return float(s)
    except Exception:
        return float(default)
from player_ids_flex import load_player_ids_flex, dst_id_by_team, _norm_name
import json
import math
import os
import random
import time
import numpy as np
import pulp as plp
import multiprocessing as mp
from risk_model import build_player_risk, build_covariance_from_corr, sample_skewed_outcomes
from risk_audit import RiskAuditAccumulator
import pandas as pd
import statistics

# import fuzzywuzzy
import itertools
import collections
import re
from scipy.stats import norm, kendalltau, gamma
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from numba import jit
import datetime

from utils import get_data_path, get_config_path
from stack_metrics import analyze_lineup
from selection_exposures import select_lineups, report_lineup_exposures

@jit(nopython=True)
def salary_boost(salary, max_salary):
    return (salary / max_salary) ** 2


def salary_boost(salary, max_salary):
    # Linear boost
    # return salary / max_salary

    # Non-linear boost (can adjust exponent for more/less emphasis)
    return (salary / max_salary) ** 2


class NFL_GPP_Simulator:
    position_map = {
        0: ["DST"],
        1: ["QB"],
        2: ["RB"],
        3: ["RB"],
        4: ["WR"],
        5: ["WR"],
        6: ["WR"],
        7: ["TE"],
        8: ["RB", "WR", "TE"],
    }

    # --- begin: robust correlation getter ---
    def get_corr_value(self, p_i, p_j, position_correlations=None):
        """
        Return correlation between two players based on their primary positions.
        Works with either a flat dict (pos->val) or nested dict (pos->pos->val).
        Defaults to 0.0 if missing.
        """
        try:
            pos_i = getattr(p_i, "pos", None) or getattr(p_i, "Position", None) or (p_i.get("Position") if isinstance(p_i, dict) else None)
            pos_j = getattr(p_j, "pos", None) or getattr(p_j, "Position", None) or (p_j.get("Position") if isinstance(p_j, dict) else None)
        except Exception:
            pos_i = pos_j = None
        pi = _canon_pos_primary(pos_i)
        pj = _canon_pos_primary(pos_j)
        if not pi or not pj:
            return 0.0

        if position_correlations is None:
            try:
                position_correlations = self.position_correlations
            except Exception:
                position_correlations = {}

        try:
            val = position_correlations[pi][pj]
        except Exception:
            try:
                val = position_correlations[pi]
            except Exception:
                val = 0.0
        try:
            return float(val)
        except Exception:
            return 0.0
    # --- end: robust correlation getter ---

    def __init__(
        self,
        site,
        field_size,
        num_iterations,
        use_contest_data,
        use_lineup_input,
        profile=None,
        pool_factor=5.0,
    ):
        # Instance attributes
        self.config = None
        self.player_dict = {}
        self.field_lineups = {}
        self.stacks_dict = {}
        self.gen_lineup_list = []
        self.roster_construction = []
        self.game_info = {}
        self.id_name_dict = {}
        self.salary = None
        self.optimal_score = None
        self.field_size = None
        self.team_list = []
        self.num_iterations = None
        self.site = site
        self.payout_structure = {}
        self.use_contest_data = False
        self.entry_fee = None
        self.use_lineup_input = use_lineup_input
        self.matchups = set()
        self.projection_minimum = 15
        self.randomness_amount = 100
        self.min_lineup_salary = 48000
        self.max_pct_off_optimal = 0.4
        self.teams_dict = collections.defaultdict(list)
        self.correlation_rules = {}
        self.seen_lineups = {}
        self.seen_lineups_ix = {}
        self.use_te_stack = True
        self.require_bring_back = True
        self.profile = profile
        self.pool_factor = pool_factor
        self.targets = {}
        self.stack_exposure_df = None

        self.output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
        self._audit = RiskAuditAccumulator(output_dir=self.output_dir)
        self.risk_table_df = None
        self._players_by_id = {}

        self.load_config()
        self.load_rules()
        if self.profile and "profiles" in self.config:
            profile_cfg = self.config["profiles"].get(self.profile)
            if profile_cfg:
                self.targets = {
                    "presence_targets_pct": profile_cfg.get("presence_targets_pct", {}),
                    "multiplicity_targets_mean": profile_cfg.get("multiplicity_targets_mean", {}),
                    "bucket_mix_pct": profile_cfg.get("bucket_mix_pct", {}),
                }
            else:
                print(f"Warning: profile {self.profile} not found in config")

        projection_path = get_data_path(site, self.config["projection_path"])
        self.load_projections(projection_path)


        # --- Begin: canonicalize positions in players_df ---
        try:
            if hasattr(self, "players_df") and self.players_df is not None:
                for col in ("pos", "Pos", "position", "Position"):
                    if col in self.players_df.columns:
                        self.players_df[col] = self.players_df[col].apply(_canon_pos_any)
                        if col != "Position":
                            self.players_df["Position"] = self.players_df[col]
                for tcol in ("team", "Team", "TeamAbbrev", "teamAbbrev"):
                    if tcol in self.players_df.columns:
                        self.players_df[tcol] = (
                            self.players_df[tcol].astype(str).str.upper().str.strip().replace({"LA": "LAR"})
                        )
                        self.players_df["TeamAbbrev"] = self.players_df[tcol]
        except Exception:
            pass
        # --- End: canonicalize positions in players_df ---
        player_path = get_data_path(site, self.config["player_path"])
        self.load_player_ids(player_path)

        # ownership_path = os.path.join(
        #    os.path.dirname(__file__),
        #    "../{}_data/{}".format(site, self.config["ownership_path"]),
        # )
        # self.load_ownership(ownership_path)

        # boom_bust_path = os.path.join(
        #    os.path.dirname(__file__),
        #    "../{}_data/{}".format(site, self.config["boom_bust_path"]),
        # )
        # self.load_boom_bust(boom_bust_path)

        #       batting_order_path = os.path.join(
        #           os.path.dirname(__file__),
        #            "../{}_data/{}".format(site, self.config["batting_order_path"]),
        #        )
        #        self.load_batting_order(batting_order_path)

        if site == "dk":
            self.roster_construction = [
                "QB",
                "RB",
                "RB",
                "WR",
                "WR",
                "WR",
                "TE",
                "FLEX",
                "DST",
            ]
            self.salary = 50000

        elif site == "fd":
            self.roster_construction = [
                "QB",
                "RB",
                "RB",
                "WR",
                "WR",
                "WR",
                "TE",
                "FLEX",
                "DST",
            ]
            self.salary = 60000

        self.use_contest_data = use_contest_data
        if use_contest_data:
            contest_path = get_data_path(site, self.config["contest_structure_path"])
            self.load_contest_data(contest_path)
            print("Contest payout structure loaded.")
        else:
            self.field_size = int(field_size)
            self.payout_structure = {0: 0.0}
            self.entry_fee = 0

        # self.adjust_default_stdev()
        self.assertPlayerDict()
        self.load_team_stacks()
        self.num_iterations = int(num_iterations)
        self.get_optimal()
        if self.use_lineup_input:
            self.load_lineups_from_file()
        # if self.match_lineup_input_to_field_size or len(self.field_lineups) == 0:
        # self.generate_field_lineups()
        self.load_correlation_rules()

    # make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    def load_rules(self):
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.min_lineup_salary = int(self.config["min_lineup_salary"])
        self.max_pct_off_optimal = float(self.config["max_pct_off_optimal"])
        self.pct_field_using_stacks = float(self.config["pct_field_using_stacks"])
        self.default_qb_var = float(self.config["default_qb_var"])
        self.default_skillpos_var = float(self.config["default_skillpos_var"])
        self.default_def_var = float(self.config["default_def_var"])
        self.overlap_limit = float(self.config["num_players_vs_def"])
        self.pct_field_double_stacks = float(self.config["pct_field_double_stacks"])
        self.correlation_rules = self.config["custom_correlations"]
        self.use_te_stack = bool(self.config.get("use_te_stack", True))
        self.require_bring_back = bool(self.config.get("require_bring_back", True))

    def assertPlayerDict(self):
        for p, s in list(self.player_dict.items()):
            if s["ID"] == 0 or s["ID"] == "" or s["ID"] is None:
                print(
                    s["Name"]
                    + " name mismatch between projections and player ids, excluding from player_dict"
                )
                self.player_dict.pop(p)

    # In order to make reasonable tournament lineups, we want to be close enough to the optimal that
    # a person could realistically land on this lineup. Skeleton here is taken from base `mlb_optimizer.py`
    def get_optimal(self):

        # --- Begin: canonicalize positions in player_dict & DST backfill ---
        try:
            # 1) Canonicalize Position / TeamAbbrev
            for _k, _rec in list(self.player_dict.items()):
                raw_pos = _rec.get("Position") or _rec.get("pos")
                _rec["Position"] = _canon_pos_any(raw_pos)
                if "TeamAbbrev" in _rec:
                    _rec["TeamAbbrev"] = str(_rec["TeamAbbrev"]).upper().strip()
                elif "Team" in _rec:
                    _rec["TeamAbbrev"] = str(_rec["Team"]).upper().strip()

            # 2) Backfill DST IDs strictly by canonical 'DST'
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
                        (_pid_df["Position"] == "DST")
                        & (_pid_df["TeamAbbrev"].astype(str).str.upper() == team)
                    ].iloc[0]
                    return int(row["ID"])
                except Exception:
                    return None

            for _k, _rec in list(self.player_dict.items()):
                if _rec.get("Position") == "DST":
                    _id_raw = _rec.get("ID", 0)
                    try:
                        _id_ok = int(str(_id_raw).replace(",", ""))
                    except Exception:
                        _id_ok = 0
                    if not _id_ok:
                        pid = _dst_id_by_team_lookup(_rec.get("TeamAbbrev"))
                        if pid:
                            _rec["ID"] = pid

            # 3) Post-normalization position counts for diagnostics
            pos_counts = {}
            for _rec in self.player_dict.values():
                p = str(_rec.get("Position", "")).upper()
                pos_counts[p] = pos_counts.get(p, 0) + 1
            if pos_counts.get("DST", 0) <= 0:
                raise AssertionError(
                    f"Simulator: no DST players after position canonicalization. POS counts={pos_counts}"
                )
        except Exception:
            raise
        # --- End: canonicalize positions in player_dict & DST backfill ---

        # --- Begin: clean numeric fields before building LP ---
        try:
            PROJ_KEYS = (
                "Projection",
                "projections_proj",
                "proj",
                "fpts_proj",
                "projected_points",
                "fpts",
                "points",
            )
            SAL_KEYS = ("Salary", "salary", "sal", "cost", "dk_salary")
            OWN_KEYS = ("own", "ownership", "proj_own", "ownership_proj")
            CEIL_KEYS = ("ceil", "ceiling", "fpts_ceil", "projection_ceil")
            STD_KEYS = ("stddev", "stdev", "sd", "projection_std", "fpts_std")

            for _k, _rec in list(self.player_dict.items()):
                _p = None
                for kk in PROJ_KEYS:
                    if kk in _rec and _rec[kk] not in (None, ""):
                        _p = _rec[kk]
                        break
                if _p is None and hasattr(self, "players_df") and "name" in self.players_df.columns:
                    try:
                        nm = str(_rec.get("Name", "")).strip().lower()
                        _row = self.players_df.loc[self.players_df["name"].str.lower() == nm]
                        if not _row.empty:
                            for alt in PROJ_KEYS:
                                if alt in _row.columns:
                                    _p = _row[alt].iloc[0]
                                    break
                    except Exception:
                        pass
                _rec["Projection"] = _sf(_p, 0.0)

                _s = None
                for kk in SAL_KEYS:
                    if kk in _rec and _rec[kk] not in (None, ""):
                        _s = _rec[kk]
                        break
                _rec["Salary"] = _sf(_s, 0.0)

                _o = None
                for kk in OWN_KEYS:
                    if kk in _rec and _rec[kk] not in (None, ""):
                        _o = _rec[kk]
                        break
                _rec["Own"] = _sf(_o, 0.0)

                _c = None
                for kk in CEIL_KEYS:
                    if kk in _rec and _rec[kk] not in (None, ""):
                        _c = _rec[kk]
                        break
                _rec["Ceil"] = _sf(_c, 0.0)

                _sd = None
                for kk in STD_KEYS:
                    if kk in _rec and _rec[kk] not in (None, ""):
                        _sd = _rec[kk]
                        break
                _rec["STDDEV"] = _sf(_sd, 0.0)

            bad = [
                (_k, _rec.get("Name", ""), _rec.get("Position", ""))
                for _k, _rec in self.player_dict.items()
                if not isinstance(_rec.get("Projection"), (int, float))
            ]
            if bad:
                raise AssertionError(
                    f"Simulator: non-numeric Projection for {len(bad)} players (e.g., {bad[:3]})."
                )
        except Exception as _e:
            raise
        # --- End: clean numeric fields before building LP ---
        # print(s['Name'],s['ID'])
        # print(self.player_dict)
        problem = plp.LpProblem("NFL", plp.LpMaximize)
        lp_variables = {
            self.player_dict[(player, pos_str, team)]["ID"]: plp.LpVariable(
                str(self.player_dict[(player, pos_str, team)]["ID"]), cat="Binary"
            )
            for (player, pos_str, team) in self.player_dict
        }

        # set the objective - maximize fpts
        problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["fieldFpts"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
            ),
            "Objective",
        )

        # Set the salary constraints
        problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["Salary"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
            )
            <= self.salary
        )

        if self.site == "dk":
            # Need 1 quarterback
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "QB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Need at least 2 RBs can have up to 3 with FLEX slot
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "RB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 2
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "RB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 3
            )
            # Need at least 3 WRs can have up to 4 with FLEX slot
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "WR" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 3
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "WR" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 4
            )
            # Need at least 1 TE
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 1
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 2
            )
            # Need 1 DEF
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "DST" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Can only roster 9 total players
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                )
                == 9
            )
            # Max 8 per team in case of weird issues with stacking on short slates
            for team in self.team_list:
                problem += (
                    plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in self.player_dict
                        if self.player_dict[(player, pos_str, team)]["Team"] == team
                    )
                    <= 8
                )

        elif self.site == "fd":
            # Need at least 1 point guard, can have up to 3 if utilizing G and UTIL slots
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "QB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Need at least 2 RBs can have up to 3 with FLEX slot
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "RB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 2
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "RB" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 3
            )
            # Need at least 3 WRs can have up to 4 with FLEX slot
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "WR" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 3
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "WR" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 4
            )
            # Need at least 1 TE
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 1
            )
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 2
            )
            # Need 1 DEF
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "DST" in self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1
            )
            # Can only roster 9 total players
            problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                )
                == 9
            )
            # Max 4 per team
            for team in self.team_list:
                problem += (
                    plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in self.player_dict
                        if self.player_dict[(player, pos_str, team)]["Team"] == team
                    )
                    <= 4
                )

        # print(f"Problem Name: {problem.name}")
        # print(f"Sense: {problem.sense}")

        # # Print the objective
        # print("\nObjective:")
        # try:
        #     for v, coef in problem.objective.items():
        #         print(f"{coef}*{v.name}", end=' + ')
        # except Exception as e:
        #     print(f"Error while printing objective: {e}")

        # # Print the constraints
        # print("\nConstraints:")
        # for constraint in problem.constraints.values():
        #     try:
        #         # Extract the left-hand side, right-hand side, and the operator
        #         lhs = "".join(f"{coef}*{var.name}" for var, coef in constraint.items())
        #         rhs = constraint.constant
        #         if constraint.sense == 1:
        #             op = ">="
        #         elif constraint.sense == -1:
        #             op = "<="
        #         else:
        #             op = "="
        #         print(f"{lhs} {op} {rhs}")
        #     except Exception as e:
        #         print(f"Error while printing constraint: {e}")

        # # Print the variables
        # print("\nVariables:")
        # try:
        #     for v in problem.variables():
        #         print(f"{v.name}: LowBound={v.lowBound}, UpBound={v.upBound}, Cat={v.cat}")
        # except Exception as e:
        #     print(f"Error while printing variable: {e}")
        # Crunch!
        try:
            problem.solve(plp.PULP_CBC_CMD(msg=0))
        except plp.PulpSolverError:
            print(
                "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                    len(self.num_lineups), self.num_lineups
                )
            )
        except TypeError:
            for p, s in self.player_dict.items():
                if s["ID"] == 0:
                    print(
                        s["Name"] + " name mismatch between projections and player ids"
                    )
                if s["ID"] == "":
                    print(
                        s["Name"] + " name mismatch between projections and player ids"
                    )
                if s["ID"] is None:
                    print(s["Name"])
        # Directly evaluate the objective using PuLP's value helper instead of
        # constructing a string and using ``eval``.  The previous approach would
        # fail when any variable had a ``None`` value, causing a ``TypeError``
        # during evaluation.  ``plp.value`` safely computes the objective value
        # from the solved problem.
        self.optimal_score = float(plp.value(problem.objective))

    @staticmethod
    def extract_matchup_time(game_string):
        # Extract the matchup, date, and time
        match = re.match(
            r"(\w{2,4}@\w{2,4}) (\d{2}/\d{2}/\d{4}) (\d{2}:\d{2}[APM]{2} ET)",
            game_string,
        )

        if match:
            matchup, date, time = match.groups()
            # Convert 12-hour time format to 24-hour format
            time_obj = datetime.datetime.strptime(time, "%I:%M%p ET")
            # Convert the date string to datetime.date
            date_obj = datetime.datetime.strptime(date, "%m/%d/%Y").date()
            # Combine date and time to get a full datetime object
            datetime_obj = datetime.datetime.combine(date_obj, time_obj.time())
            return matchup, datetime_obj
        return None

    # Load player IDs for exporting
    
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
            pos = _canon_pos_any(r["Position"])
            team = str(r.get("TeamAbbrev", "") or "").upper()

            self.id_position_dict[pid] = pos
            self.id_teamabbrev_dict[pid] = team
            self.id_name_dict[pid] = name
            canon = _norm_name(name)
            key_name = canon.replace("-", "#")
            self.name_pos_to_id[(key_name, pos)] = pid
            parts = canon.split()
            if len(parts) >= 2:
                alias = f"{parts[0][0]} {parts[-1]}".replace("-", "#")
                if (alias, pos) not in self.name_pos_to_id:
                    self.name_pos_to_id[(alias, pos)] = pid

        # Match IDs onto existing player_dict entries

        for key, rec in list(self.player_dict.items()):
            name_key = _norm_name(rec.get("Name", "")).replace("-", "#")
            pos = rec.get("Position")
            if isinstance(pos, list):
                pos = pos[0] if pos else ""
            pos = _canon_pos_any(pos)
            pid = self.name_pos_to_id.get((name_key, pos))
            if pid:
                rec["ID"] = pid
                if not rec.get("TeamAbbrev"):
                    rec["TeamAbbrev"] = self.id_teamabbrev_dict.get(pid, "")
            elif pos == "DST":
                team = str(rec.get("TeamAbbrev") or "").upper()
                if not team and isinstance(key, tuple) and len(key) >= 3:
                    team = str(key[2]).upper()
                pid_team = dst_id_by_team(self._player_ids_df, team)
                if pid_team:
                    rec["ID"] = str(pid_team)
        for rec in self.player_dict.values():
            pid = rec.get("ID")
            if pid:
                self._players_by_id[pid] = rec
        return df

    def load_contest_data(self, path):
        """Load contest metadata including payout structure.

        The contest structure CSV is expected to contain at least
        ``place`` and ``payout`` columns.  Additional rows may include
        metadata such as ``entries`` (field size) and ``entry_fee``.
        This function is resilient to slight format variations and
        ignores any lines that do not match the expected format.
        """

        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                place = row.get("place", "").strip().lower()
                payout = row.get("payout", "").strip()

                if place in {"entries", "field size"}:
                    try:
                        self.field_size = int(payout.replace(",", ""))
                    except ValueError:
                        pass
                    continue

                if place in {"entry fee", "entry_fee"}:
                    try:
                        self.entry_fee = float(
                            payout.replace("$", "").replace(",", "")
                        )
                    except ValueError:
                        pass
                    continue

                # Standard payout rows – ``place`` should be an integer
                try:
                    place_idx = int(place) - 1
                    prize = float(payout.split(".")[0].replace(",", ""))
                except ValueError:
                    # Skip rows that don't conform to expected format
                    continue

                self.payout_structure[place_idx] = prize

        # If field size wasn't provided, infer it from the payout structure
        if self.field_size is None:
            self.field_size = len(self.payout_structure)

    def load_correlation_rules(self):
        if len(self.correlation_rules.keys()) > 0:
            for primary_player in self.correlation_rules.keys():
                # Convert primary_player to the consistent format
                formatted_primary_player = (
                    _norm_name(primary_player).replace("-", "#")
                )
                for (
                    player_name,
                    pos_str,
                    team,
                ), player_data in self.player_dict.items():
                    if formatted_primary_player == player_name:
                        for second_entity, correlation_value in self.correlation_rules[
                            primary_player
                        ].items():
                            # Convert second_entity to the consistent format
                            formatted_second_entity = (
                                _norm_name(second_entity).replace("-", "#")
                            )

                            # Check if the formatted_second_entity is a player name
                            found_second_entity = False
                            for (
                                se_name,
                                se_pos_str,
                                se_team,
                            ), se_data in self.player_dict.items():
                                if formatted_second_entity == se_name:
                                    player_data["Player Correlations"][
                                        formatted_second_entity
                                    ] = correlation_value
                                    se_data["Player Correlations"][
                                        formatted_primary_player
                                    ] = correlation_value
                                    found_second_entity = True
                                    break

                            # If the second_entity is not found as a player, assume it's a position and update 'Correlations'
                            if not found_second_entity:
                                player_data["Correlations"][
                                    second_entity
                                ] = correlation_value

    # Load config from file
    def load_config(self):
        config_path = get_config_path()
        with open(config_path, encoding="utf-8-sig") as json_file:
            self.config = json.load(json_file)

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for idx, row in enumerate(reader):
                player_name = _norm_name(row["name"]).replace("-", "#")
                try:
                    fpts = float(row["projections_proj"])
                except:
                    fpts = 0
                    print(
                        "unable to load player fpts: "
                        + player_name
                        + ", fpts:"
                        + row["projections_proj"]
                    )
                actpts = (
                    float(row.get("projections_actpts", 0))
                    if row.get("projections_actpts", "") not in ["", None]
                    else 0
                )
                fieldFpts = fpts
                position = [pos for pos in row["pos"].split("/")]
                position.sort()
                # if qb and dst not in position add flex
                if self.site == "fd":
                    if "D" in position:
                        position = ["DST"]
                if "QB" not in position and "DST" not in position:
                    position.append("FLEX")
                pos = position[0]
                risk = build_player_risk(
                    mean_fpts=fpts,
                    position=pos,
                    default_qb_var=self.default_qb_var,
                    default_skill_var=self.default_skillpos_var,
                    default_def_var=self.default_def_var,
                    ceiling=(row.get("projections_ceiling") or row.get("ceiling")),
                    floor=(row.get("projections_floor") or row.get("floor")),
                    consistency_raw=row.get("fantasyyear_consistency"),
                    upside_raw=row.get("fantasyyear_upside"),
                    duds_raw=row.get("fantasyyear_duds"),
                    beta_consistency=1.0,
                )
                stddev = risk["sigma_eff"]
                sigma_base = risk["sigma_base"]
                r_plus = risk["r_plus"]
                r_minus = risk["r_minus"]
                ceil = float(
                    row.get("projections_ceiling")
                    or row.get("ceiling")
                    or (fpts + stddev)
                )
                if os.environ.get("RISK_DEBUG", "0") == "1" and idx < 3:
                    print(
                        "RISK:",
                        row.get("name"),
                        pos,
                        "mean=",
                        fpts,
                        "σ_base=",
                        round(sigma_base, 2),
                        "σ_eff=",
                        round(stddev, 2),
                        "r+=",
                        round(r_plus, 2),
                        "r-=",
                        round(r_minus, 2),
                        "cons=",
                        row.get("fantasyyear_consistency"),
                        "up=",
                        row.get("fantasyyear_upside"),
                        "dn=",
                        row.get("fantasyyear_duds"),
                    )
                if row["salary"]:
                    sal = int(row["salary"].replace(",", ""))
                if pos == "QB":
                    corr = {
                        "QB": 1,
                        "RB": 0.08,
                        "WR": 0.62,
                        "TE": 0.32,
                        "DST": -0.09,
                        "Opp QB": 0.24,
                        "Opp RB": 0.04,
                        "Opp WR": 0.19,
                        "Opp TE": 0.1,
                        "Opp DST": -0.41,
                    }
                elif pos == "RB":
                    corr = {
                        "QB": 0.08,
                        "RB": 1,
                        "WR": -0.09,
                        "TE": -0.02,
                        "DST": 0.07,
                        "Opp QB": 0.04,
                        "Opp RB": -0.08,
                        "Opp WR": 0.01,
                        "Opp TE": 0.03,
                        "Opp DST": -0.33,
                    }
                elif pos == "WR":
                    corr = {
                        "QB": 0.62,
                        "RB": -0.09,
                        "WR": 1,
                        "TE": -0.07,
                        "DST": -0.08,
                        "Opp QB": 0.19,
                        "Opp RB": 0.01,
                        "Opp WR": 0.16,
                        "Opp TE": 0.08,
                        "Opp DST": -0.22,
                    }
                elif pos == "TE":
                    corr = {
                        "QB": 0.32,
                        "RB": -0.02,
                        "WR": -0.07,
                        "TE": 1,
                        "DST": -0.08,
                        "Opp QB": 0.1,
                        "Opp RB": 0.03,
                        "Opp WR": 0.08,
                        "Opp TE": 0,
                        "Opp DST": -0.14,
                    }
                elif pos == "DST":
                    corr = {
                        "QB": -0.09,
                        "RB": 0.07,
                        "WR": -0.08,
                        "TE": -0.08,
                        "DST": 1,
                        "Opp QB": -0.41,
                        "Opp RB": -0.33,
                        "Opp WR": -0.22,
                        "Opp TE": -0.14,
                        "Opp DST": -0.27,
                    }
                team = row["team"]
                opp = row.get("opp", "")

                if team == "LA":
                    team = "LAR"

                if self.site == "fd":
                    if team == "JAX":
                        team = "JAC"
                    if opp == "JAX":
                        opp = "JAC"

                self._audit.add_risk_row(
                    name=str(row.get("name") or row.get("Name") or player_name),
                    pos=pos,
                    team=team,
                    proj=fpts,
                    floor=(row.get("projections_floor") or row.get("floor")),
                    ceiling=(row.get("projections_ceiling") or row.get("ceiling")),
                    consistency=row.get("fantasyyear_consistency"),
                    upside=row.get("fantasyyear_upside"),
                    duds=row.get("fantasyyear_duds"),
                    sigma_base=sigma_base,
                    sigma_eff=stddev,
                    r_plus=r_plus,
                    r_minus=r_minus,
                )

                # Build matchup identifiers that are consistent for both teams
                if opp:
                    matchup_teams = tuple(sorted([team, opp]))
                    matchup = "@".join(matchup_teams)
                else:
                    matchup_teams = tuple()
                    matchup = ""

                own = float(row["projections_projown"]) if row["projections_projown"] != "" else 0
                if own == 0:
                    own = 0.1
                pos_str = str(position)
                player_data = {
                    "Fpts": fpts,
                    "ActPts": actpts,
                    "fieldFpts": fieldFpts,
                    "Position": position,
                    "Name": player_name,
                    "Team": team,
                    "Opp": opp,
                    "ID": "",
                    "Matchup": matchup,
                    "Salary": int(row["salary"].replace(",", "")),
                    "StdDev": stddev,
                    "SigmaBase": sigma_base,
                    "RPlus": r_plus,
                    "RMinus": r_minus,
                    "Ceiling": ceil,
                    "Ownership": own,
                    "Correlations": corr,
                    "Player Correlations": {},
                    "In Lineup": False,
                }

                # Retain existing info if player already present
                key = (player_name, pos_str, team)
                if key in self.player_dict:
                    player_data["ID"] = self.player_dict[key].get("ID", "")
                    if not player_data.get("Matchup"):
                        player_data["Matchup"] = self.player_dict[key].get("Matchup", matchup)

                self.player_dict[key] = player_data
                self.teams_dict[team].append(player_data)  # Add player data to their respective team
                if matchup_teams:
                    self.matchups.add(matchup_teams)

    def load_team_stacks(self):
        # Initialize a dictionary to hold QB ownership by team
        qb_ownership_by_team = {}
        # Reset stacks_dict to ensure it reflects the current player pool
        self.stacks_dict = {}

        for p in self.player_dict:
            # Check if player is a QB
            if "QB" in self.player_dict[p]["Position"]:
                # Fetch the team of the QB
                team = self.player_dict[p]["Team"]

                # Convert the ownership percentage string to a float and divide by 100
                own_percentage = float(self.player_dict[p]["Ownership"]) / 100

                # Add the ownership to the accumulated ownership for the team
                if team in qb_ownership_by_team:
                    qb_ownership_by_team[team] += own_percentage
                else:
                    qb_ownership_by_team[team] = own_percentage

        # Now, update the stacks_dict with the QB ownership by team
        for team, own_percentage in qb_ownership_by_team.items():
            self.stacks_dict[team] = own_percentage

    def extract_id(self, cell_value):
        if "(" in cell_value and ")" in cell_value:
            return cell_value.split("(")[1].replace(")", "")
        elif ":" in cell_value:
            return cell_value.split(":")[1]
        else:
            return cell_value

    def load_lineups_from_file(self):
        print("loading lineups")
        i = 0
        path = get_data_path(self.site, "tournament_lineups.csv")
        with open(path) as file:
            reader = pd.read_csv(file)
            lineup = []
            j = 0
            for i, row in reader.iterrows():
                # print(row)
                if i == self.field_size:
                    break
                lineup = [self.extract_id(str(row[j])) for j in range(9)]
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                for l in lineup:
                    ids = [self.player_dict[k]["ID"] for k in self.player_dict]
                    if l not in ids:
                        print("lineup {} is missing players {}".format(i, l))
                        if l in self.id_name_dict:
                            print(self.id_name_dict[l])
                        error = True
                if len(lineup) < 9:
                    print("lineup {} is missing players".format(i))
                    continue
                # storing if this lineup was made by an optimizer or with the generation process in this script
                error = False
                for l in lineup:
                    ids = [self.player_dict[k]["ID"] for k in self.player_dict]
                    if l not in ids:
                        print("lineup {} is missing players {}".format(i, l))
                        if l in self.id_name_dict:
                            print(self.id_name_dict[l])
                        error = True
                if len(lineup) < 9:
                    print("lineup {} is missing players".format(i))
                    continue
                if not error:
                    # reshuffle lineup to match temp_roster_construction
                    temp_roster_construction = [
                        "DST",
                        "QB",
                        "RB",
                        "RB",
                        "WR",
                        "WR",
                        "WR",
                        "TE",
                        "FLEX",
                    ]
                    shuffled_lu = []

                    id_to_player_dict = {
                        v["ID"]: v for k, v in self.player_dict.items()
                    }
                    lineup_copy = lineup.copy()
                    position_counts = {
                        "DST": 0,
                        "QB": 0,
                        "RB": 0,
                        "WR": 0,
                        "TE": 0,
                        "FLEX": 0,
                    }
                    z = 0

                    # Rebuild DK slots deterministically with canonical positions
                    id_to_player_dict = {v.get("ID"): v for v in self.player_dict.values()}
                    # slot map we will fill
                    slot_names = {"QB": None, "RB1": None, "RB2": None, "WR1": None, "WR2": None, "WR3": None, "TE": None, "FLEX": None, "DST": None}
                    # Collect (id, pos)
                    details = []
                    for l in list(lineup):
                        rec = id_to_player_dict.get(l)
                        if not rec:
                            # Missing ID in dictionary; skip and continue safely
                            continue
                        pos = _canon_pos_primary(rec.get("Position"))
                        details.append((l, pos))
                    # First pass: primaries
                    flex_pool = []
                    for pid, pos in details:
                        if pos == "QB" and slot_names["QB"] is None:
                            slot_names["QB"] = pid
                        elif pos == "DST" and slot_names["DST"] is None:
                            slot_names["DST"] = pid
                        elif pos == "TE" and slot_names["TE"] is None:
                            slot_names["TE"] = pid
                        elif pos == "RB":
                            if slot_names["RB1"] is None:
                                slot_names["RB1"] = pid
                            elif slot_names["RB2"] is None:
                                slot_names["RB2"] = pid
                            else:
                                flex_pool.append(pid)
                        elif pos == "WR":
                            if slot_names["WR1"] is None:
                                slot_names["WR1"] = pid
                            elif slot_names["WR2"] is None:
                                slot_names["WR2"] = pid
                            elif slot_names["WR3"] is None:
                                slot_names["WR3"] = pid
                            else:
                                flex_pool.append(pid)
                        else:
                            # Non-skill (or unknown) does not go to FLEX
                            pass
                    # FLEX: RB/WR/TE only
                    if slot_names["FLEX"] is None:
                        # include TE to flex_pool if it overflowed primaries
                        if slot_names["TE"] is not None:
                            # already assigned primary TE; extras would have been in flex_pool above
                            pass
                        if flex_pool:
                            slot_names["FLEX"] = flex_pool[0]
                    # If any core slot is still missing, bail out gracefully to avoid hanging
                    core_slots = ["QB","RB1","RB2","WR1","WR2","WR3","TE","DST"]
                    if any(slot_names[s] is None for s in core_slots):
                        # skip malformed saved lineup
                        continue
                    # Construct shuffled_lu matching DK order
                    shuffled_lu = [
                        slot_names["QB"], slot_names["RB1"], slot_names["RB2"],
                        slot_names["WR1"], slot_names["WR2"], slot_names["WR3"],
                        slot_names["TE"], slot_names["FLEX"], slot_names["DST"],
                    ]
                    lineup_copy = []  # no longer needed; keep variable around for later code
                    position_counts = {}  # not used with deterministic mapping

                    lineup_list = sorted(shuffled_lu)           
                    lineup_set = frozenset(lineup_list)

                    # Keeping track of lineup duplication counts
                    if lineup_set in self.seen_lineups:
                        self.seen_lineups[lineup_set] += 1
                    else:
                        # Add to seen_lineups and seen_lineups_ix
                        self.seen_lineups[lineup_set] = 1
                        self.seen_lineups_ix[lineup_set] = j
                        self.field_lineups[j] = {
                            "Lineup": shuffled_lu,
                            "Wins": 0,
                            "Top1Percent": 0,
                            "ROI": 0,
                            "Cashes": 0,
                            "Type": "opto",
                            "Count" : 1
                        }
                        j += 1
        print("loaded {} lineups".format(j))
        #print(len(self.field_lineups))

    @staticmethod
    def generate_lineups(
        lu_num,
        ids,
        in_lineup,
        pos_matrix,
        ownership,
        salary_floor,
        salary_ceiling,
        optimal_score,
        salaries,
        projections,
        max_pct_off_optimal,
        teams,
        opponents,
        team_stack,
        stack_len,
        overlap_limit,
        max_stack_len,
        matchups,
        num_players_in_roster,
        site,
        use_te_stack,
        require_bring_back,
    ):
        # new random seed for each lineup (without this there is a ton of dupes)
        rng = np.random.Generator(np.random.PCG64())
        lus = {}
        # make sure nobody is already showing up in a lineup
        if sum(in_lineup) != 0:
            in_lineup.fill(0)
        reject = True
        iteration_count = 0
        max_iterations = 5000
        total_players = num_players_in_roster
        issue = ""
        complete = ""
        if optimal_score is None:
            reasonable_projection = 0
            reasonable_stack_projection = 0
        else:
            reasonable_projection = optimal_score - (
                max_pct_off_optimal * optimal_score
            )
            reasonable_stack_projection = optimal_score - (
                (max_pct_off_optimal * 1.25) * optimal_score
            )
        max_players_per_team = 4 if site == "fd" else None
        # reject_counters = {
        #     "salary_too_low": 0,
        #     "salary_too_high": 0,
        #     "projection_too_low": 0,
        #     "invalid_matchups": 0,
        #     "stack_length_insufficient": 0,
        # }
        # print(lu_num, ' started',  team_stack, max_stack_len)
        while reject:
            iteration_count += 1
            if iteration_count > max_iterations:
                print(
                    f"Failed to generate lineup after {max_iterations} iterations"
                )
                # Returning ``None`` allows the caller to skip this lineup
                # without the entire simulation aborting due to an exception.
                return None
            if team_stack == "":
                salary = 0
                proj = 0
                if sum(in_lineup) != 0:
                    in_lineup.fill(0)
                lineup = []
                player_teams = []
                def_opps = []
                players_opposing_def = 0
                lineup_matchups = []
                k = 0
                for pos in pos_matrix.T:
                    if k < 1:
                        # check for players eligible for the position and make sure they arent in a lineup, returns a list of indices of available player
                        valid_players = np.nonzero((pos > 0) & (in_lineup == 0))[0]
                        # grab names of players eligible
                        plyr_list = ids[valid_players]
                        # create np array of probability of being seelcted based on ownership and who is eligible at the position
                        prob_list = ownership[valid_players]
                        prob_list = prob_list / prob_list.sum()
                        try:
                            choice = rng.choice(plyr_list, p=prob_list)
                        except:
                            print(plyr_list, prob_list)
                            print("find failed on nonstack and first player selection")
                        # --- begin: lineup generation fallback tuning ---
                        # If we repeatedly fail to select the first/non-stack player, relax the strictness a bit.
                        # This prevents 5000-iteration timeouts on small or multi-eligible pools.
                        if "fail_counter" not in locals():
                            fail_counter = 0
                        fail_counter += 1
                        if fail_counter in (500, 1000, 2000, 3000, 4000):
                            try:
                                # widen pool: allow any skill position as seed (WR/RB/TE), keep QB/DST out
                                widened = [p for p in players if _canon_pos_primary(getattr(p, "pos", None) or getattr(p, "Position", None)) in ("WR","RB","TE")]
                                if widened:
                                    first_player_pool = widened
                                # as a last resort after 3000 attempts, allow stack leniency by one
                                if fail_counter >= 3000 and hasattr(self, "stack_rules"):
                                    self.stack_rules = max(0, int(self.stack_rules) - 1)
                            except Exception:
                                pass
                        # --- end: lineup generation fallback tuning ---

                        choice_idx = np.nonzero(ids == choice)[0]
                        lineup.append(str(choice))
                        in_lineup[choice_idx] = 1
                        salary += _scalar(salaries[choice_idx])
                        proj += _scalar(projections[choice_idx])
                        def_opp = opponents[choice_idx][0]
                        lineup_matchups.append(matchups[choice_idx[0]])
                        player_teams.append(teams[choice_idx][0])
                    if k >= 1:
                        remaining_salary = salary_ceiling - salary
                        if players_opposing_def < overlap_limit:
                            if k == total_players - 1:
                                valid_players = np.nonzero(
                                    (pos > 0)
                                    & (in_lineup == 0)
                                    & (salaries <= remaining_salary)
                                    & (salary + salaries >= salary_floor)
                                )[0]
                            else:
                                valid_players = np.nonzero(
                                    (pos > 0)
                                    & (in_lineup == 0)
                                    & (salaries <= remaining_salary)
                                )[0]
                            # grab names of players eligible
                            plyr_list = ids[valid_players]
                            # create np array of probability of being seelcted based on ownership and who is eligible at the position
                            prob_list = ownership[valid_players]
                            prob_list = prob_list / prob_list.sum()
                            if k == total_players - 1:
                                boosted_salaries = np.array(
                                    [
                                        salary_boost(s, salary_ceiling)
                                        for s in salaries[valid_players]
                                    ]
                                )
                                boosted_probabilities = prob_list * boosted_salaries
                                boosted_probabilities /= (
                                    boosted_probabilities.sum()
                                )  # normalize to ensure it sums to 1
                            try:
                                if k == total_players - 1:
                                    choice = rng.choice(
                                        plyr_list, p=boosted_probabilities
                                    )
                                else:
                                    choice = rng.choice(plyr_list, p=prob_list)
                            except:
                                # if remaining_salary <= np.min(salaries):
                                #     reject_counters["salary_too_high"] += 1
                                # else:
                                #     reject_counters["salary_too_low"]
                                salary = 0
                                proj = 0
                                if team_stack == "":
                                    lineup = []
                                else:
                                    lineup = np.zeros(shape=pos_matrix.shape[1]).astype(
                                        str
                                    )
                                player_teams = []
                                def_opps = []
                                players_opposing_def = 0
                                lineup_matchups = []
                                in_lineup.fill(0)  # Reset the in_lineup array
                                k = 0  # Reset the player index
                                continue  # Skip to the next iteration of the while loop
                            choice_idx = np.nonzero(ids == choice)[0]
                            lineup.append(str(choice))
                            in_lineup[choice_idx] = 1
                            salary += _scalar(salaries[choice_idx])
                            proj += _scalar(projections[choice_idx])
                            player_teams.append(teams[choice_idx][0])
                            lineup_matchups.append(matchups[choice_idx[0]])
                            if teams[choice_idx][0] == def_opp:
                                players_opposing_def += 1
                            if max_players_per_team is not None:
                                team_count = Counter(player_teams)
                                if any(
                                    count > max_players_per_team
                                    for count in team_count.values()
                                ):
                                    salary = 0
                                    proj = 0
                                    if team_stack == "":
                                        lineup = []
                                    else:
                                        lineup = np.zeros(
                                            shape=pos_matrix.shape[1]
                                        ).astype(str)
                                    player_teams = []
                                    def_opps = []
                                    players_opposing_def = 0
                                    lineup_matchups = []
                                    in_lineup.fill(0)  # Reset the in_lineup array
                                    k = 0  # Reset the player index
                                    continue  # Skip to the next iteration of the while loop
                        else:
                            if k == total_players - 1:
                                valid_players = np.nonzero(
                                    (pos > 0)
                                    & (in_lineup == 0)
                                    & (salaries <= remaining_salary)
                                    & (salary + salaries >= salary_floor)
                                    & (teams != def_opp)
                                )[0]
                            else:
                                valid_players = np.nonzero(
                                    (pos > 0)
                                    & (in_lineup == 0)
                                    & (salaries <= remaining_salary)
                                    & (teams != def_opp)
                                )[0]
                            # grab names of players eligible
                            plyr_list = ids[valid_players]
                            # create np array of probability of being seelcted based on ownership and who is eligible at the position
                            prob_list = ownership[valid_players]
                            prob_list = prob_list / prob_list.sum()
                            boosted_salaries = np.array(
                                [
                                    salary_boost(s, salary_ceiling)
                                    for s in salaries[valid_players]
                                ]
                            )
                            boosted_probabilities = prob_list * boosted_salaries
                            boosted_probabilities /= (
                                boosted_probabilities.sum()
                            )  # normalize to ensure it sums to 1
                            try:
                                choice = rng.choice(plyr_list, p=boosted_probabilities)
                            except:
                                salary = 0
                                proj = 0
                                if team_stack == "":
                                    lineup = []
                                else:
                                    lineup = np.zeros(shape=pos_matrix.shape[1]).astype(
                                        str
                                    )
                                player_teams = []
                                def_opps = []
                                players_opposing_def = 0
                                lineup_matchups = []
                                in_lineup.fill(0)  # Reset the in_lineup array
                                k = 0  # Reset the player index
                                continue  # Skip to the next iteration of the while loop
                                # if remaining_salary <= np.min(salaries):
                                #     reject_counters["salary_too_high"] += 1
                                # else:
                                #     reject_counters["salary_too_low"]
                            choice_idx = np.nonzero(ids == choice)[0]
                            lineup.append(str(choice))
                            in_lineup[choice_idx] = 1
                            salary += _scalar(salaries[choice_idx])
                            proj += _scalar(projections[choice_idx])
                            player_teams.append(teams[choice_idx][0])
                            lineup_matchups.append(matchups[choice_idx[0]])
                            if teams[choice_idx][0] == def_opp:
                                players_opposing_def += 1
                            if max_players_per_team is not None:
                                team_count = Counter(player_teams)
                                if any(
                                    count > max_players_per_team
                                    for count in team_count.values()
                                ):
                                    salary = 0
                                    proj = 0
                                    if team_stack == "":
                                        lineup = []
                                    else:
                                        lineup = np.zeros(
                                            shape=pos_matrix.shape[1]
                                        ).astype(str)
                                    player_teams = []
                                    def_opps = []
                                    players_opposing_def = 0
                                    lineup_matchups = []
                                    in_lineup.fill(0)  # Reset the in_lineup array
                                    k = 0  # Reset the player index
                                    continue  # Skip to the next iteration of the while loop
                    k += 1
                # Must have a reasonable salary
                # if salary > salary_ceiling:
                #     reject_counters["salary_too_high"] += 1
                # elif salary < salary_floor:
                #     reject_counters["salary_too_low"] += 1
                if salary >= salary_floor and salary <= salary_ceiling:
                    # Must have a reasonable projection (within 60% of optimal) **people make a lot of bad lineups
                    if proj >= reasonable_projection:
                        if len(set(lineup_matchups)) > 1:
                            if max_players_per_team is not None:
                                team_count = Counter(player_teams)
                                if all(
                                    count <= max_players_per_team
                                    for count in team_count.values()
                                ):
                                    reject = False
                                    lus[lu_num] = {
                                        "Lineup": lineup,
                                        "Wins": 0,
                                        "Top1Percent": 0,
                                        "ROI": 0,
                                        "Cashes": 0,
                                        "Type": "generated",
                                        "Count": 0
                                    }
                                    if len(set(lineup)) != 9:
                                        print(
                                            "non stack lineup dupes",
                                            lu_num,
                                            plyr_stack_indices,
                                            str(lu_num),
                                            salaries[plyr_stack_indices],
                                            lineup,
                                            stack_len,
                                            team_stack,
                                            x,
                                        )
                            else:
                                reject = False
                                lus[lu_num] = {
                                    "Lineup": lineup,
                                    "Wins": 0,
                                    "Top1Percent": 0,
                                    "ROI": 0,
                                    "Cashes": 0,
                                    "Type": "generated",
                                    "Count": 0
                                }
                                if len(set(lineup)) != 9:
                                    print(
                                        "stack lineup dupes",
                                        lu_num,
                                        plyr_stack_indices,
                                        str(lu_num),
                                        salaries[plyr_stack_indices],
                                        lineup,
                                        stack_len,
                                        team_stack,
                                        x,
                                    )
                            # complete = 'completed'
                            # print(str(lu_num) + ' ' + complete)
                    #     else:
                    #         reject_counters["invalid_matchups"] += 1
                    # else:
                    #     reject_counters["projection_too_low"] += 1
            else:
                salary = 0
                proj = 0
                if sum(in_lineup) != 0:
                    in_lineup.fill(0)
                player_teams = []
                def_opps = []
                lineup_matchups = []
                filled_pos = np.zeros(shape=pos_matrix.shape[1])
                team_stack_len = 0
                k = 0
                stack = True
                lineup = np.zeros(shape=pos_matrix.shape[1]).astype(str)
                valid_team = np.nonzero(teams == team_stack)[0]
                # select qb
                qb_candidates = np.unique(
                    valid_team[np.nonzero(pos_matrix[valid_team, 1] > 0)[0]]
                )
                if qb_candidates.size == 0:
                    # No quarterback available for the selected team stack. Skip this lineup.
                    return None
                qb = qb_candidates[0]
                salary += salaries[qb]
                proj += projections[qb]
                # print(salary)
                team_stack_len += 1
                lineup[1] = ids[qb]
                in_lineup[qb] = 1
                lineup_matchups.append(matchups[qb])
                if use_te_stack:
                    valid_players = np.unique(
                        valid_team[
                            np.nonzero(pos_matrix[valid_team, 4:8] > 0)[0]
                        ]
                    )
                else:
                    valid_players = np.unique(
                        valid_team[
                            np.nonzero(pos_matrix[valid_team, 4:7] > 0)[0]
                        ]
                    )
                player_teams.append(teams[qb])
                players_opposing_def = 0
                opp_team = opponents[qb]
                plyr_list = ids[valid_players]
                prob_list = ownership[valid_players]
                prob_list = prob_list / prob_list.sum()
                while stack:
                    try:
                        choices = rng.choice(
                            a=plyr_list, p=prob_list, size=stack_len, replace=False
                        )
                        if len(set(choices)) != len(choices):
                            print(
                                "choice dupe",
                                plyr_stack_indices,
                                str(lu_num),
                                salaries[plyr_stack_indices],
                                lineup,
                                stack_len,
                                team_stack,
                                x,
                            )
                    except:
                        stack = False
                        continue
                    plyr_stack_indices = np.nonzero(np.in1d(ids, choices))[0]
                    x = 0
                    for p in plyr_stack_indices:
                        player_placed = False
                        for l in np.nonzero(pos_matrix[p] > 0)[0]:
                            if lineup[l] == "0.0":
                                lineup[l] = ids[p]
                                lineup_matchups.append(matchups[p])
                                player_teams.append(teams[p])
                                x += 1
                                player_placed = True
                                break
                            if player_placed:
                                break
                    # print(plyr_stack_indices, str(lu_num), salaries[plyr_stack_indices], lineup, stack_len, x)
                    if x == stack_len:
                        in_lineup[plyr_stack_indices] = 1
                        salary += sum(salaries[plyr_stack_indices])
                        # rint(salary)
                        proj += sum(projections[plyr_stack_indices])
                        # print(proj)
                        team_stack_len += stack_len
                        x = 0
                        stack = False
                    else:
                        stack = False
                # print(sum(in_lineup), stack_len)
                for ix, (l, pos) in enumerate(zip(lineup, pos_matrix.T)):
                    if l == "0.0":
                        if ix == 0:
                            valid_players = np.nonzero(
                                (pos > 0)
                                & (in_lineup == 0)
                                & (teams != team_stack)
                                & (opponents != team_stack)
                            )[0]
                            if valid_players.size == 0:
                                valid_players = np.nonzero(
                                    (pos > 0) & (in_lineup == 0)
                                )[0]
                            plyr_list = ids[valid_players]
                            prob_list = ownership[valid_players]
                            prob_list = prob_list / prob_list.sum()
                            try:
                                choice = rng.choice(plyr_list, p=prob_list)
                            except:
                                print("find failed on stack DST selection")
                            choice_idx = np.nonzero(ids == choice)[0]
                            in_lineup[choice_idx] = 1
                            try:
                                lineup[ix] = str(choice)
                            except IndexError:
                                print(lineup, choice, ix)
                            salary += _scalar(salaries[choice_idx])
                            proj += _scalar(projections[choice_idx])
                            def_opp = opponents[choice_idx][0]
                            lineup_matchups.append(matchups[choice_idx[0]])
                            player_teams.append(teams[choice_idx][0])
                            players_opposing_def = sum(
                                1 for team in player_teams if team == def_opp
                            )
                            if players_opposing_def > overlap_limit:
                                salary = 0
                                proj = 0
                                if team_stack == "":
                                    lineup = []
                                else:
                                    lineup = np.zeros(
                                        shape=pos_matrix.shape[1]
                                    ).astype(str)
                                player_teams = []
                                def_opps = []
                                players_opposing_def = 0
                                lineup_matchups = []
                                in_lineup.fill(0)
                                k = 0
                                continue
                            continue
                        elif k < 1:
                            if require_bring_back:
                                valid_players = np.nonzero(
                                    (pos > 0)
                                    & (in_lineup == 0)
                                    & (teams == opp_team)
                                )[0]
                                if valid_players.size == 0:
                                    valid_players = np.nonzero(
                                        (pos > 0) & (in_lineup == 0)
                                    )[0]
                            else:
                                valid_players = np.nonzero(
                                    (pos > 0) & (in_lineup == 0)
                                )[0]
                            # grab names of players eligible
                            plyr_list = ids[valid_players]
                            # create np array of probability of being selected based on ownership and who is eligible at the position
                            prob_list = ownership[valid_players]
                            prob_list = prob_list / prob_list.sum()
                            try:
                                choice = rng.choice(plyr_list, p=prob_list)
                            except:
                                print("find failed on stack and first player selection")

                            #    print(k, pos)
                            choice_idx = np.nonzero(ids == choice)[0]
                            in_lineup[choice_idx] = 1
                            try:
                                lineup[ix] = str(choice)
                            except IndexError:
                                print(lineup, choice, ix)
                            salary += _scalar(salaries[choice_idx])
                            proj += _scalar(projections[choice_idx])
                            def_opp = opponents[choice_idx][0]
                            lineup_matchups.append(matchups[choice_idx[0]])
                            player_teams.append(teams[choice_idx][0])
                            k += 1
                        elif k >= 1:
                            remaining_salary = salary_ceiling - salary
                            if players_opposing_def < overlap_limit:
                                if k == total_players - 1:
                                    valid_players = np.nonzero(
                                        (pos > 0)
                                        & (in_lineup == 0)
                                        & (salaries <= remaining_salary)
                                        & (salary + salaries >= salary_floor)
                                    )[0]
                                else:
                                    valid_players = np.nonzero(
                                        (pos > 0)
                                        & (in_lineup == 0)
                                        & (salaries <= remaining_salary)
                                    )[0]
                                # grab names of players eligible
                                plyr_list = ids[valid_players]
                                # create np array of probability of being seelcted based on ownership and who is eligible at the position
                                prob_list = ownership[valid_players]
                                prob_list = prob_list / prob_list.sum()
                                if k == total_players - 1:
                                    boosted_salaries = np.array(
                                        [
                                            salary_boost(s, salary_ceiling)
                                            for s in salaries[valid_players]
                                        ]
                                    )
                                    boosted_probabilities = prob_list * boosted_salaries
                                    boosted_probabilities /= (
                                        boosted_probabilities.sum()
                                    )  # normalize to ensure it sums to 1
                                try:
                                    if k == total_players - 1:
                                        choice = rng.choice(
                                            plyr_list, p=boosted_probabilities
                                        )
                                    else:
                                        choice = rng.choice(plyr_list, p=prob_list)
                                except:
                                    salary = 0
                                    proj = 0
                                    if team_stack == "":
                                        lineup = []
                                    else:
                                        lineup = np.zeros(
                                            shape=pos_matrix.shape[1]
                                        ).astype(str)
                                    player_teams = []
                                    def_opps = []
                                    players_opposing_def = 0
                                    lineup_matchups = []
                                    in_lineup.fill(0)  # Reset the in_lineup array
                                    k = 0  # Reset the player index
                                    continue  # Skip to the next iteration of the while loop
                                    # if remaining_salary <= np.min(salaries):
                                    #     reject_counters["salary_too_high"] += 1
                                    # else:
                                    #     reject_counters["salary_too_low"]
                                choice_idx = np.nonzero(ids == choice)[0]
                                try:
                                    lineup[ix] = str(choice)
                                except IndexError:
                                    print(lineup, choice, ix)
                                in_lineup[choice_idx] = 1
                                salary += _scalar(salaries[choice_idx])
                                proj += _scalar(projections[choice_idx])
                                player_teams.append(teams[choice_idx][0])
                                lineup_matchups.append(matchups[choice_idx[0]])
                                if max_players_per_team is not None:
                                    team_count = Counter(player_teams)
                                    if any(
                                        count > max_players_per_team
                                        for count in team_count.values()
                                    ):
                                        salary = 0
                                        proj = 0
                                        if team_stack == "":
                                            lineup = []
                                        else:
                                            lineup = np.zeros(
                                                shape=pos_matrix.shape[1]
                                            ).astype(str)
                                        player_teams = []
                                        def_opps = []
                                        players_opposing_def = 0
                                        lineup_matchups = []
                                        in_lineup.fill(0)  # Reset the in_lineup array
                                        k = 0  # Reset the player index
                                        continue  # Skip to the next iteration of the while loop
                                if teams[choice_idx][0] == def_opp:
                                    players_opposing_def += 1
                                if teams[choice_idx][0] == team_stack:
                                    team_stack_len += 1
                            else:
                                if k == total_players - 1:
                                    valid_players = np.nonzero(
                                        (pos > 0)
                                        & (in_lineup == 0)
                                        & (salaries <= remaining_salary)
                                        & (salary + salaries >= salary_floor)
                                        & (teams != def_opp)
                                    )[0]
                                else:
                                    valid_players = np.nonzero(
                                        (pos > 0)
                                        & (in_lineup == 0)
                                        & (salaries <= remaining_salary)
                                        & (teams != def_opp)
                                    )[0]
                                # grab names of players eligible
                                plyr_list = ids[valid_players]
                                # create np array of probability of being seelcted based on ownership and who is eligible at the position
                                prob_list = ownership[valid_players]
                                prob_list = prob_list / prob_list.sum()
                                boosted_salaries = np.array(
                                    [
                                        salary_boost(s, salary_ceiling)
                                        for s in salaries[valid_players]
                                    ]
                                )
                                boosted_probabilities = prob_list * boosted_salaries
                                boosted_probabilities /= (
                                    boosted_probabilities.sum()
                                )  # normalize to ensure it sums to 1
                                try:
                                    choice = rng.choice(
                                        plyr_list, p=boosted_probabilities
                                    )
                                except:
                                    salary = 0
                                    proj = 0
                                    if team_stack == "":
                                        lineup = []
                                    else:
                                        lineup = np.zeros(
                                            shape=pos_matrix.shape[1]
                                        ).astype(str)
                                    player_teams = []
                                    def_opps = []
                                    players_opposing_def = 0
                                    lineup_matchups = []
                                    in_lineup.fill(0)  # Reset the in_lineup array
                                    k = 0  # Reset the player index
                                    continue  # Skip to the next iteration of the while loop
                                    # if remaining_salary <= np.min(salaries):
                                    #     reject_counters["salary_too_high"] += 1
                                    # else:
                                    #     reject_counters["salary_too_low"]
                                choice_idx = np.nonzero(ids == choice)[0]
                                lineup[ix] = str(choice)
                                in_lineup[choice_idx] = 1
                                salary += _scalar(salaries[choice_idx])
                                proj += _scalar(projections[choice_idx])
                                player_teams.append(teams[choice_idx][0])
                                lineup_matchups.append(matchups[choice_idx[0]])
                                if teams[choice_idx][0] == def_opp:
                                    players_opposing_def += 1
                                if teams[choice_idx][0] == team_stack:
                                    team_stack_len += 1
                                if max_players_per_team is not None:
                                    team_count = Counter(player_teams)
                                    if any(
                                        count > max_players_per_team
                                        for count in team_count.values()
                                    ):
                                        salary = 0
                                        proj = 0
                                        if team_stack == "":
                                            lineup = []
                                        else:
                                            lineup = np.zeros(
                                                shape=pos_matrix.shape[1]
                                            ).astype(str)
                                        player_teams = []
                                        def_opps = []
                                        players_opposing_def = 0
                                        lineup_matchups = []
                                        in_lineup.fill(0)  # Reset the in_lineup array
                                        k = 0  # Reset the player index
                                        continue  # Skip to the next iteration of the while loop
                            k += 1
                    else:
                        k += 1
                # Must have a reasonable salary
                if team_stack_len >= stack_len:
                    if salary >= salary_floor and salary <= salary_ceiling:
                        # loosening reasonable projection constraint for team stacks
                        if proj >= reasonable_stack_projection:
                            if len(set(lineup_matchups)) > 1:
                                if max_players_per_team is not None:
                                    team_count = Counter(player_teams)
                                    if all(
                                        count <= max_players_per_team
                                        for count in team_count.values()
                                    ):
                                        reject = False
                                        lus[lu_num] = {
                                            "Lineup": lineup,
                                            "Wins": 0,
                                            "Top1Percent": 0,
                                            "ROI": 0,
                                            "Cashes": 0,
                                            "Type": "generated",
                                            "Count": 0,
                                        }
                                        if len(set(lineup)) != 9:
                                            print(
                                                "stack lineup dupes",
                                                lu_num,
                                                plyr_stack_indices,
                                                str(lu_num),
                                                salaries[plyr_stack_indices],
                                                lineup,
                                                stack_len,
                                                team_stack,
                                                x,
                                            )

                                else:
                                    reject = False
                                    lus[lu_num] = {
                                        "Lineup": lineup,
                                        "Wins": 0,
                                        "Top1Percent": 0,
                                        "ROI": 0,
                                        "Cashes": 0,
                                        "Type": "generated",
                                        "Count": 0,
                                    }
                                    if len(set(lineup)) != 9:
                                        print(
                                            "stack lineup dupes",
                                            lu_num,
                                            plyr_stack_indices,
                                            str(lu_num),
                                            salaries[plyr_stack_indices],
                                            lineup,
                                            stack_len,
                                            team_stack,
                                            x,
                                        )
                #             else:
                #                 reject_counters["invalid_matchups"] += 1
                #         else:
                #             reject_counters["projection_too_low"] += 1
                #     else:
                #         if salary > salary_ceiling:
                #             reject_counters["salary_too_high"] += 1
                #         elif salary < salary_floor:
                #             reject_counters["salary_too_low"] += 1
                # else:
                #     reject_counters["stack_length_insufficient"] += 1
        # return lus, reject_counters
        return lus

    def generate_field_lineups(self):

        # --- Begin: DST ID backfill for sim player pool & guard ---
        try:
            # If simulator has a players table, normalize its pos/team columns
            if hasattr(self, "players_df") and self.players_df is not None:
                for col in ("pos", "Pos", "position", "Position"):
                    if col in self.players_df.columns:
                        self.players_df[col] = self.players_df[col].apply(_canon_pos_any)
                        if col != "Position":
                            self.players_df["Position"] = self.players_df[col]
                if "team" in self.players_df.columns:
                    self.players_df["team"] = (
                        self.players_df["team"].astype(str).str.upper().str.strip().replace({"LA": "LAR"})
                    )
                    self.players_df["TeamAbbrev"] = self.players_df["team"]
        
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
        pool_size = (
            max(int(self.field_size * self.pool_factor), self.field_size)
            if self.profile
            else self.field_size
        )
        diff = pool_size - len(self.field_lineups)
        if diff <= 0:
            print(
                "supplied lineups >= contest field size. only retrieving the first "
                + str(self.field_size)
                + " lineups"
            )
        else:
            print("Generating " + str(diff) + " lineups.")
            ids = []
            ownership = []
            salaries = []
            projections = []
            positions = []
            teams = []
            opponents = []
            matchups = []
            # put def first to make it easier to avoid overlap
            temp_roster_construction = [
                "DST",
                "QB",
                "RB",
                "RB",
                "WR",
                "WR",
                "WR",
                "TE",
                "FLEX",
            ]
            for k in self.player_dict.keys():
                if "Team" not in self.player_dict[k].keys():
                    print(
                        self.player_dict[k]["Name"],
                        " name mismatch between projections and player ids!",
                    )
                ids.append(self.player_dict[k]["ID"])
                ownership.append(self.player_dict[k]["Ownership"])
                salaries.append(self.player_dict[k]["Salary"])
                if self.player_dict[k]["fieldFpts"] >= self.projection_minimum:
                    projections.append(self.player_dict[k]["fieldFpts"])
                else:
                    projections.append(0)
                teams.append(self.player_dict[k]["Team"])
                opponents.append(self.player_dict[k]["Opp"])
                matchups.append(self.player_dict[k]["Matchup"])
                pos_list = []
                p = _canon_pos_primary(self.player_dict[k].get("Position"))
                for slot in temp_roster_construction:
                    if slot == "FLEX":
                        pos_list.append(1 if p in ("RB","WR","TE") else 0)
                    else:
                        pos_list.append(1 if p == slot else 0)
                positions.append(np.array(pos_list, dtype=int))
            in_lineup = np.zeros(shape=len(ids))
            ownership = np.array(ownership)
            salaries = np.array(salaries)
            projections = np.array(projections)
            pos_matrix = np.array(positions)
            ids = np.array(ids)
            optimal_score = self.optimal_score
            salary_floor = self.min_lineup_salary
            salary_ceiling = self.salary
            max_pct_off_optimal = self.max_pct_off_optimal
            stack_usage = self.pct_field_using_stacks
            teams = np.array(teams)
            opponents = np.array(opponents)
            overlap_limit = self.overlap_limit
            problems = []
            stacks = np.random.binomial(n=1, p=self.pct_field_using_stacks, size=diff)
            stack_len = np.random.choice(
                a=[1, 2],
                p=[1 - self.pct_field_double_stacks, self.pct_field_double_stacks],
                size=diff,
            )
            max_stack_len = 2
            num_players_in_roster = len(self.roster_construction)
            a = list(self.stacks_dict.keys())
            p = np.array(list(self.stacks_dict.values()))
            probs = p / sum(p)
            stacks = stacks.astype(str)
            for i in range(len(stacks)):
                if stacks[i] == "1":
                    choice = random.choices(a, weights=probs, k=1)
                    stacks[i] = choice[0]
                else:
                    stacks[i] = ""
            # creating tuples of the above np arrays plus which lineup number we are going to create
            for i in range(diff):
                lu_tuple = (
                    i,
                    ids,
                    in_lineup,
                    pos_matrix,
                    ownership,
                    salary_floor,
                    salary_ceiling,
                    optimal_score,
                    salaries,
                    projections,
                    max_pct_off_optimal,
                    teams,
                    opponents,
                    stacks[i],
                    stack_len[i],
                    overlap_limit,
                    max_stack_len,
                    matchups,
                    num_players_in_roster,
                    self.site,
                    self.use_te_stack,
                    self.require_bring_back,
                )
                problems.append(lu_tuple)
            start_time = time.time()
            with mp.Pool() as pool:
                output = pool.starmap(self.generate_lineups, problems)
                print(
                    "number of running processes =",
                    pool.__dict__["_processes"]
                    if (pool.__dict__["_state"]).upper() == "RUN"
                    else None,
                )
                pool.close()
                pool.join()
            print("pool closed")
            valid_output = [o for o in output if o is not None]
            failed = len(output) - len(valid_output)
            if failed:
                print(f"{failed} lineups failed to generate and were skipped")
            if not valid_output and diff > 0:
                top_players = sorted(
                    self.player_dict.values(),
                    key=lambda v: v.get("fieldFpts", 0),
                    reverse=True,
                )
                fallback_ids = [
                    p["ID"] for p in top_players[: len(self.roster_construction)]
                ]
                fallback = {
                    0: {
                        "Lineup": fallback_ids,
                        "Wins": 0,
                        "Top1Percent": 0,
                        "Cashes": 0,
                        "ROI": 0,
                        "Type": "Fallback",
                        "Count": 1,
                    }
                }
                valid_output = [fallback]
            self.update_field_lineups(valid_output, len(valid_output))
            end_time = time.time()
            print("lineups took " + str(end_time - start_time) + " seconds")
            print(str(len(valid_output)) + " field lineups successfully generated")
            # print("Reject counters:", dict(overall_reject_counters))

            # print(self.field_lineups)

        if self.profile and self.targets:
            candidates = [v["Lineup"] for v in self.field_lineups.values()]
            selected = select_lineups(
                candidates, self.player_dict, self.targets, self.field_size
            )

            self.field_lineups = {}
            self.seen_lineups = {}
            self.seen_lineups_ix = {}
            for i, lu in enumerate(selected):
                if self.site == "dk":
                    sorted_lineup = self.sort_lineup_by_start_time(lu)
                else:
                    sorted_lineup = lu
                self.field_lineups[i] = {
                    "Lineup": sorted_lineup,
                    "Wins": 0,
                    "Top1Percent": 0,
                    "Cashes": 0,
                    "ROI": 0,
                    "Type": "Profile",
                    "Count": 1,
                }
                lineup_set = frozenset(sorted_lineup)
                self.seen_lineups[lineup_set] = 1
                self.seen_lineups_ix[lineup_set] = i

    def get_start_time(self, player_id):
        for _, player in self.player_dict.items():
            if player["ID"] == player_id:
                matchup = player["Matchup"]
                return self.game_info.get(matchup)
        return None

    def get_player_attribute(self, player_id, attribute):
        for _, player in self.player_dict.items():
            if player["ID"] == player_id:
                return player.get(attribute, None)
        return None
    def is_valid_for_position(self, player, position_idx):
        # Accept both single-string and iterable positions; canonicalize
        raw = self.get_player_attribute(player, "Position")
        try:
            if isinstance(raw, (list, tuple, set)):
                pos_list = list(raw)
            else:
                pos_list = [raw]
        except Exception:
            pos_list = [raw]
        pos_list = [(_canon_pos_primary(p) if p is not None else "") for p in pos_list]
        pos_list = [p for p in pos_list if p]
        return any(p in self.position_map[position_idx] for p in pos_list)



    def sort_lineup_by_start_time(self, lineup):
        flex_index = 8  # Assuming FLEX is at index 8
        flex_player = lineup[flex_index]
        flex_player_start_time = self.get_start_time(flex_player)

        # Initialize variables to track the best swap candidate
        latest_start_time = flex_player_start_time
        swap_candidate_index = None

        # Iterate over RB, WR, and TE positions (indices 2 to 7)
        for i in range(2, 8):
            current_player = lineup[i]
            current_player_start_time = self.get_start_time(current_player)

            # Update the latest start time and swap candidate index
            if (
                current_player_start_time
                and (latest_start_time is None or current_player_start_time > latest_start_time)
                and self.is_valid_for_position(flex_player, i)
                and self.is_valid_for_position(current_player, flex_index)
            ):

                latest_start_time = current_player_start_time
                swap_candidate_index = i

        # Perform the swap if a suitable candidate is found
        if swap_candidate_index is not None:
            #print(f"Swapping: {lineup[swap_candidate_index]} with {flex_player}")
            lineup[flex_index], lineup[swap_candidate_index] = lineup[swap_candidate_index], lineup[flex_index]

        return lineup

    def update_field_lineups(self, output, diff):
        if len(self.field_lineups) == 0:
            new_keys = list(range(0, self.field_size))
        else:
            new_keys = list(
                range(
                    max(self.field_lineups.keys()) + 1,
                    max(self.field_lineups.keys()) + 1 + diff,
                )
            )

        nk = new_keys[0]
        for i, o in enumerate(output):
            lineup_list = sorted(next(iter(o.values()))["Lineup"])
            lineup_set = frozenset(lineup_list)

            # Keeping track of lineup duplication counts
            if lineup_set in self.seen_lineups:
                self.seen_lineups[lineup_set] += 1

                # Increase the count in field_lineups using the index stored in seen_lineups_ix
                self.field_lineups[self.seen_lineups_ix[lineup_set]]["Count"] += 1
            else:
                self.seen_lineups[lineup_set] = 1

                # Updating the field lineups dictionary
                if nk in self.field_lineups.keys():
                    print("bad lineups dict, please check dk_data files")
                else:
                    if self.site == "dk":
                        sorted_lineup = self.sort_lineup_by_start_time(
                            next(iter(o.values()))["Lineup"]
                        )
                    else:
                        sorted_lineup = next(iter(o.values()))["Lineup"]


                    self.field_lineups[nk] = next(iter(o.values()))
                    self.field_lineups[nk]["Lineup"] = sorted_lineup
                    self.field_lineups[nk]["Count"] += self.seen_lineups[lineup_set]
                    # Store the new nk in seen_lineups_ix for quick access in the future
                    self.seen_lineups_ix[lineup_set] = nk
                    nk += 1

    def calc_gamma(self, mean, sd):
        alpha = (mean / sd) ** 2
        beta = sd**2 / mean
        return alpha, beta

    @staticmethod
    def run_simulation_for_game(
        game_idx,
        game_payload,
        num_iterations,
        roster_construction,
        seed=None,
    ):
        """
        Worker for one game. Accepts an explicit seed so we don’t rely on `self` in a subprocess.
        """
        team1_id, team1, team2_id, team2 = game_payload
        # Define correlations between positions

        def get_corr_value(player1, player2):
            """Robust correlation: canonical positions + safe defaults."""
            # Player-specific override first
            try:
                pc = player1.get("Player Correlations", {})
                if player2.get("Name") in pc:
                    return float(pc[player2["Name"]])
            except Exception:
                pass

            # Canonicalize primary positions
            try:
                pos1 = _canon_pos_primary(player1.get("Position"))
            except Exception:
                pos1 = ""
            try:
                pos2 = _canon_pos_primary(player2.get("Position"))
            except Exception:
                pos2 = ""
            if not pos1 or not pos2:
                return 0.0

            # Base correlations by position (fallback)
            position_correlations = {
                "QB": -0.5,
                "RB": -0.2,
                "WR": 0.1,
                "TE": -0.2,
                "K": -0.5,
                "DST": -0.5,
            }

            # Same team & same canonical pos -> base table
            try:
                same_team = player1.get("Team") == player2.get("Team")
            except Exception:
                same_team = False

            if same_team and pos1 == pos2:
                return float(position_correlations.get(pos1, 0.0))

            # Else use player1["Correlations"] with canonical keys
            try:
                corr = player1.get("Correlations", {})
                key = (f"Opp {pos2}") if not same_team else pos2
                return float(corr.get(key, 0.0))
            except Exception:
                return 0.0

        def build_covariance_matrix(players):
            N = len(players)
            corr_matrix = [[0 for _ in range(N)] for _ in range(N)]
            for i in range(N):
                for j in range(N):
                    if i == j:
                        corr_matrix[i][j] = 1
                    else:
                        corr_matrix[i][j] = get_corr_value(players[i], players[j])

            risks = [
                {
                    "sigma_base": p["SigmaBase"],
                    "r_plus": p["RPlus"],
                    "r_minus": p["RMinus"],
                    "sigma_eff": p["StdDev"],
                }
                for p in players
            ]
            means = [p["Fpts"] for p in players]
            corr = np.array(corr_matrix, dtype=float)
            matrix = build_covariance_from_corr(means, risks, corr).tolist()
            return matrix, corr_matrix

        game = team1 + team2
        covariance_matrix, corr_matrix = build_covariance_matrix(game)
        corr = np.array(corr_matrix, dtype=float)

        means = [player["Fpts"] for player in game]
        risks = [
            {
                "sigma_base": player["SigmaBase"],
                "r_plus": player["RPlus"],
                "r_minus": player["RMinus"],
                "sigma_eff": player["StdDev"],
            }
            for player in game
        ]
        rng = np.random.default_rng(seed)
        samples = np.vstack(
            [sample_skewed_outcomes(rng, means, risks, corr) for _ in range(num_iterations)]
        )

        player_samples = []
        for i, player in enumerate(game):
            if "QB" in player["Position"]:
                sample = samples[:, i]
            else:
                sample = samples[:, i]
            # if player['Team'] in ['LAR','SEA']:
            #     print(player['Name'], player['Fpts'], player['StdDev'], sample, np.mean(sample), np.std(sample))
            player_samples.append(sample)

        temp_fpts_dict = {}
        # print(team1_id, team2_id, len(game), uniform_samples.T.shape, len(player_samples), covariance_matrix.shape )

        for i, player in enumerate(game):
            temp_fpts_dict[player["ID"]] = player_samples[i]

        # fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, figsize=(15, 25))
        # fig.tight_layout(pad=5.0)

        # for i, player in enumerate(game):
        #     sns.kdeplot(player_samples[i], ax=ax1, label=player['Name'])

        # ax1.legend(loc='upper right', fontsize=14)
        # ax1.set_xlabel('Fpts', fontsize=14)
        # ax1.set_ylabel('Density', fontsize=14)
        # ax1.set_title(f'Team {team1_id}{team2_id} Distributions', fontsize=14)
        # ax1.tick_params(axis='both', which='both', labelsize=14)

        # y_min, y_max = ax1.get_ylim()
        # ax1.set_ylim(y_min, y_max*1.1)

        # ax1.set_xlim(-5, 50)

        # # # Sorting players and correlating their data
        # player_names = [f"{player['Name']} ({player['Position']})" if player['Position'] is not None else f"{player['Name']} (P)" for player in game]

        # # # Ensuring the data is correctly structured as a 2D array
        # sorted_samples_array = np.array(player_samples)
        # if sorted_samples_array.shape[0] < sorted_samples_array.shape[1]:
        #     sorted_samples_array = sorted_samples_array.T

        # correlation_matrix = pd.DataFrame(np.corrcoef(sorted_samples_array.T), columns=player_names, index=player_names)

        # sns.heatmap(correlation_matrix, annot=True, ax=ax2, cmap='YlGnBu', cbar_kws={"shrink": .5})
        # ax2.set_title(f'Correlation Matrix for Game {team1_id}{team2_id}', fontsize=14)

        # original_corr_matrix = pd.DataFrame(corr_matrix, columns=player_names, index=player_names)
        # sns.heatmap(original_corr_matrix, annot=True, ax=ax3, cmap='YlGnBu', cbar_kws={"shrink": .5})
        # ax3.set_title(f'Original Correlation Matrix for Game {team1_id}{team2_id}', fontsize=14)

        # cov_matrix = pd.DataFrame(covariance_matrix, columns=player_names, index=player_names)
        # sns.heatmap(cov_matrix, annot=True, ax=ax4, cmap='YlGnBu', cbar_kws={"shrink": .5})
        # ax4.set_title(f'Original Covariance Matrix for Game {team1_id}{team2_id}', fontsize=14)

        # plt.savefig(f'output/Team_{team1_id}{team2_id}_Distributions_Correlation.png', bbox_inches='tight')
        # plt.close()

        return temp_fpts_dict
    
    @staticmethod
    @jit(nopython=True)
    def calculate_payouts(args):
        (
            ranks,
            payout_array,
            entry_fee,
            field_lineup_keys,
            use_contest_data,
            field_lineups_count,
        ) = args
        num_lineups = len(field_lineup_keys)
        combined_result_array = np.zeros(num_lineups)

        payout_cumsum = np.cumsum(payout_array)

        for r in range(ranks.shape[1]):
            ranks_in_sim = ranks[:, r]
            payout_index = 0
            for lineup_index in ranks_in_sim:
                lineup_count = field_lineups_count[lineup_index]
                prize_for_lineup = (
                    (
                        payout_cumsum[payout_index + lineup_count - 1]
                        - payout_cumsum[payout_index - 1]
                    )
                    / lineup_count
                    if payout_index != 0
                    else payout_cumsum[payout_index + lineup_count - 1] / lineup_count
                )
                combined_result_array[lineup_index] += prize_for_lineup
                payout_index += lineup_count
        return combined_result_array    

    def run_tournament_simulation(self):
        self._normalize_positions_in_tables()
        print("Running " + str(self.num_iterations) + " simulations")
        for f in self.field_lineups:
            if len(self.field_lineups[f]["Lineup"]) != 9:
                print("bad lineup", f, self.field_lineups[f])
        print(f"Number of unique field lineups: {len(self.field_lineups.keys())}")

        start_time = time.time()
        temp_fpts_dict = {}
        size = self.num_iterations
        parent_seed = getattr(self, "seed", None)

        def _child_seed(parent_seed, idx):
            if parent_seed is None:
                return None
            return (hash((int(parent_seed), int(idx))) & 0xFFFFFFFF) or 1

        game_simulation_params = []
        for idx, m in enumerate(self.matchups):
            payload = (
                m[0],
                self.teams_dict[m[0]],
                m[1],
                self.teams_dict[m[1]],
            )
            game_simulation_params.append(
                (
                    idx,
                    payload,
                    self.num_iterations,
                    self.roster_construction,
                    _child_seed(parent_seed, idx),
                )
            )
        with mp.Pool() as pool:
            results = pool.starmap(self.run_simulation_for_game, game_simulation_params)

        for res in results:
            temp_fpts_dict.update(res)

        # generate arrays for every sim result for each player in the lineup and sum
        fpts_array = np.zeros(shape=(len(self.field_lineups), self.num_iterations))
        # converting payout structure into an np friendly format, could probably just do this in the load contest function
        # print(self.field_lineups)
        # print(temp_fpts_dict)
        # print(payout_array)
        # print(self.player_dict[('patrick mahomes', 'FLEX', 'KC')])
        field_lineups_count = np.array(
            [self.field_lineups[idx]["Count"] for idx in self.field_lineups.keys()]
        )

        for index, values in self.field_lineups.items():
            try:
                fpts_sim = sum([temp_fpts_dict[player] for player in values["Lineup"]])
            except KeyError:
                for player in values["Lineup"]:
                    if player not in temp_fpts_dict.keys():
                        print(player)
                        # for k,v in self.player_dict.items():
                        # if v['ID'] == player:
                        #        print(k,v)
                # print('cant find player in sim dict', values["Lineup"], temp_fpts_dict.keys())
            # store lineup fpts sum in 2d np array where index (row) corresponds to index of field_lineups and columns are the fpts from each sim
            fpts_array[index] = fpts_sim

        fpts_array = fpts_array.astype(np.float16)
        # ranks = np.argsort(fpts_array, axis=0)[::-1].astype(np.uint16)
        ranks = np.argsort(-fpts_array, axis=0).astype(np.uint32)

        # jitter tracking removed for simulator

        # count wins, top 10s vectorized
        wins, win_counts = np.unique(ranks[0, :], return_counts=True)
        cashes, cash_counts = np.unique(ranks[0:len(list(self.payout_structure.values()))], return_counts=True)

        top1pct, top1pct_counts = np.unique(
            ranks[0 : math.ceil(0.01 * len(self.field_lineups)), :], return_counts=True
        )

        payout_array = np.array(list(self.payout_structure.values()))
        # subtract entry fee
        payout_array = payout_array - self.entry_fee
        l_array = np.full(
            shape=self.field_size - len(payout_array), fill_value=-self.entry_fee
        )
        payout_array = np.concatenate((payout_array, l_array))
        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))

        # Adjusted ROI calculation
        # print(field_lineups_count.shape, payout_array.shape, ranks.shape, fpts_array.shape)

        # Split the simulation indices into chunks
        field_lineups_keys_array = np.array(list(self.field_lineups.keys()))

        chunk_size = max(1, self.num_iterations // 16)  # Ensure non-zero chunk size
        simulation_chunks = [
            (
                ranks[:, i : min(i + chunk_size, self.num_iterations)].copy(),
                payout_array,
                self.entry_fee,
                field_lineups_keys_array,
                self.use_contest_data,
                field_lineups_count,
            )  # Adding field_lineups_count here
            for i in range(0, self.num_iterations, chunk_size)
        ]

        # Use the pool to process the chunks in parallel
        with mp.Pool() as pool:
            results = pool.map(self.calculate_payouts, simulation_chunks)

        combined_result_array = np.sum(results, axis=0)

        total_sum = 0
        index_to_key = list(self.field_lineups.keys())
        for idx, roi in enumerate(combined_result_array):
            lineup_key = index_to_key[idx]
            lineup_count = self.field_lineups[lineup_key][
                "Count"
            ]  # Assuming "Count" holds the count of the lineups
            total_sum += roi * lineup_count
            self.field_lineups[lineup_key]["ROI"] += roi

        for idx in self.field_lineups.keys():
            if idx in wins:
                self.field_lineups[idx]["Wins"] += win_counts[np.where(wins == idx)][0]
            if idx in top1pct:
                self.field_lineups[idx]["Top1Percent"] += top1pct_counts[
                    np.where(top1pct == idx)
                ][0]
            if idx in cashes:
                self.field_lineups[idx]["Cashes"] += cash_counts[np.where(cashes == idx)][0]
        try:
            if self._audit:
                self._audit.output_dir = self.output_dir
                self.risk_table_df = self._audit.build_risk_table()
                self._audit.save_risk_table("risk_table_simulator.csv")
        except Exception:
            pass

        end_time = time.time()
        diff = end_time - start_time
        print(
            str(self.num_iterations)
            + " tournament simulations finished in "
            + str(diff)
            + "seconds. Outputting."
        )

    def output(self):
        if not self.field_lineups:
            self.generate_field_lineups()

        id_player_dict = {}
        for v in self.player_dict.values():
            # ``analyze_lineup`` expects the player dictionary to be keyed by
            # player ID with ``Position`` stored as a simple string.  The
            # simulator keeps positions as a list (e.g. ["WR", "FLEX"]) which
            # caused all lookups to fail and every lineup to appear as
            # ``No Stack``.  Convert the position list to its primary element
            # here so stack detection mirrors the optimiser.
            pos = v.get("Position")
            if isinstance(pos, list):
                pos = pos[0]
            id_player_dict[v["ID"]] = {
                **v,
                "Position": pos,
                "Opponent": v.get("Opp"),
            }

        report = report_lineup_exposures(
            [x["Lineup"] for x in self.field_lineups.values()],
            id_player_dict,
            self.config,
        )
        self.stack_exposure_df = (
            pd.concat(
                [
                    pd.Series(report.get("presence", {}), name="presence"),
                    pd.Series(report.get("multiplicity", {}), name="multiplicity"),
                    pd.Series(report.get("bucket", {}), name="bucket"),
                ],
                axis=1,
            )
            .fillna(0)
            .rename_axis("Stack")
            .reset_index()
        )
        unique = {}
        for index, x in self.field_lineups.items():
            # if index == 0:
            #    print(x)
            lu_type = x["Type"]
            salary = 0
            fpts_p = 0
            fieldFpts_p = 0
            act_p = 0
            ceil_p = 0
            own_p = []
            lu_names = []
            lu_teams = []
            players_vs_def = 0
            def_opps = []
            simDupes = x['Count']
            for id in x["Lineup"]:
                for k, v in self.player_dict.items():
                    if v["ID"] == id:
                        if "DST" in v["Position"]:
                            def_opps.append(v["Opp"])
            for id in x["Lineup"]:
                for k, v in self.player_dict.items():
                    if v["ID"] == id:
                        salary += v["Salary"]
                        fpts_p += v["Fpts"]
                        fieldFpts_p += v["fieldFpts"]
                        act_p += v.get("ActPts", 0)
                        ceil_p += v["Ceiling"]
                        own_p.append(v["Ownership"] / 100)
                        lu_names.append(v["Name"])
                        if "DST" not in v["Position"]:
                            lu_teams.append(v["Team"])
                            if v["Team"] in def_opps:
                                players_vs_def += 1
                        continue


            # Analyze stacking metrics for the current lineup. ``analyze_lineup``
            # expects a list of player IDs and a dictionary keyed by those IDs.
            # ``id_player_dict`` was created above for this purpose. The function
            # returns information about stack presence and counts, which is used
            # to build the ``primaryStack`` string below. Previously ``metrics``
            # was referenced without being defined, resulting in a ``NameError``
            # when this method was executed.
            metrics = analyze_lineup(x["Lineup"], id_player_dict)

            # Determine the primary and secondary stack types.  ``analyze_lineup``
            # returns counts for each stack; pick the top two (excluding the
            # "No Stack" placeholder) so that the simulator output mirrors the
            # optimiser's stack columns.
            stack_items = [
                (k, v)
                for k, v in metrics["counts"].items()
                if v > 0 and k != "No Stack"
            ]
            stack_items.sort(key=lambda kv: kv[1], reverse=True)

            def _fmt(item):
                name, count = item
                return f"{name} x{count}" if count > 1 else name

            if stack_items:
                primaryStack = _fmt(stack_items[0])
                secondaryStack = _fmt(stack_items[1]) if len(stack_items) > 1 else ""
            elif metrics["presence"].get("No Stack"):
                primaryStack = "No Stack"
                secondaryStack = ""
            else:
                primaryStack = ""
                secondaryStack = ""
            own_p = np.prod(own_p)
            win_p = round(x["Wins"] / self.num_iterations * 100, 2)
            top10_p = round(x["Top1Percent"] / self.num_iterations * 100, 2)
            cash_p = round(x["Cashes"] / self.num_iterations * 100, 2)
            if self.site == "dk":

                # Build player name/id pairs (deterministic slot mapping)
                # Reconstruct DK slots from canonical positions
                slot_names = {"QB": None, "RB1": None, "RB2": None, "WR1": None, "WR2": None, "WR3": None, "TE": None, "FLEX": None, "DST": None}
                details = []
                for i, pid in enumerate(x["Lineup"]):
                    name = lu_names[i]
                    pos_raw = None
                    for k, v in self.player_dict.items():
                        if v.get("ID") == pid:
                            pos_raw = v.get("Position")
                            break
                    pos = _canon_pos_primary(pos_raw)
                    details.append((pid, name, pos))
                flex_pool = []
                for pid, name, pos in details:
                    label = f"{name.replace('#','-')} ({pid})"
                    if pos == "QB" and slot_names["QB"] is None:
                        slot_names["QB"] = label
                    elif pos == "DST" and slot_names["DST"] is None:
                        slot_names["DST"] = label
                    elif pos == "TE" and slot_names["TE"] is None:
                        slot_names["TE"] = label
                    elif pos == "RB":
                        if slot_names["RB1"] is None:
                            slot_names["RB1"] = label
                        elif slot_names["RB2"] is None:
                            slot_names["RB2"] = label
                        else:
                            # Only allow skill players to be considered for FLEX
                            if pos in ("RB","WR","TE"):
                                flex_pool.append(label)
                    elif pos == "WR":
                        if slot_names["WR1"] is None:
                            slot_names["WR1"] = label
                        elif slot_names["WR2"] is None:
                            slot_names["WR2"] = label
                        elif slot_names["WR3"] is None:
                            slot_names["WR3"] = label
                        else:
                            # Only allow skill players to be considered for FLEX
                            if pos in ("RB","WR","TE"):
                                flex_pool.append(label)
                    else:
                        # Only allow skill players to be considered for FLEX
                        if pos in ("RB","WR","TE"):
                            flex_pool.append(label)
                if slot_names["FLEX"] is None and flex_pool:
                    slot_names["FLEX"] = flex_pool[0]
                player_parts = [
                    slot_names.get("QB",""),
                    slot_names.get("RB1",""),
                    slot_names.get("RB2",""),
                    slot_names.get("WR1",""),
                    slot_names.get("WR2",""),
                    slot_names.get("WR3",""),
                    slot_names.get("TE",""),
                    slot_names.get("FLEX",""),
                    slot_names.get("DST",""),
                ]
                if self.use_contest_data:
                    roi_p = round(
                        x["ROI"] / self.entry_fee / self.num_iterations * 100, 2
                    )
                    roi_round = round(
                        x["ROI"] / x["Count"] / self.num_iterations, 2
                    )
                    extra_parts = [
                        fpts_p,
                        fieldFpts_p,
                        act_p,
                        ceil_p,
                        salary,
                        f"{win_p}%",
                        f"{top10_p}%",
                        f"{roi_p}%",
                        own_p,
                        roi_round,
                        primaryStack,
                        secondaryStack,
                        players_vs_def,
                        lu_type,
                        simDupes,
                    ]
                else:
                    extra_parts = [
                        fpts_p,
                        fieldFpts_p,
                        act_p,
                        ceil_p,
                        salary,
                        f"{win_p}%",
                        f"{top10_p}%",
                        f"{own_p}%",
                        primaryStack,
                        secondaryStack,
                        players_vs_def,
                        lu_type,
                        simDupes,
                    ]
                lineup_str = ",".join(map(str, player_parts + extra_parts))
            elif self.site == "fd":
                if self.use_contest_data:
                    roi_p = round(
                        x["ROI"] / x['Count'] / self.entry_fee / self.num_iterations * 100, 2
                    )
                    roi_round = round(x["ROI"] / x['Count'] / self.num_iterations, 2)
                    lineup_str = "{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{},{}%,{}%,{}%,{},${},{},{},{},{},{}".format(
                        lu_names[1].replace("#", "-"),
                        x["Lineup"][1],
                        lu_names[2].replace("#", "-"),
                        x["Lineup"][2],
                        lu_names[3].replace("#", "-"),
                        x["Lineup"][3],
                        lu_names[4].replace("#", "-"),
                        x["Lineup"][4],
                        lu_names[5].replace("#", "-"),
                        x["Lineup"][5],
                        lu_names[6].replace("#", "-"),
                        x["Lineup"][6],
                        lu_names[7].replace("#", "-"),
                        x["Lineup"][7],
                        lu_names[8].replace("#", "-"),
                        x["Lineup"][8],
                        lu_names[0].replace("#", "-"),
                        x["Lineup"][0],
                        fpts_p,
                        fieldFpts_p,
                        act_p,
                        ceil_p,
                        salary,
                        win_p,
                        top10_p,
                        roi_p,
                        own_p,
                        roi_round,
                        primaryStack,
                        secondaryStack,
                        players_vs_def,
                        lu_type,
                        simDupes
                    )
                else:
                    lineup_str = "{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{}:{},{},{},{},{},{},{}%,{}%,{},{},{},{},{},{}".format(
                        lu_names[1].replace("#", "-"),
                        x["Lineup"][1],
                        lu_names[2].replace("#", "-"),
                        x["Lineup"][2],
                        lu_names[3].replace("#", "-"),
                        x["Lineup"][3],
                        lu_names[4].replace("#", "-"),
                        x["Lineup"][4],
                        lu_names[5].replace("#", "-"),
                        x["Lineup"][5],
                        lu_names[6].replace("#", "-"),
                        x["Lineup"][6],
                        lu_names[7].replace("#", "-"),
                        x["Lineup"][7],
                        lu_names[8].replace("#", "-"),
                        x["Lineup"][8],
                        lu_names[0].replace("#", "-"),
                        x["Lineup"][0],
                        fpts_p,
                        fieldFpts_p,
                        act_p,
                        ceil_p,
                        salary,
                        win_p,
                        top10_p,
                        own_p,
                        primaryStack,
                        secondaryStack,
                        players_vs_def,
                        lu_type,
                        simDupes
                    )
            unique[index] = lineup_str

        lineups_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_gpp_sim_lineups_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(lineups_path, "w") as f:
            if self.site == "dk":
                if self.use_contest_data:
                    f.write(
                        "QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts Proj,Field Fpts Proj,Fpts Act,Ceiling,Salary,Win %,Top 10%,ROI%,Proj. Own. Product,Avg. Return,Stack1 Type,Stack2 Type,Players vs DST,Lineup Type, Sim Dupes\n"
                    )
                else:
                    f.write(
                        "QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts Proj,Field Fpts Proj,Fpts Act,Ceiling,Salary,Win %,Top 10%, Proj. Own. Product,Stack1 Type,Stack2 Type,Players vs DST,Lineup Type, Sim Dupes\n"
                    )
            elif self.site == "fd":
                if self.use_contest_data:
                    f.write(
                        "QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts Proj,Field Fpts Proj,Fpts Act,Ceiling,Salary,Win %,Top 10%,ROI%,Proj. Own. Product,Avg. Return,Stack1 Type,Stack2 Type,Players vs DST,Lineup Type, Sim Dupes\n"
                    )
                else:
                    f.write(
                        "QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Fpts Proj,Field Fpts Proj,Fpts Act,Ceiling,Salary,Win %,Top 10%,Proj. Own. Product,Stack1 Type,Stack2 Type,Players vs DST,Lineup Type, Sim Dupes\n"
                    )

            for fpts, lineup_str in unique.items():
                f.write("%s\n" % lineup_str)

        exposure_path = os.path.join(
            os.path.dirname(__file__),
            "../output/{}_gpp_sim_player_exposure_{}_{}.csv".format(
                self.site, self.field_size, self.num_iterations
            ),
        )
        with open(exposure_path, "w") as f:
            f.write(
                "Player,Position,Team,Fpts Act,Win%,Top1%,Sim. Own%,Proj. Own%,Avg. Return\n"
            )
            unique_players = {}
            for val in self.field_lineups.values():
                for player in val["Lineup"]:
                    if player not in unique_players:
                        unique_players[player] = {
                            "Wins": val["Wins"],
                            "Top1Percent": val["Top1Percent"],
                            "In": val['Count'],
                            "ROI": val["ROI"],
                        }
                    else:
                        unique_players[player]["Wins"] = (
                            unique_players[player]["Wins"] + val["Wins"]
                        )
                        unique_players[player]["Top1Percent"] = (
                            unique_players[player]["Top1Percent"] + val["Top1Percent"]
                        )
                        unique_players[player]["In"] = unique_players[player]["In"] + val['Count']
                        unique_players[player]["ROI"] = (
                            unique_players[player]["ROI"] + val["ROI"]
                        )
            top1PercentCount = (0.01) * self.field_size
            for player, data in unique_players.items():
                field_p = round(data["In"] / self.field_size * 100, 2)
                win_p = round(data["Wins"] / self.num_iterations * 100, 2)
                top10_p = round(
                    data["Top1Percent"] / top1PercentCount / self.num_iterations * 100, 2
                )
                roi_p = round(data["ROI"] / data["In"] / self.num_iterations, 2)
                for k, v in self.player_dict.items():
                    if player == v["ID"]:
                        proj_own = v["Ownership"]
                        p_name = v["Name"]
                        position = "/".join(v.get("Position"))
                        team = v.get("Team")
                        act_pts = v.get("ActPts", 0)
                        break
                f.write(
                    "{},{},{},{},{}%,{}%,{}%,{}%,${}\n".format(
                        p_name.replace("#", "-"),
                        position,
                        team,
                        act_pts,
                        win_p,
                        top10_p,
                        field_p,
                        proj_own,
                        roi_p,
                    )
                )

        stack_path = None
        if self.stack_exposure_df is not None:
            stack_path = os.path.join(
                os.path.dirname(__file__),
                "../output/{}_gpp_sim_stack_exposure_{}_{}.csv".format(
                    self.site, self.field_size, self.num_iterations
                ),
            )
            self.stack_exposure_df.to_csv(stack_path, index=False)

        return lineups_path, exposure_path, stack_path

    def _normalize_positions_in_tables(self):
        def _norm(p):
            p = str(p or "").upper().strip()
            return "DST" if p in ("D","DEF") else p
        try:
            if hasattr(self, "player_dict"):
                for _k, _rec in self.player_dict.items():
                    if isinstance(_rec, dict) and "Position" in _rec:
                        _rec["Position"] = _norm(_rec.get("Position"))
        except Exception:
            pass
