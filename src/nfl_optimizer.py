import json
import csv
import os
import datetime
import pytz
import timedelta
import numpy as np
import pandas as pd
import pulp as plp
import copy
import itertools
import re
import difflib

def _normalize_pos_key_and_value(rec):
    # Ensure dict has 'Position' key with normalized value (D/DEF/DS/D/ST->DST).
    def _norm(p):
        p = str(p or "").strip().upper()
        p = re.sub(r"[\[\]\"']", "", p)
        return "DST" if p in ("D","DEF","DS","D/ST","DST") else p
    # Source may have 'pos' or 'Position'
    if "Position" in rec:
        rec["Position"] = _norm(rec.get("Position"))
    elif "pos" in rec:
        rec["Position"] = _norm(rec.get("pos"))
    else:
        # Nothing present; leave as is
        pass
    return rec
from random import shuffle, choice

from utils import get_data_path, get_config_path
from selection_exposures import select_lineups, report_lineup_exposures
from stack_metrics import analyze_lineup
from player_ids_flex import load_player_ids_flex, dst_id_by_team, _norm_name


# --- begin Player dataclass (optimizer) ---
try:
    from dataclasses import dataclass
except Exception:
    dataclass = None

if dataclass is not None and "class Player(" not in globals():
    @dataclass
    class Player:
        name: str
        pos: str
        team: str
        salary: float
        proj: float
        ceil: float = 0.0
        stddev: float = 0.0
        own: float = 0.0
        id: int = 0
        key: str = ""
# --- end Player dataclass (optimizer) ---


class NFL_Optimizer:
    team_rename_dict = {"LA": "LAR"}

    def __init__(self, site=None, num_lineups=0, num_uniques=1, profile=None, pool_factor: float = 5.0):
        self.site = site
        self.config = None
        self.problem = None
        self.output_dir = None
        self.num_lineups = int(num_lineups)
        self.num_uniques = int(num_uniques)
        self.profile = profile
        self.pool_factor = float(pool_factor)
        # Instance-specific containers; these previously lived on the class
        # and caused state to leak across optimizer runs, requiring an app
        # restart after each optimization.
        self.team_list = []
        self.players_by_team = {}
        self.lineups = []
        self.player_dict = {}
        self.at_least = {}
        self.at_most = {}
        self.team_limits = {}
        self.matchup_limits = {}
        self.matchup_at_least = {}
        self.allow_qb_vs_dst = False
        self.stack_rules = {}
        self.global_team_limit = None
        self.use_double_te = True
        self.use_te_stack = True
        self.require_bring_back = True
        self.projection_minimum = 0
        self.randomness_amount = 0
        self.default_qb_var = 0.4
        self.default_skillpos_var = 0.5
        self.default_def_var = 0.5
        self.min_lineup_salary = 0
        self.stack_exposure_df = None

        self.load_config()
        # optional mapping of projection names -> player_ids names
        self.name_aliases = self.config.get("name_aliases", {})
        # Honor off-optimal floor from config (0..1). 0 disables.
        self.max_pct_off_optimal = float(self.config.get("max_pct_off_optimal", 0.0))
        if not (0.0 <= self.max_pct_off_optimal <= 1.0):
            self.max_pct_off_optimal = 0.0
        self.load_rules()

        self.problem = plp.LpProblem("NFL", plp.LpMaximize)

        projection_path = get_data_path(site, self.config["projection_path"])
        self.load_projections(projection_path)

        player_path = get_data_path(site, self.config["player_path"])
        self.load_player_ids(player_path)
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



        self.assertPlayerDict()
        # Fast lookup: var ID -> (name, pos, team)
        self.id_to_key = {rec["ID"]: key for key, rec in self.player_dict.items()}

    # === begin get_players helper ===
    def get_players(self, ids):
        """
        Map LP variable IDs -> Player objects using player_dict + id_to_key.
        """
        out = []
        # Build index lazily if needed
        if not hasattr(self, "id_to_key") or not self.id_to_key:
            try:
                self.id_to_key = {rec["ID"]: key for key, rec in self.player_dict.items()}
            except Exception:
                self.id_to_key = {}
        for _pid in ids:
            try:
                pid = int(str(_pid).strip())
            except Exception:
                continue
            key = self.id_to_key.get(pid)
            if key is None:
                # fallback: scan (slower, but safe)
                for k, rec in self.player_dict.items():
                    try:
                        if int(rec.get("ID", -1)) == pid:
                            key = k
                            break
                    except Exception:
                        pass
            if key is None:
                continue
            (name, pos_str, team) = key
            rec = self.player_dict.get(key, {})
            # safe fetches
            def _sf(x, d=0.0):
                try:
                    if x is None:
                        return float(d)
                    return float(x)
                except Exception:
                    return float(d)
            def _si(x, d=0):
                try:
                    return int(float(x))
                except Exception:
                    return int(d)

            pos = str(rec.get("Position", pos_str)).upper()
            team_abbrev = str(rec.get("TeamAbbrev", team)).upper()
            p = Player(
                name=str(name),
                pos=pos,
                team=team_abbrev,
                salary=_sf(rec.get("Salary", 0.0), 0.0),
                proj=_sf(rec.get("Fpts", 0.0), 0.0),
                ceil=_sf(rec.get("Ceil", rec.get("CEIL", 0.0)), 0.0),
                stddev=_sf(rec.get("StdDev", rec.get("STDDEV", 0.0)), 0.0),
                own=_sf(rec.get("Own", rec.get("OWN", 0.0)), 0.0),
                id=_si(rec.get("ID", pid), pid),
                key=str(rec.get("ID", pid)),
            )
            out.append(p)
        return out
    # === end get_players helper ===

    def select_slot_players(self, players):
        """Slot players into QB/RB/WR/TE/FLEX/DST based on position and salary."""
        normed = []
        for p in players:
            pos = str(getattr(p, "pos", "")).upper().strip()
            if pos in ("D", "DEF", "DS", "D/ST", "DST"):
                pos = "DST"
            p.pos = pos
            normed.append(p)

        by_pos = {"QB": [], "RB": [], "WR": [], "TE": [], "DST": []}
        for p in normed:
            if p.pos in by_pos:
                by_pos[p.pos].append(p)

        qb = by_pos["QB"][0] if by_pos["QB"] else None
        dst = by_pos["DST"][0] if by_pos["DST"] else None

        prio = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
        skill = [p for p in normed if p.pos != "DST"]
        skill.sort(key=lambda x: (prio.get(x.pos, 9), -float(getattr(x, "salary", 0.0) or 0.0), getattr(x, "name", "")))

        rb = [p for p in skill if p.pos == "RB"][:2]
        wr = [p for p in skill if p.pos == "WR"][:3]
        te = [p for p in skill if p.pos == "TE"][:1]

        used_ids = {id(x) for x in ([qb, dst] + rb + wr + te) if x is not None}
        flex = next((p for p in skill if id(p) not in used_ids), None)

        return {
            "QB": qb,
            "RB1": rb[0] if len(rb) > 0 else None,
            "RB2": rb[1] if len(rb) > 1 else None,
            "WR1": wr[0] if len(wr) > 0 else None,
            "WR2": wr[1] if len(wr) > 1 else None,
            "WR3": wr[2] if len(wr) > 2 else None,
            "TE": te[0] if len(te) > 0 else None,
            "FLEX": flex,
            "DST": dst,
        }

    def flatten(self, list):
        return [item for sublist in list for item in sublist]

    # make column lookups on datafiles case insensitive
    def lower_first(self, iterator):
        return itertools.chain([next(iterator).lower()], iterator)

    # Load config from file
    def load_config(self):
        config_path = get_config_path()
        with open(config_path) as json_file:
            self.config = json.load(json_file)

    # Load player IDs for exporting
    def load_player_ids(self, path="data/player_ids.csv"):
        """
        Load DraftKings player IDs from any of the supported schemas and build maps:
          - self.player_ids[(name_lower, position)] -> {ID, Position, TeamAbbrev}
          - self.player_ids_by_id[ID] -> {Name, Position, TeamAbbrev}
        Normalizes positions (D/DEF->DST). Stores the loaded df for later lookups.
        """
        import os
        import pandas as pd

        def _name_variants(canon: str, team: str):
            parts = canon.split()
            variants = [
                canon,
                canon.replace("-", "#"),
                canon.replace("-", " "),
                canon.replace("-", ""),
            ]
            if len(parts) >= 2:
                initial = parts[0][0]
                last = parts[-1]
                fi = f"{initial} {last}"
                variants.extend([
                    fi,
                    fi.replace("-", "#"),
                    fi.replace("-", " "),
                    fi.replace("-", ""),
                ])
                if team:
                    fi_team = f"{fi} {team}"
                    variants.extend([
                        fi_team,
                        fi_team.replace("-", "#"),
                        fi_team.replace("-", " "),
                        fi_team.replace("-", ""),
                    ])
            seen = set()
            out = []
            for v in variants:
                if v and v not in seen:
                    seen.add(v)
                    out.append(v)
            return out

        self.player_ids_path = path
        if not os.path.exists(path):
            # Fallback to repo data location if present
            alt = os.path.join(os.path.dirname(__file__), "..", "data", "player_ids.csv")
            if os.path.exists(alt):
                path = alt
            else:
                raise FileNotFoundError(f"player_ids file not found at {self.player_ids_path} or {alt}")

        df = load_player_ids_flex(path)
        self._player_ids_df = df.copy()

        # Canonical maps
        self.player_ids = {}
        for _, r in df.iterrows():
            name = str(r["Name"]).strip()
            pos = str(r["Position"]).strip().upper()
            pid = int(r["ID"])
            team = str(r.get("TeamAbbrev", "") or "").upper()
            canon = _norm_name(name)
            info = {"ID": pid, "Position": pos, "TeamAbbrev": team}
            for nk in _name_variants(canon, team):
                if (nk, pos) not in self.player_ids:
                    self.player_ids[(nk, pos)] = info
                else:
                    print(f"alias collision for {nk} at position {pos}")

        self.player_ids_by_id = {
            int(r["ID"]): {
                "Name": r["Name"],
                "Position": r["Position"],
                "TeamAbbrev": str(r.get("TeamAbbrev", "") or "").upper(),
            }
            for _, r in df.iterrows()
        }

        # Match loaded IDs onto existing player_dict entries
        for key, rec in self.player_dict.items():
            pos = str(rec.get("Position", "")).upper()
            if pos in ("D", "DEF"):
                pos = "DST"
                rec["Position"] = "DST"

            team = str(rec.get("TeamAbbrev") or rec.get("Team") or "").upper()
            name_canon = _norm_name(rec.get("Name"))
            name_variants = _name_variants(name_canon, team)
            info = None
            for nk in name_variants:
                info = self.player_ids.get((nk, pos))
                if info:
                    break

            if not info and getattr(self, "name_aliases", {}):
                for nk in [rec.get("Name")] + name_variants:
                    if nk in self.name_aliases:
                        alias_canon = _norm_name(self.name_aliases[nk])
                        alias_variants = _name_variants(alias_canon, team)
                        for ak in alias_variants:
                            if ak not in name_variants:
                                name_variants.append(ak)
                            info = self.player_ids.get((ak, pos))
                            if info:
                                break
                        if info:
                            break

            rec["_tried_aliases"] = name_variants
            if info:
                rec["ID"] = info["ID"]
                if not rec.get("TeamAbbrev"):
                    rec["TeamAbbrev"] = info.get("TeamAbbrev", "")
            else:
                # Capture potential matches to aid debugging
                last = re.sub(r"[^a-z]", "", rec.get("Name", "").split()[-1].lower())
                query = f"{last} {team}".strip()
                cand_map = {}
                for (nkey, p), inf in self.player_ids.items():
                    if p != pos:
                        continue
                    lname = nkey.split()[-1]
                    cand = f"{lname} {inf.get('TeamAbbrev', '')}".strip().lower()
                    cand_map[cand] = inf
                matches = difflib.get_close_matches(query, list(cand_map.keys()), n=5, cutoff=0.6)
                rec["_match_candidates"] = [
                    {
                        "name": self.player_ids_by_id[cand_map[m]["ID"]]["Name"],
                        "team": cand_map[m].get("TeamAbbrev", ""),
                        "id": cand_map[m]["ID"],
                    }
                    for m in matches
                ]

        return df

    def load_rules(self):
        self.at_most = self.config["at_most"]
        self.at_least = self.config["at_least"]
        self.team_limits = self.config["team_limits"]
        self.global_team_limit = int(self.config["global_team_limit"])
        self.projection_minimum = int(self.config["projection_minimum"])
        self.randomness_amount = float(self.config["randomness"])
        self.use_double_te = bool(self.config["use_double_te"])
        self.use_te_stack = bool(self.config.get("use_te_stack", True))
        self.require_bring_back = bool(self.config.get("require_bring_back", True))
        self.stack_rules = copy.deepcopy(self.config["stack_rules"])
        if not self.use_te_stack:
            for rule in self.stack_rules.get("pair", []):
                if rule.get("key") == "QB" and rule.get("type") == "same-team":
                    rule["positions"] = [
                        pos for pos in rule.get("positions", []) if pos != "TE"
                    ]
        if not self.require_bring_back:
            self.stack_rules["pair"] = [
                r
                for r in self.stack_rules.get("pair", [])
                if r.get("type") != "opp-team"
            ]
        self.matchup_at_least = self.config["matchup_at_least"]
        self.matchup_limits = self.config["matchup_limits"]
        self.allow_qb_vs_dst = bool(self.config["allow_qb_vs_dst"])
        self.min_lineup_salary = int(self.config.get("min_lineup_salary", 0))
        self.default_qb_var = (
            self.config["default_qb_var"] if "default_qb_var" in self.config else 0.333
        )
        self.default_skillpos_var = (
            self.config["default_skillpos_var"]
            if "default_skillpos_var" in self.config
            else 0.5
        )
        self.default_def_var = (
            self.config["default_def_var"] if "default_def_var" in self.config else 0.5
        )

    def assertPlayerDict(self):
        for p, s in list(self.player_dict.items()):
            if s["ID"] == 0 or s["ID"] == "" or s["ID"] is None:
                cands = s.get("_match_candidates", [])
                cand_str = "; ".join(
                    f"{c['name']} ({c['team']}) id:{c['id']}" for c in cands
                ) or "no close matches found"
                canon = _norm_name(s.get("Name"))
                tried = s.get("_tried_aliases", [])
                alias_str = ", ".join(tried) if tried else "none"
                print(
                    f"{s['Name']} (canon: {canon}) name mismatch between projections and player ids, "
                    "excluding from player_dict. Tried aliases: "
                    f"{alias_str}. Names are normalized by stripping periods, "
                    "suffixes (jr/sr/ii/iii/iv/v), trailing roman numerals, and "
                    "lower-casing. Hyphens may appear as '-', '#', spaces, or be removed. "
                    "Ensure team abbreviations and positions match between files. "
                    f"Potential matches: {cand_str}"
                )
                player, pos, team = p
                if team in self.players_by_team and pos in self.players_by_team[team]:
                    try:
                        self.players_by_team[team][pos].remove(s)
                    except ValueError:
                        pass
                self.player_dict.pop(p)

    # Load projections from file
    def load_projections(self, path):
        # Read projections into a dictionary
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                canon = _norm_name(row["name"])
                if getattr(self, "name_aliases", {}):
                    alias = None
                    for lk in (row["name"], canon, canon.replace("-", "#")):
                        if lk in self.name_aliases:
                            alias = self.name_aliases[lk]
                            break
                    if alias:
                        canon = _norm_name(alias)
                player_name = canon.replace("-", "#")
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
                position = row["pos"]
                if position == "D" or position == "DEF":
                    position = "DST"

                team = row["team"]
                if team in self.team_rename_dict:
                    team = self.team_rename_dict[team]

                if team == "JAX" and self.site == "fd":
                    team = "JAC"

                opp = row.get("opp", "")
                if opp in self.team_rename_dict:
                    opp = self.team_rename_dict[opp]

                if opp == "JAX" and self.site == "fd":
                    opp = "JAC"

                matchup = f"{team} @ {opp}" if opp else ""
                if "fantasyyear_consistency" in row:
                    if row["fantasyyear_consistency"] == "" or float(row["fantasyyear_consistency"]) == 0:
                        if position == "QB":
                            stddev = fpts * self.default_qb_var
                        elif position == "DST":
                            stddev = fpts * self.default_def_var
                        else:
                            stddev = fpts * self.default_skillpos_var
                    else:
                        stddev = float(row["fantasyyear_consistency"])
                else:
                    if position == "QB":
                        stddev = fpts * self.default_qb_var
                    elif position == "DST":
                        stddev = fpts * self.default_def_var
                    else:
                        stddev = fpts * self.default_skillpos_var
                if "ceiling" in row:
                    if row["ceiling"] == "" or float(row["ceiling"]) == 0:
                        ceil = fpts + stddev
                    else:
                        ceil = float(row["ceiling"])
                else:
                    ceil = fpts + stddev
                own = float(row["projections_projown"]) if row["projections_projown"] != "" else 0
                if own == 0:
                    own = 0.1
                if (float(row["projections_proj"]) < self.projection_minimum) and position != "DST":
                    continue
                self.player_dict[(player_name, position, team)] = {
                    "Fpts": fpts,
                    "ActPts": actpts,
                    "Position": position,
                    "ID": 0,
                    "Salary": int(row["salary"].replace(",", "")),
                    "Name": row["name"],
                    "Matchup": matchup,
                    "Team": team,
                    "Opponent": opp,
                    "Ownership": own,
                    "Ceiling": ceil,
                    "StdDev": stddev,
                }

                if team not in self.team_list:
                    self.team_list.append(team)

                if team not in self.players_by_team:
                    self.players_by_team[team] = {
                        "QB": [],
                        "RB": [],
                        "WR": [],
                        "TE": [],
                        "DST": [],
                    }

                self.players_by_team[team][position].append(
                    self.player_dict[(player_name, position, team)]
                )

    def optimize(self, progress_callback=None):
        # Setup our linear programming equation - https://en.wikipedia.org/wiki/Linear_programming
        # We will use PuLP as our solver - https://coin-or.github.io/pulp/

        # We want to create a variable for each roster slot.
        # There will be an index for each player and the variable will be binary (0 or 1) representing whether the player is included or excluded from the roster.
        lp_variables = {
            self.player_dict[(player, pos_str, team)]["ID"]: plp.LpVariable(
                str(self.player_dict[(player, pos_str, team)]["ID"]), cat="Binary"
            )
            for (player, pos_str, team) in self.player_dict
        }

        # set the objective - maximize fpts & set randomness amount from config
        if self.randomness_amount != 0:
            self.problem += (
                plp.lpSum(
                    np.random.normal(
                        self.player_dict[(player, pos_str, team)]["Fpts"],
                        (
                            self.player_dict[(player, pos_str, team)]["StdDev"]
                            * self.randomness_amount
                            / 100
                        ),
                    )
                    * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                ),
                "Objective",
            )
        else:
            self.problem += (
                plp.lpSum(
                    self.player_dict[(player, pos_str, team)]["Fpts"]
                    * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                ),
                "Objective",
            )

        # Set the salary constraints
        max_salary = 50000 if self.site == "dk" else 60000
        min_salary = (
            self.min_lineup_salary
            if self.min_lineup_salary
            else (45000 if self.site == "dk" else 55000)
        )
        self.problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["Salary"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
            )
            <= max_salary,
            "Max Salary",
        )
        self.problem += (
            plp.lpSum(
                self.player_dict[(player, pos_str, team)]["Salary"]
                * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
            )
            >= min_salary,
            "Min Salary",
        )

        # Address limit rules if any
        for limit, groups in self.at_least.items():
            for group in groups:
                tuple_name_list = []
                for key, value in self.player_dict.items():
                    if value["Name"] in group:
                        tuple_name_list.append(key)

                self.problem += (
                    plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in tuple_name_list
                    )
                    >= int(limit),
                    f"At least {limit} players {tuple_name_list}",
                )

        for limit, groups in self.at_most.items():
            for group in groups:
                tuple_name_list = []
                for key, value in self.player_dict.items():
                    if value["Name"] in group:
                        tuple_name_list.append(key)

                self.problem += (
                    plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in tuple_name_list
                    )
                    <= int(limit),
                    f"At most {limit} players {tuple_name_list}",
                )

        # Address team limits
        for teamIdent, limit in self.team_limits.items():
            self.problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if self.player_dict[(player, pos_str, team)]["Team"] == teamIdent
                )
                <= int(limit),
                f"Team limit {teamIdent} {limit}",
            )

        if self.global_team_limit is not None:
            team_limit = int(self.global_team_limit)
        else:
            team_limit = 5 if self.site == "dk" else 4

        for limit_team in self.team_list:
            self.problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if self.player_dict[(player, pos_str, team)]["Team"] == limit_team
                )
                <= team_limit,
                f"Global team limit {limit_team} {team_limit}",
            )

        # Address matchup limits
        if self.matchup_limits is not None:
            for matchup, limit in self.matchup_limits.items():
                players_in_game = []
                for key, value in self.player_dict.items():
                    if value["Matchup"] == matchup:
                        players_in_game.append(key)
                self.problem += (
                    plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in players_in_game
                    )
                    <= int(limit),
                    f"Matchup limit {matchup} {limit}",
                )

        if self.matchup_at_least is not None:
            for matchup, limit in self.matchup_at_least.items():
                players_in_game = []
                for key, value in self.player_dict.items():
                    if value["Matchup"] == matchup:
                        players_in_game.append(key)

                self.problem += (
                    plp.lpSum(
                        lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in players_in_game
                    )
                    >= int(limit),
                    f"Matchup at least {matchup} {limit}",
                )

        # Address player vs dst (only applies to QB vs DST)
        if not self.allow_qb_vs_dst:
            for team, players in self.players_by_team.items():
                for qb in players.get("QB", []):
                    opponent = qb.get("Opponent")
                    if opponent is None:
                        continue
                    opposing_dsts = self.players_by_team.get(opponent, {}).get(
                        "DST", []
                    )
                    for dst in opposing_dsts:
                        self.problem += (
                            plp.lpSum(lp_variables[qb["ID"]] + lp_variables[dst["ID"]])
                            <= 1,
                            f"No QB vs DST {qb['Name']} vs {dst['Name']}",
                        )

                # self.problem += (
                #     plp.lpSum(
                #         lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                #         for (player, pos_str, team) in players_in_game
                #     )
                #     >= int(limit),
                #     f"Matchup at least {matchup} {limit}",
                # )

        # Address stack rules
        for rule_type in self.stack_rules:
            for rule in self.stack_rules[rule_type]:
                if rule_type == "pair":
                    pos_key = rule["key"]
                    stack_positions = rule["positions"]
                    count = rule["count"]
                    stack_type = rule["type"]
                    excluded_teams = rule["exclude_teams"]

                    # Iterate each team, less excluded teams, and apply the rule for each key player pos
                    for team in self.players_by_team:
                        if team in excluded_teams:
                            continue

                        pos_key_players = self.players_by_team[team][pos_key]
                        if len(pos_key_players) == 0:
                            continue

                        opp_team = pos_key_players[0]["Opponent"]

                        if (
                            stack_type in ["opp-team", "same-game"]
                            and opp_team not in self.players_by_team
                        ):
                            continue

                        for pos_key_player in pos_key_players:
                            stack_players = []
                            if stack_type == "same-team":
                                for pos in stack_positions:
                                    stack_players.append(
                                        self.players_by_team[team][pos]
                                    )

                            elif stack_type == "opp-team":
                                for pos in stack_positions:
                                    stack_players.append(
                                        self.players_by_team[opp_team][pos]
                                    )

                            elif stack_type == "same-game":
                                for pos in stack_positions:
                                    stack_players.append(
                                        self.players_by_team[team][pos]
                                    )
                                    stack_players.append(
                                        self.players_by_team[opp_team][pos]
                                    )

                            stack_players = self.flatten(stack_players)
                            # player cannot exist as both pos_key_player and be present in the stack_players
                            stack_players = [
                                p
                                for p in stack_players
                                if not (
                                    p["Name"] == pos_key_player["Name"]
                                    and p["Position"] == pos_key_player["Position"]
                                    and p["Team"] == pos_key_player["Team"]
                                )
                            ]
                            pos_key_player_tuple = None
                            stack_players_tuples = []
                            for key, value in self.player_dict.items():
                                if (
                                    value["Name"] == pos_key_player["Name"]
                                    and value["Position"] == pos_key_player["Position"]
                                    and value["Team"] == pos_key_player["Team"]
                                ):
                                    pos_key_player_tuple = key
                                elif (
                                    value["Name"],
                                    value["Position"],
                                    value["Team"],
                                ) in [
                                    (player["Name"], player["Position"], player["Team"])
                                    for player in stack_players
                                ]:
                                    stack_players_tuples.append(key)

                            if (
                                pos_key_player_tuple is None
                                or len(stack_players_tuples) == 0
                            ):
                                continue
                            # [sum of stackable players] + -n*[stack_player] >= 0
                            self.problem += (
                                plp.lpSum(
                                    [
                                        lp_variables[
                                            self.player_dict[(player, pos_str, team)][
                                                "ID"
                                            ]
                                        ]
                                        for (
                                            player,
                                            pos_str,
                                            team,
                                        ) in stack_players_tuples
                                    ]
                                    + [
                                        -count
                                        * lp_variables[
                                            self.player_dict[pos_key_player_tuple]["ID"]
                                        ]
                                    ]
                                )
                                >= 0,
                                f"Stack rule {pos_key_player_tuple} {stack_players_tuples} {count}",
                            )

                elif rule_type == "limit":
                    limit_positions = rule["positions"]  # ["RB"]
                    stack_type = rule["type"]
                    count = rule["count"]
                    excluded_teams = rule["exclude_teams"]
                    if "unless_positions" in rule or "unless_type" in rule:
                        unless_positions = rule["unless_positions"]
                        unless_type = rule["unless_type"]
                    else:
                        unless_positions = None
                        unless_type = None

                    # Iterate each team, less excluded teams, and apply the rule for each key player pos
                    for team in self.players_by_team:
                        opp_team = self.players_by_team[team]["QB"]

                        if len(opp_team) == 0:
                            continue

                        opp_team = opp_team[0]["Opponent"]
                        if team in excluded_teams:
                            continue

                        if (
                            stack_type in ["opp-team", "same-game"]
                            and opp_team not in self.players_by_team
                        ):
                            continue

                        limit_players = []
                        if stack_type == "same-team":
                            for pos in limit_positions:
                                limit_players.append(self.players_by_team[team][pos])

                        elif stack_type == "opp-team":
                            for pos in limit_positions:
                                limit_players.append(
                                    self.players_by_team[opp_team][pos]
                                )

                        elif stack_type == "same-game":
                            for pos in limit_positions:
                                limit_players.append(self.players_by_team[team][pos])
                                limit_players.append(
                                    self.players_by_team[opp_team][pos]
                                )

                        limit_players = self.flatten(limit_players)
                        if unless_positions is None or unless_type is None:
                            # [sum of limit players] + <= n
                            limit_players_tuples = []
                            for key, value in self.player_dict.items():
                                if (
                                    value["Name"],
                                    value["Position"],
                                    value["Team"],
                                ) in [
                                    (player["Name"], player["Position"], player["Team"])
                                    for player in limit_players
                                ]:
                                    limit_players_tuples.append(key)

                            if len(limit_players_tuples) == 0:
                                continue
                            self.problem += (
                                plp.lpSum(
                                    [
                                        lp_variables[
                                            self.player_dict[(player, pos_str, team)][
                                                "ID"
                                            ]
                                        ]
                                        for (
                                            player,
                                            pos_str,
                                            team,
                                        ) in limit_players_tuples
                                    ]
                                )
                                <= int(count),
                                f"Limit rule {limit_players_tuples} {count}",
                            )
                        else:
                            unless_players = []
                            if unless_type == "same-team":
                                for pos in unless_positions:
                                    unless_players.append(
                                        self.players_by_team[team][pos]
                                    )
                            elif unless_type == "opp-team":
                                for pos in unless_positions:
                                    if opp_team in self.players_by_team:
                                        unless_players.append(
                                            self.players_by_team[opp_team][pos]
                                        )
                            elif unless_type == "same-game":
                                for pos in unless_positions:
                                    unless_players.append(
                                        self.players_by_team[team][pos]
                                    )
                                    if opp_team in self.players_by_team:
                                        unless_players.append(
                                            self.players_by_team[opp_team][pos]
                                        )

                            unless_players = self.flatten(unless_players)

                            # player cannot exist as both limit_players and unless_players
                            unless_players = [
                                p
                                for p in unless_players
                                if not any(
                                    p["Name"] == key_player["Name"]
                                    and p["Position"] == key_player["Position"]
                                    and p["Team"] == key_player["Team"]
                                    for key_player in limit_players
                                )
                            ]

                            limit_players_tuples = []
                            unless_players_tuples = []
                            for key, value in self.player_dict.items():
                                if (
                                    value["Name"],
                                    value["Position"],
                                    value["Team"],
                                ) in [
                                    (player["Name"], player["Position"], player["Team"])
                                    for player in limit_players
                                ]:
                                    limit_players_tuples.append(key)
                                elif (
                                    value["Name"],
                                    value["Position"],
                                    value["Team"],
                                ) in [
                                    (player["Name"], player["Position"], player["Team"])
                                    for player in unless_players
                                ]:
                                    unless_players_tuples.append(key)

                            # [sum of limit players] + -count(unless_players)*[unless_players] <= n
                            if len(limit_players_tuples) == 0:
                                continue
                            self.problem += (
                                plp.lpSum(
                                    [
                                        lp_variables[
                                            self.player_dict[(player, pos_str, team)][
                                                "ID"
                                            ]
                                        ]
                                        for (
                                            player,
                                            pos_str,
                                            team,
                                        ) in limit_players_tuples
                                    ]
                                    - int(count)
                                    * plp.lpSum(
                                        [
                                            lp_variables[
                                                self.player_dict[
                                                    (player, pos_str, team)
                                                ]["ID"]
                                            ]
                                            for (
                                                player,
                                                pos_str,
                                                team,
                                            ) in unless_players_tuples
                                        ]
                                    )
                                )
                                <= int(count),
                                f"Limit rule {limit_players_tuples} unless {unless_players_tuples} {count}",
                            )

        # Need exactly 1 QB
        self.problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
                if "QB" == self.player_dict[(player, pos_str, team)]["Position"]
            )
            == 1,
            f"QB limit 1",
        )

        # Need at least 2 RB, up to 3 if using FLEX
        self.problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
                if "RB" == self.player_dict[(player, pos_str, team)]["Position"]
            )
            >= 2,
            f"RB >= 2",
        )
        self.problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
                if "RB" == self.player_dict[(player, pos_str, team)]["Position"]
            )
            <= 3,
            f"RB <= 3",
        )

        # Need at least 3 WR, up to 4 if using FLEX
        self.problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
                if "WR" == self.player_dict[(player, pos_str, team)]["Position"]
            )
            >= 3,
            f"WR >= 3",
        )
        self.problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
                if "WR" == self.player_dict[(player, pos_str, team)]["Position"]
            )
            <= 4,
            f"WR <= 4",
        )

        # Need at least 1 TE, up to 2 if using FLEX
        if self.use_double_te:
            self.problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" == self.player_dict[(player, pos_str, team)]["Position"]
                )
                >= 1,
                f"TE >= 1",
            )
            self.problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" == self.player_dict[(player, pos_str, team)]["Position"]
                )
                <= 2,
                f"TE <= 2",
            )
        else:
            self.problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                    for (player, pos_str, team) in self.player_dict
                    if "TE" == self.player_dict[(player, pos_str, team)]["Position"]
                )
                == 1,
                f"TE == 1",
            )

        # Need exactly 1 DST
        self.problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
                if "DST" == self.player_dict[(player, pos_str, team)]["Position"]
            )
            == 1,
            f"DST == 1",
        )

        # Can only roster 9 total players
        self.problem += (
            plp.lpSum(
                lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                for (player, pos_str, team) in self.player_dict
            )
            == 9,
            f"Total Players == 9",
        )
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
                # Persist for visibility/debug
                self.optimal_fpts = _opt
                self.min_fpts_floor = _min_fpts
                # Add a hard floor constraint based on *deterministic* projections,
                # so it remains valid even when we randomize the objective later.
                self.problem += (_det_fpts_sum >= _min_fpts), "MinFptsOffOptimal"
        # --- End: enforce off-optimal floor ---
    
        # Crunch!

        # --- Optimizer cap/floor info ---
        try:
            import streamlit as st
            st.info(
                f"Optimizer: site={self.site}, cap={50000 if self.site=='dk' else 60000}, "
                f"min_floor={self.min_lineup_salary or (45000 if self.site=='dk' else 55000)}"
            )
        except Exception:
            pass
        num_pool = max(int(self.num_lineups * self.pool_factor), self.num_lineups)
        for i in range(num_pool):
            try:
                self.problem.solve(plp.PULP_CBC_CMD(msg=0))
            except plp.PulpSolverError:
                print(
                    "Infeasibility reached - only generated {} lineups out of {}. Continuing with export.".format(
                        len(self.num_lineups), self.num_lineups
                    )
                )

            # Get the lineup and add it to our list
            player_ids = [
                player for player in lp_variables if lp_variables[player].varValue != 0
            ]
            players = []
            for key, value in self.player_dict.items():
                if value["ID"] in player_ids:
                    players.append(key)

            # --- Salary bounds guard ---
            # Convert chosen variable IDs -> Player objects, then slot to nine
            players_ids = player_ids
            players_objs = self.get_players(players_ids)
            slots = self.select_slot_players(players_objs)
            nine = [slots["QB"], slots["RB1"], slots["RB2"], slots["WR1"], slots["WR2"], slots["WR3"], slots["TE"], slots["FLEX"], slots["DST"]]

            # Deterministic totals based on the *nine* that will be exported
            det_proj   = sum(p.proj   for p in nine)
            det_salary = sum(p.salary for p in nine)

            # Active cap/floor (match LP constraints)
            max_salary = 50000 if self.site == "dk" else 60000
            min_salary = self.min_lineup_salary if self.min_lineup_salary else (45000 if self.site == "dk" else 55000)

            # Enforce at runtime (tiny epsilon for float safety)
            if det_salary > max_salary + 1e-6 or det_salary < min_salary - 1e-6:
                raise AssertionError(
                    f"Lineup salary {det_salary} out of bounds "
                    f"(site={self.site}, cap={max_salary}, floor={min_salary}). "
                    f"QB={nine[0].name}, RB1={nine[1].name}, RB2={nine[2].name}, "
                    f"WR1={nine[3].name}, WR2={nine[4].name}, WR3={nine[5].name}, "
                    f"TE={nine[6].name}, FLEX={nine[7].name}, DST={nine[8].name}"
                )

            # Store lineup with deterministic projection (same metric as constraints)
            self.lineups.append((players, det_proj))

            progress = i + 1
            percent = (progress / num_pool) * 100



            # Ensure this lineup isn't picked again
            self.problem += (
                plp.lpSum(
                    lp_variables[self.player_dict[player]["ID"]] for player in players
                )
                <= len(players) - self.num_uniques,
                f"Lineup {i}",
            )

            # Set a new random fpts projection within their distribution
            if self.randomness_amount != 0:
                self.problem += (
                    plp.lpSum(
                        np.random.normal(
                            self.player_dict[(player, pos_str, team)]["Fpts"],
                            (
                                self.player_dict[(player, pos_str, team)]["StdDev"]
                                * self.randomness_amount
                                / 100
                            ),
                        )
                        * lp_variables[self.player_dict[(player, pos_str, team)]["ID"]]
                        for (player, pos_str, team) in self.player_dict
                    ),
                    "Objective",
                )

        if self.profile:
            profiles = self.config.get("profiles", {})
            prof = profiles.get(self.profile)
            if prof:
                targets = {
                    "presence_targets_pct": prof.get("presence_targets_pct", {}),
                    "multiplicity_targets_mean": prof.get("multiplicity_targets_mean", {}),
                    "bucket_mix_pct": prof.get("bucket_mix_pct", {}),
                }
                candidate_players = [players for players, _ in self.lineups]
                selected_players = select_lineups(
                    candidate_players, self.player_dict, targets, self.num_lineups
                )
                selected_set = {tuple(lp) for lp in selected_players}
                self.lineups = [
                    (players, fpts)
                    for (players, fpts) in self.lineups
                    if tuple(players) in selected_set
                ]

            else:
                print(f"Warning: profile {self.profile} not found in config")
        else:
            # truncate pool to requested number of lineups
            self.lineups = self.lineups[: self.num_lineups]

    def output(self):
        print("Lineups done generating. Outputting.")
        # Normalize Position field in player_dict (D/DEF -> DST) before ordering
        for _k, _rec in self.player_dict.items():
            p = str(_rec.get("Position","")).upper()
            if p in ("D","DEF"):
                _rec["Position"] = "DST"



        sorted_lineups = []
        for lineup, fpts_used in self.lineups:
            sorted_lineup = self.sort_lineup(lineup)
            sorted_lineups.append((sorted_lineup, fpts_used))

        # Aggregate stack metrics across all generated lineups
        report = report_lineup_exposures(
            [lu for lu, _ in sorted_lineups], self.player_dict, self.config
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

        team_stack_counts = {}

        formatted_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_out = (
            f"../output/{self.site}_optimal_lineups_{formatted_timestamp}.csv"
        )
        out_path = os.path.join(os.path.dirname(__file__), filename_out)

        # BEGIN: patched CSV writer
        from lineup_writer_patch import Player, write_lineup_csv

        def _to_player(key):
            data = self.player_dict[key]
            name = (
                f"{data['Name']} ({data['ID']})"
                if self.site == "dk"
                else f"{data['ID']}:{data['Name']}"
            )
            pos = data["Position"]
            if pos in ("D", "DEF"):
                pos = "DST"
            pl = Player(
                name=name,
                pos=pos,
                team=data.get("Team", ""),
                salary=float(data["Salary"]),
                proj=float(data["Fpts"]),
                act=float(data.get("ActPts", 0.0)),
                ceil=float(data.get("Ceiling", 0.0)),
                own=float(data.get("Ownership", 0.0)),
                stddev=float(data.get("StdDev", 0.0)),
            )
            pl.key = key
            pl.opp = data.get("Opponent", "")
            return pl

        all_lineups_as_players = [[_to_player(p) for p in x] for x, _ in sorted_lineups]

        def _stddev_fn(L):
            return sum(getattr(p, "stddev", 0.0) for p in L)

        def _players_vs_dst_fn(L):
            dst_opponents = {
                getattr(p, "opp", "")
                for p in L
                if (p.pos or "").upper() == "DST" and getattr(p, "opp", "")
            }
            return sum(
                1
                for p in L
                if (p.pos or "").upper() != "DST" and p.team in dst_opponents
            )

        def _stack_str_fn(L):
            return self.construct_stack_string([getattr(p, "key") for p in L])
        # Surface optimal and floor for downstream reporting
        if hasattr(self, "optimal_fpts"):
            os.environ["OPTIMAL_FPTS"] = f"{self.optimal_fpts:.4f}"
        if hasattr(self, "min_fpts_floor"):
            os.environ["MIN_FPTS_FLOOR"] = f"{self.min_fpts_floor:.4f}"

        # BEGIN: pass site/cap/floor to writer for independent verification
        import os as _os
        _cap  = 50000 if self.site == "dk" else 60000
        _floor = self.min_lineup_salary if self.min_lineup_salary else (45000 if self.site == "dk" else 55000)
        _os.environ["OPT_SITE"] = str(self.site)
        _os.environ["OPT_CAP"] = str(int(_cap))
        _os.environ["OPT_FLOOR"] = str(int(_floor))
        # END: pass site/cap/floor to writer

        write_lineup_csv(
            all_lineups_as_players,
            out_path=out_path,
            stddev_fn=_stddev_fn,
            players_vs_dst_fn=_players_vs_dst_fn,
            stack_str_fn=_stack_str_fn,
        )
        # END: patched CSV writer

        stack_path = None
        if self.stack_exposure_df is not None:
            stack_filename = (
                f"../output/{self.site}_stack_exposure_{formatted_timestamp}.csv"
            )
            stack_path = os.path.join(os.path.dirname(__file__), stack_filename)
            self.stack_exposure_df.to_csv(stack_path, index=False)

        print("Output done.")
        return out_path, stack_path

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
    def construct_stack_string(self, lineup):
        metrics = analyze_lineup(lineup, self.player_dict)
        parts = []
        for k, v in metrics["counts"].items():
            if v > 0 and k != "No Stack":
                if v > 1:
                    parts.append(f"{k} x{v}")
                else:
                    parts.append(k)
        if not parts and metrics["presence"].get("No Stack"):
            parts.append("No Stack")
        return " ; ".join(parts)
