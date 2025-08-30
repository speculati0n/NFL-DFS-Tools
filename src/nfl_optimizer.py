import json
import csv
import os
import datetime
import pytz
import timedelta
import numpy as np
import pulp as plp
import copy
import itertools
from random import shuffle, choice
from collections import Counter

from utils import get_data_path, get_config_path
from selection_exposures import select_lineups
from stack_metrics import analyze_lineup


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

        self.load_config()
        self.load_rules()

        self.problem = plp.LpProblem("NFL", plp.LpMaximize)

        projection_path = get_data_path(site, self.config["projection_path"])
        self.load_projections(projection_path)

        player_path = get_data_path(site, self.config["player_path"])
        self.load_player_ids(player_path)
        self.assertPlayerDict()

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
    def load_player_ids(self, path):
        with open(path, encoding="utf-8-sig") as file:
            reader = csv.DictReader(self.lower_first(file))
            for row in reader:
                if self.site == "dk":
                    position = row["position"]
                    team = row.get("team") or row.get("teamabbrev")
                    names = set()
                    for col in ["displayname", "firstname", "lastname", "shortname"]:
                        val = row.get(col)
                        if val:
                            names.add(val)
                    if row.get("firstname") and row.get("lastname"):
                        names.add(f"{row['firstname']} {row['lastname']}")
                    matched = False
                    for name in names:
                        player_name = name.replace("-", "#").lower().strip()
                        if team:
                            key = (player_name, position, team)
                            if key in self.player_dict:
                                self.player_dict[key]["ID"] = int(row["draftableid"])
                                if not self.player_dict[key].get("Matchup"):
                                    self.player_dict[key]["Matchup"] = row.get("start_date", "")
                                matched = True
                                break
                        else:
                            for key in list(self.player_dict.keys()):
                                pname, ppos, pteam = key
                                if pname == player_name and ppos == position:
                                    self.player_dict[key]["ID"] = int(row["draftableid"])
                                    if not self.player_dict[key].get("Matchup"):
                                        self.player_dict[key]["Matchup"] = row.get("start_date", "")
                                    matched = True
                                    break
                        if matched:
                            break
                else:
                    position = row["position"]
                    if position == "D":
                        position = "DST"
                    team = row["team"]
                    names = set()
                    for col in ["nickname", "displayname", "firstname", "lastname", "shortname"]:
                        val = row.get(col)
                        if val:
                            names.add(val)
                    if row.get("firstname") and row.get("lastname"):
                        names.add(f"{row['firstname']} {row['lastname']}")
                    for name in names:
                        player_name = name.replace("-", "#").lower().strip()
                        key = (player_name, position, team)
                        if key in self.player_dict:
                            matchup = row.get("game", "")
                            opponent = row.get("opponent", "")
                            self.player_dict[key]["Opponent"] = opponent
                            self.player_dict[key]["Matchup"] = matchup
                            self.player_dict[key]["ID"] = row.get("id", "")
                            break

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
                print(
                    s["Name"]
                    + " name mismatch between projections and player ids, excluding from player_dict"
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
                player_name = row["name"].replace("-", "#").lower().strip()
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
                if (
                    float(row["projections_proj"]) < self.projection_minimum
                    and row["pos"] != "DST"
                ):
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

    def optimize(self):
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

        # Crunch!
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

            fpts_used = self.problem.objective.value()
            self.lineups.append((players, fpts_used))

            if i % 100 == 0:
                print(i)

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
                presence_tot = Counter()
                mult_tot = Counter()
                bucket_tot = Counter()
                for lineup in selected_players:
                    metrics = analyze_lineup(lineup, self.player_dict)
                    presence_tot.update(metrics["presence"])
                    mult_tot.update(metrics["counts"])
                    bucket_tot[metrics["bucket"]] += 1
                n = len(selected_players)
                print("Exposure Results:")
                for k, t in targets.get("presence_targets_pct", {}).items():
                    ach = presence_tot.get(k, 0) / n if n else 0
                    print(f"Presence {k}: {ach:.2f} (target {t:.2f})")
                for k, t in targets.get("multiplicity_targets_mean", {}).items():
                    ach = mult_tot.get(k, 0) / n if n else 0
                    print(f"Multiplicity {k}: {ach:.2f} (target {t:.2f})")
                for k, t in targets.get("bucket_mix_pct", {}).items():
                    ach = bucket_tot.get(k, 0) / n if n else 0
                    print(f"Bucket {k}: {ach:.2f} (target {t:.2f})")
            else:
                print(f"Warning: profile {self.profile} not found in config")
        else:
            # truncate pool to requested number of lineups
            self.lineups = self.lineups[: self.num_lineups]

    def output(self):
        print("Lineups done generating. Outputting.")

        sorted_lineups = []
        for lineup, fpts_used in self.lineups:
            sorted_lineup = self.sort_lineup(lineup)
            sorted_lineups.append((sorted_lineup, fpts_used))

        team_stack_counts = {}

        formatted_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_out = (
            f"../output/{self.site}_optimal_lineups_{formatted_timestamp}.csv"
        )
        out_path = os.path.join(os.path.dirname(__file__), filename_out)
        with open(out_path, "w") as f:
            f.write(
                "QB,RB,RB,WR,WR,WR,TE,FLEX,DST,Salary,Fpts Proj,Fpts Used,Fpts Act,Ceiling,Own. Sum,Own. Product,STDDEV,Stack\n"
            )
            for x, fpts_used in sorted_lineups:
                stack_str = self.construct_stack_string(x)

                salary = sum(self.player_dict[player]["Salary"] for player in x)
                fpts_p = sum(self.player_dict[player]["Fpts"] for player in x)
                act_p = sum(self.player_dict[player].get("ActPts", 0) for player in x)
                own_s = sum(self.player_dict[player]["Ownership"] for player in x)
                own_p = np.prod(
                    [self.player_dict[player]["Ownership"] / 100 for player in x]
                )
                ceil = sum([self.player_dict[player]["Ceiling"] for player in x])
                stddev = sum([self.player_dict[player]["StdDev"] for player in x])
                if self.site == "dk":
                    player_fields = [
                        f"{self.player_dict[p]['Name']} ({self.player_dict[p]['ID']})" for p in x
                    ]
                else:
                    player_fields = [
                        f"{self.player_dict[p]['ID']}:{self.player_dict[p]['Name']}" for p in x
                    ]
                fields = player_fields + [
                    salary,
                    round(fpts_p, 2),
                    round(fpts_used, 2),
                    round(act_p, 2),
                    ceil,
                    own_s,
                    own_p,
                    stddev,
                    stack_str,
                ]
                lineup_str = ",".join(map(str, fields))
                f.write(f"{lineup_str}\n")

        print("Output done.")
        return out_path

    def sort_lineup(self, lineup):
        copy_lineup = copy.deepcopy(lineup)
        positional_order = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DST"]
        final_lineup = []

        # Sort players based on their positional order
        for position in positional_order:
            if position != "FLEX":
                eligible_players = [
                    player
                    for player in copy_lineup
                    if self.player_dict[player]["Position"] == position
                ]
                if eligible_players:
                    eligible_player = eligible_players[0]
                    final_lineup.append(eligible_player)
                    copy_lineup.remove(eligible_player)
                else:
                    print(f"No players found with position: {position}")
                    # Handle the case here (perhaps append a placeholder or skip appending)
            else:
                eligible_players = [
                    player
                    for player in copy_lineup
                    if self.player_dict[player]["Position"] in ["RB", "WR", "TE"]
                ]
                if eligible_players:
                    eligible_player = eligible_players[0]
                    final_lineup.append(eligible_player)
                    copy_lineup.remove(eligible_player)
                else:
                    print(f"No players found for FLEX position")
                    # Handle the case here (perhaps append a placeholder or skip appending)
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
