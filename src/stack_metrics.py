from typing import Dict, List
from collections import defaultdict, Counter
import itertools


def detect_presence(lineup: List[str], player_dict: Dict) -> Dict[str, int]:
    presence = {
        "QB+WR": 0,
        "QB+TE": 0,
        "QB+WR+OppWR": 0,
        "QB+WR+WR+OppWR": 0,
        "WR vs OppWR": 0,
        "WR vs OppTE": 0,
        "TE vs OppWR": 0,
        "RB+WR same-team": 0,
    }

    qb_team = None
    opp_team = None
    for key in lineup:
        info = player_dict.get(key)
        if info and info.get("Position") == "QB":
            qb_team = info["Team"]
            opp_team = info.get("Opponent")
            break

    wr_by_team = defaultdict(list)
    te_by_team = defaultdict(list)
    rb_by_team = defaultdict(list)
    for key in lineup:
        info = player_dict.get(key)
        if not info:
            continue

        pos = info.get("Position")
        team = info.get("Team")
        if pos == "WR":
            wr_by_team[team].append(key)
        elif pos == "TE":
            te_by_team[team].append(key)
        elif pos == "RB":
            rb_by_team[team].append(key)

    if qb_team is not None:
        wr_same = wr_by_team.get(qb_team, [])
        te_same = te_by_team.get(qb_team, [])
        wr_opp = wr_by_team.get(opp_team, []) if opp_team else []
        presence["QB+WR"] = 1 if wr_same else 0
        presence["QB+TE"] = 1 if te_same else 0
        presence["QB+WR+OppWR"] = 1 if wr_same and wr_opp else 0
        presence["QB+WR+WR+OppWR"] = 1 if len(wr_same) >= 2 and wr_opp else 0

    for team, wrs in wr_by_team.items():
        opp = player_dict[wrs[0]].get("Opponent")
        if opp in wr_by_team and team < opp:
            presence["WR vs OppWR"] = 1
            break

    for team, wrs in wr_by_team.items():
        opp = player_dict[wrs[0]].get("Opponent")
        if opp in te_by_team:
            presence["WR vs OppTE"] = 1
            break

    for team, tes in te_by_team.items():
        opp = player_dict[tes[0]].get("Opponent")
        if opp in wr_by_team:
            presence["TE vs OppWR"] = 1
            break

    for team, wrs in wr_by_team.items():
        if rb_by_team.get(team):
            presence["RB+WR same-team"] = 1
            break

    presence["No Stack"] = 1 if not any(presence.values()) else 0
    return presence


def count_multiplicity(lineup: List[str], player_dict: Dict) -> Dict[str, int]:
    counts = Counter()
    qb_team = None
    opp_team = None
    for key in lineup:
        info = player_dict.get(key)
        if info and info.get("Position") == "QB":
            qb_team = info["Team"]
            opp_team = info.get("Opponent")
            break

    wr_by_team = defaultdict(list)
    te_by_team = defaultdict(list)
    rb_by_team = defaultdict(list)
    for key in lineup:
        info = player_dict.get(key)
        if not info:
            continue

        pos = info.get("Position")
        team = info.get("Team")
        if pos == "WR":
            wr_by_team[team].append(key)
        elif pos == "TE":
            te_by_team[team].append(key)
        elif pos == "RB":
            rb_by_team[team].append(key)

    if qb_team is not None:
        wr_same = wr_by_team.get(qb_team, [])
        te_same = te_by_team.get(qb_team, [])
        wr_opp = wr_by_team.get(opp_team, []) if opp_team else []
        counts["QB+WR"] = len(wr_same)
        counts["QB+TE"] = len(te_same)
        counts["QB+WR+OppWR"] = len(wr_same) * len(wr_opp)
        counts["QB+WR+WR+OppWR"] = (
            (len(wr_same) * (len(wr_same) - 1) // 2) * len(wr_opp)
        )

    for team, wrs in wr_by_team.items():
        opp = player_dict[wrs[0]].get("Opponent")
        if opp in wr_by_team and team < opp:
            counts["WR vs OppWR"] += len(wrs) * len(wr_by_team[opp])

    for team, wrs in wr_by_team.items():
        opp = player_dict[wrs[0]].get("Opponent")
        if opp in te_by_team:
            counts["WR vs OppTE"] += len(wrs) * len(te_by_team[opp])

    for team, tes in te_by_team.items():
        opp = player_dict[tes[0]].get("Opponent")
        if opp in wr_by_team:
            counts["TE vs OppWR"] += len(tes) * len(wr_by_team[opp])

    for team, rbs in rb_by_team.items():
        if team in wr_by_team:
            counts["RB+WR same-team"] += len(rbs) * len(wr_by_team[team])

    presence = detect_presence(lineup, player_dict)
    counts["No Stack"] = 1 if presence.get("No Stack") else 0
    return dict(counts)


def exclusive_bucket(lineup: List[str], player_dict: Dict) -> str:
    qb_team = None
    opp_team = None
    for key in lineup:
        info = player_dict.get(key)
        if info and info.get("Position") == "QB":
            qb_team = info["Team"]
            opp_team = info.get("Opponent")
            break

    wr_by_team = defaultdict(list)
    te_by_team = defaultdict(list)
    rb_by_team = defaultdict(list)
    for key in lineup:
        info = player_dict.get(key)
        if not info:
            continue

        pos = info.get("Position")
        team = info.get("Team")
        if pos == "WR":
            wr_by_team[team].append(key)
        elif pos == "TE":
            te_by_team[team].append(key)
        elif pos == "RB":
            rb_by_team[team].append(key)

    wr_same = wr_by_team.get(qb_team, []) if qb_team else []
    te_same = te_by_team.get(qb_team, []) if qb_team else []
    wr_opp = wr_by_team.get(opp_team, []) if opp_team else []

    if len(wr_same) >= 2 and wr_opp:
        return "QB+WR+WR+OppWR"
    if wr_same and te_same:
        return "QB+WR+TE"
    if te_same and wr_opp:
        return "QB+TE+OppWR"
    if wr_same and wr_opp:
        return "QB+WR+OppWR"
    if wr_same:
        return "QB+WR"
    if te_same:
        return "QB+TE"
    for team, wrs in wr_by_team.items():
        if rb_by_team.get(team):
            return "RB+WR same-team"
    return "No Stack"


def analyze_lineup(lineup: List[str], player_dict: Dict) -> Dict[str, Dict]:
    presence = detect_presence(lineup, player_dict)
    counts = count_multiplicity(lineup, player_dict)
    bucket = exclusive_bucket(lineup, player_dict)
    return {"presence": presence, "counts": counts, "bucket": bucket}
