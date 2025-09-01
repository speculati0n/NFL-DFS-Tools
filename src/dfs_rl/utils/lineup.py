from typing import List, Dict
import pandas as pd

from dfs.constraints import DEFAULT_SALARY_CAP

DK_CAP = DEFAULT_SALARY_CAP
SLOTS = ["QB","RB","RB","WR","WR","WR","TE","FLEX","DST"]

def valid_lineup(players: List[Dict]) -> bool:
    if len(players) != 9: return False
    cnt = {"QB":0,"RB":0,"WR":0,"TE":0,"DST":0}
    names = set()
    salary = 0
    for p in players:
        pos = str(p.get("pos",""))
        if p.get("name") in names: return False
        names.add(p.get("name"))
        cnt[pos] = cnt.get(pos, 0) + 1
        salary += int(p.get("salary",0))
    if cnt["QB"] != 1 or cnt["DST"] != 1 or cnt["RB"] < 2 or cnt["WR"] < 3 or cnt["TE"] < 1: return False
    if salary > DK_CAP: return False
    return True

def score_lineup(players: List[Dict], col="projections_proj") -> float:
    return float(sum(float(p.get(col,0.0)) for p in players))

def to_df(players: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(players)
