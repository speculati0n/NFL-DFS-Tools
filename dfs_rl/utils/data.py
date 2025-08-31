import os, re, glob
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Hard-wired to your examples
PROJ_COLS = ["name","pos","team","opp","salary","projections_proj"]

def load_week_folder(week_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load projections, players_ids, and contest CSVs for a given historical week.
    - projections.csv: copy of *adjusted.csv from FantasyLabs (must include name,pos,team,opp,salary,projections_proj)
    - players_ids.csv: copy of YYYY-MM-DD.csv (must include displayname,draftableid)
    - contests/*.csv: Aggregated_Lineup_Stats_*.csv (leaderboard)
    """
    if not os.path.isdir(week_dir):
        raise FileNotFoundError(f"Week folder not found: {week_dir}")

    # projections
    proj_fp = os.path.join(week_dir, "projections.csv")
    if not os.path.exists(proj_fp):
        # fallback: auto-pick any *adjusted.csv if user drops it directly
        cand = [p for p in glob.glob(os.path.join(week_dir, "*adjusted.csv"))
                if "Aggregated_Lineup_Stats" not in os.path.basename(p)]
        if cand:
            proj_fp = cand[0]
    proj = pd.read_csv(proj_fp)
    # ensure required columns
    missing = [c for c in PROJ_COLS if c not in proj.columns]
    if missing:
        raise ValueError(f"projections.csv missing columns: {missing}")
    # normalize dtypes
    proj["salary"] = pd.to_numeric(proj["salary"], errors="coerce").fillna(0).astype(int)
    proj["projections_proj"] = pd.to_numeric(proj["projections_proj"], errors="coerce").fillna(0.0)

    # players ids
    pid_fp = os.path.join(week_dir, "players_ids.csv")
    if not os.path.exists(pid_fp):
        # allow singular form
        pid_fp = os.path.join(week_dir, "player_ids.csv")
    if not os.path.exists(pid_fp):
        # fallback: try a dated csv in the folder (like '2019-09-22.csv')
        cand = [p for p in glob.glob(os.path.join(week_dir, "*.csv"))
                if os.path.basename(p).startswith("201") and "Aggregated_Lineup_Stats" not in os.path.basename(p)
                   and "adjusted" not in os.path.basename(p).lower()]
        if cand:
            pid_fp = cand[0]
    pids = pd.read_csv(pid_fp) if os.path.exists(pid_fp) else pd.DataFrame()
    # map columns to canonical names
    if not pids.empty:
        # we only need displayname and draftableid for downstream lookup
        rename = {}
        if "displayname" in pids.columns: rename["displayname"] = "name"
        if "draftableid" in pids.columns: rename["draftableid"] = "player_id"
        if rename:
            pids = pids.rename(columns=rename)
        keep = [c for c in ["name","player_id","position","shortname","firstname","lastname","start_date"] if c in pids.columns]
        pids = pids[keep]

    # contest files
    contests_dir = os.path.join(week_dir, "contests")
    contest_files = sorted(glob.glob(os.path.join(contests_dir, "*.csv"))) if os.path.isdir(contests_dir) else []
    if not contest_files:
        # allow placing the Aggregated_*.csv directly in the week folder
        contest_files = sorted(glob.glob(os.path.join(week_dir, "Aggregated_Lineup_Stats_*.csv")))

    return {"projections": proj, "players_ids": pids, "contest_files": contest_files}


def find_weeks(root="data/historical") -> List[Tuple[str,str]]:
    """Return (label, path) for all historical week folders: data/historical/YYYY/YYYY-MM-DD."""
    out = []
    if not os.path.isdir(root): return out
    for year in sorted(os.listdir(root)):
        ydir = os.path.join(root, year)
        if not os.path.isdir(ydir): continue
        for d in sorted(os.listdir(ydir)):
            wdir = os.path.join(ydir, d)
            if os.path.isdir(wdir) and re.match(r"^\d{4}-\d{2}-\d{2}$", d):
                out.append((f"{year} — {d}", wdir))
    return out
