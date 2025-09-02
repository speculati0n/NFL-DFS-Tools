import os, re, glob
from datetime import datetime
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Hard-wired to your examples
PROJ_COLS = ["name","pos","team","opp","salary","projections_proj"]

def load_week_folder(week: str, root: str = "data/historical") -> Dict[str, pd.DataFrame]:
    """Load projections, player ids and contest CSVs for a given week.

    Parameters
    ----------
    week:
        Either a legacy week directory (``data/historical/YYYY/YYYY-MM-DD``)
        or an ISO date string (``YYYY-MM-DD``) matching files stored under
        ``root/{projections,players,contests}``.
    root:
        Base directory containing the historical subfolders.
    """

    # --- legacy path-style handling -------------------------------------------------
    if os.path.isdir(week):
        week_dir = week

        proj_fp = os.path.join(week_dir, "projections.csv")
        if not os.path.exists(proj_fp):
            cand = [p for p in glob.glob(os.path.join(week_dir, "*adjusted.csv"))
                    if "Aggregated_Lineup_Stats" not in os.path.basename(p)]
            if cand:
                proj_fp = cand[0]
        proj = pd.read_csv(proj_fp)

        missing = [c for c in PROJ_COLS if c not in proj.columns]
        if missing:
            raise ValueError(f"projections.csv missing columns: {missing}")
        proj["salary"] = pd.to_numeric(proj["salary"], errors="coerce").fillna(0).astype(int)
        proj["projections_proj"] = pd.to_numeric(proj["projections_proj"], errors="coerce").fillna(0.0)

        pid_fp = os.path.join(week_dir, "players_ids.csv")
        if not os.path.exists(pid_fp):
            pid_fp = os.path.join(week_dir, "player_ids.csv")
        if not os.path.exists(pid_fp):
            cand = [p for p in glob.glob(os.path.join(week_dir, "*.csv"))
                    if os.path.basename(p).startswith("201") and "Aggregated_Lineup_Stats" not in os.path.basename(p)
                       and "adjusted" not in os.path.basename(p).lower()]
            if cand:
                pid_fp = cand[0]
        pids = pd.read_csv(pid_fp) if os.path.exists(pid_fp) else pd.DataFrame()

        if not pids.empty:
            rename = {}
            if "displayname" in pids.columns: rename["displayname"] = "name"
            if "draftableid" in pids.columns: rename["draftableid"] = "player_id"
            if rename:
                pids = pids.rename(columns=rename)
            keep = [c for c in ["name","player_id","position","shortname","firstname","lastname","start_date"] if c in pids.columns]
            pids = pids[keep]

        contests_dir = os.path.join(week_dir, "contests")
        contest_files = sorted(glob.glob(os.path.join(contests_dir, "*.csv"))) if os.path.isdir(contests_dir) else []
        if not contest_files:
            contest_files = sorted(glob.glob(os.path.join(week_dir, "Aggregated_Lineup_Stats_*.csv")))

        return {"projections": proj, "players_ids": pids, "contest_files": contest_files}

    # --- new flat structure ---------------------------------------------------------
    date = week
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise FileNotFoundError(f"Invalid week identifier: {week}") from exc

    # projections
    proj_dir = os.path.join(root, "projections")
    date_token = f"{dt.month}.{dt.day}.{str(dt.year)[-2:]}"
    proj_candidates = glob.glob(os.path.join(proj_dir, f"*{date_token}*adjusted.csv"))
    if not proj_candidates:
        raise FileNotFoundError(f"Projection file for {date} not found in {proj_dir}")
    proj_fp = proj_candidates[0]
    proj = pd.read_csv(proj_fp)
    missing = [c for c in PROJ_COLS if c not in proj.columns]
    if missing:
        raise ValueError(f"projections.csv missing columns: {missing}")
    proj["salary"] = pd.to_numeric(proj["salary"], errors="coerce").fillna(0).astype(int)
    proj["projections_proj"] = pd.to_numeric(proj["projections_proj"], errors="coerce").fillna(0.0)

    # players ids
    pid_fp = os.path.join(root, "players", f"{date}.csv")
    pids = pd.read_csv(pid_fp) if os.path.exists(pid_fp) else pd.DataFrame()
    if not pids.empty:
        rename = {}
        if "displayname" in pids.columns: rename["displayname"] = "name"
        if "draftableid" in pids.columns: rename["draftableid"] = "player_id"
        if rename:
            pids = pids.rename(columns=rename)
        keep = [c for c in ["name","player_id","position","shortname","firstname","lastname","start_date"] if c in pids.columns]
        pids = pids[keep]

    # contest files
    contests_dir = os.path.join(root, "contests")
    contest_files = sorted(glob.glob(os.path.join(contests_dir, f"Aggregated_Lineup_Stats_{date}_*.csv")))

    return {"projections": proj, "players_ids": pids, "contest_files": contest_files}


def find_weeks(root="data/historical") -> List[Tuple[str, str]]:
    """Return ``(label, key)`` for all available historical weeks.

    The simplified data layout stores files under ``root`` with three
    subfolders: ``contests``, ``projections`` and ``players``.  Each week is
    identified by its ISO date (``YYYY-MM-DD``).  The returned *key* can be
    passed directly to :func:`load_week_folder`.
    """

    out: List[Tuple[str, str]] = []
    players_dir = os.path.join(root, "players")
    if not os.path.isdir(players_dir):
        return out
    for fname in sorted(os.listdir(players_dir)):
        m = re.match(r"(\d{4}-\d{2}-\d{2})\.csv$", fname)
        if m:
            date = m.group(1)
            out.append((date, date))
    return out
