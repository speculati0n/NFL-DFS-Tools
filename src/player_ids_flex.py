#!/usr/bin/env python3
# src/player_ids_flex.py
"""
Flexible DraftKings player-IDs loader that supports:
  1) DK weekly salary CSV   (Position, Name + ID, Name, ID, TeamAbbrev, ...)
  2) DK bulk IDs CSV        (start_date, draftableid, displayname, position, ...)
  3) Custom mapping CSV     (must include id/name/position; optional team)

Returns a canonical DataFrame with columns:
  ID (int), Name (str), Position (str), TeamAbbrev (str or None)

Also provides:
  - dst_id_by_team(df, team_abbrev) -> int|None
"""

import pandas as pd
import re

CANON_COLS = ["ID", "Name", "Position", "TeamAbbrev"]

def _norm_pos(p: str) -> str:
    p = str(p or "").strip().upper()
    p = re.sub(r"[\[\]\"']", "", p)
    return "DST" if p in ("D", "DEF", "DS", "D/ST", "DST") else p

def _norm_name(n: str) -> str:
    n = str(n or "").strip()
    # A.J. -> AJ, squeeze whitespace
    n = re.sub(r"\.", "", n)
    n = re.sub(r"\s+", " ", n)
    # Drop common suffixes and trailing roman numerals
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n, flags=re.IGNORECASE)
    n = re.sub(r"\b[ivxlcdm]+\b$", "", n, flags=re.IGNORECASE)
    n = n.strip()
    return n.lower()

def _detect_schema(cols):
    s = {c.lower(): c for c in cols}
    if {"name + id", "id", "position"}.issubset(s.keys()):
        return "dk_salary"
    if {"draftableid", "displayname", "position"}.issubset(s.keys()):
        return "dk_bulk"
    if any(k in s for k in ("id","dk_id","draftableid")) and \
       any(k in s for k in ("name","displayname")) and \
       any(k in s for k in ("position","pos")):
        return "custom"
    return "unknown"

def _to_canon_from_salary(df: pd.DataFrame) -> pd.DataFrame:
    s = {c.lower(): c for c in df.columns}
    out = pd.DataFrame({
        "ID": df[s["id"]].astype(str).str.replace(r"[^\d]", "", regex=True).astype(int),
        "Name": df[s.get("name","Name")].astype(str).map(_norm_name),
        "Position": df[s.get("position","Position")].astype(str).map(_norm_pos),
        "TeamAbbrev": df[s.get("teamabbrev","TeamAbbrev")] if "teamabbrev" in s else None
    })
    return out

def _to_canon_from_bulk(df: pd.DataFrame) -> pd.DataFrame:
    s = {c.lower(): c for c in df.columns}
    # prefer latest row per (displayname, position)
    if "start_date" in s:
        df = df.assign(_start=pd.to_datetime(df[s["start_date"]], errors="coerce")) \
               .sort_values("_start") \
               .drop_duplicates(subset=[s["displayname"], s["position"]], keep="last")
    out = pd.DataFrame({
        "ID": df[s["draftableid"]].astype(str).str.replace(r"[^\d]", "", regex=True).astype(int),
        "Name": df[s["displayname"]].astype(str).map(_norm_name),
        "Position": df[s["position"]].astype(str).map(_norm_pos),
        "TeamAbbrev": None
    })
    return out

def _to_canon_from_custom(df: pd.DataFrame) -> pd.DataFrame:
    s = {c.lower(): c for c in df.columns}
    id_col  = s.get("id") or s.get("dk_id") or s.get("draftableid")
    name_col = s.get("name") or s.get("displayname")
    pos_col  = s.get("position") or s.get("pos")
    team_col = s.get("teamabbrev") or s.get("team") or s.get("team_abbrev")

    if not (id_col and name_col and pos_col):
        raise ValueError(f"Custom player_ids requires id/name/position columns. Got: {list(df.columns)}")

    out = pd.DataFrame({
        "ID": df[id_col].astype(str).str.replace(r"[^\d]", "", regex=True).astype(int),
        "Name": df[name_col].astype(str).map(_norm_name),
        "Position": df[pos_col].astype(str).map(_norm_pos),
        "TeamAbbrev": df[team_col] if team_col else None
    })
    return out

def load_player_ids_flex(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    schema = _detect_schema(df.columns)
    if schema == "dk_salary":
        canon = _to_canon_from_salary(df)
    elif schema == "dk_bulk":
        canon = _to_canon_from_bulk(df)
    elif schema == "custom":
        canon = _to_canon_from_custom(df)
    else:
        # Best-effort coercion
        id_like  = [c for c in df.columns if df[c].astype(str).str.match(r"^\d+$").mean() > 0.5]
        name_like = [c for c in df.columns if "name" in c.lower()]
        pos_like  = [c for c in df.columns if c.lower() in ("pos","position")]
        if id_like and name_like and pos_like:
            temp = df.rename(columns={id_like[0]:"id", name_like[0]:"name", pos_like[0]:"position"})
            canon = _to_canon_from_custom(temp)
        else:
            raise ValueError(f"Unrecognized player_ids schema: {list(df.columns)}")

    canon["Position"] = canon["Position"].map(_norm_pos)
    canon["Name"] = canon["Name"].map(_norm_name)
    if "TeamAbbrev" not in canon.columns:
        canon["TeamAbbrev"] = None
    # drop duplicates by (ID, Position)
    canon = canon.drop_duplicates(subset=["ID","Position"], keep="last").reset_index(drop=True)
    return canon

def dst_id_by_team(player_ids_df: pd.DataFrame, team_abbrev: str):
    team = str(team_abbrev or "").upper().strip()
    try:
        row = player_ids_df[
            (player_ids_df["Position"]=="DST") &
            (player_ids_df["TeamAbbrev"].astype(str).str.upper()==team)
        ].iloc[0]
        return int(row["ID"])
    except Exception:
        return None
