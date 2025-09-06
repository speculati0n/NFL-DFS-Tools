#!/usr/bin/env python3
# scripts/normalize_dst_names.py
# Usage:
#   python scripts/normalize_dst_names.py /path/to/projections.csv [--fd]
#
# Maps D/DEF->DST, normalizes teams (LA->LAR; add --fd for JAX->JAC), sets DST names as "<TEAM> DST".
import argparse, pandas as pd, os

def normalize_dst_names(path:str, fd_mode:bool=False)->str:
    df = pd.read_csv(path, dtype=str)
    cols = {c.lower(): c for c in df.columns}
    for r in ["name","pos","team"]:
        if r not in cols:
            raise SystemExit(f"Missing required column: {r}")
    name_col, pos_col, team_col = cols["name"], cols["pos"], cols["team"]
    df[pos_col]  = df[pos_col].astype(str).str.upper().str.strip().replace({"DEF":"DST","D":"DST"})
    df[team_col] = df[team_col].astype(str).str.upper().str.strip().replace({"LA":"LAR"})
    if fd_mode:
        df[team_col] = df[team_col].replace({"JAX":"JAC"})
    is_dst = df[pos_col].eq("DST")
    needs  = is_dst & (df[name_col].astype(str).str.strip().eq("") | ~df[name_col].str.contains("DST", case=False, na=True))
    df.loc[needs, name_col] = df.loc[needs, team_col].astype(str).str.upper().str.strip() + " DST"
    out = os.path.splitext(path)[0] + "_dstnorm.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"✅ Wrote: {out}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--fd", action="store_true", help="Apply FanDuel mapping JAX->JAC")
    args = ap.parse_args()
    normalize_dst_names(args.csv_path, fd_mode=args.fd)
