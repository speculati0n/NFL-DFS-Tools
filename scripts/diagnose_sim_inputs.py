#!/usr/bin/env python3
# Usage:
#   python scripts/diagnose_sim_inputs.py /path/to/projections.csv
# Prints POS counts and top offenders for non-numeric projections/salaries.

import sys, pandas as pd

PROJ_KEYS = ("projections_proj","proj","projection","fpts_proj","projected_points","fpts","points")
SAL_KEYS  = ("salary","sal","cost","dk_salary")

def _norm_pos(p):
    p = str(p or "").upper().strip()
    return "DST" if p in ("D","DEF","DS","D/ST") else p

def main(argv):
    if len(argv) != 2:
        print("Usage: python scripts/diagnose_sim_inputs.py /path/to/projections.csv"); sys.exit(2)
    df = pd.read_csv(argv[1], dtype=str)
    pos_col = next((c for c in df.columns if c.lower() in ("pos","position")), None)
    if pos_col:
        df[pos_col] = df[pos_col].map(_norm_pos)
        print("POS counts:", df[pos_col].value_counts().to_dict())
    # find a projection column
    pcol = next((c for c in PROJ_KEYS if c in df.columns), None)
    scol = next((c for c in SAL_KEYS if c in df.columns), None)
    if pcol:
        badp = df[pcol].isna() | (df[pcol].astype(str).str.strip() == "") | (df[pcol].astype(str).str.lower().isin(["nan","none"]))
        print(f"Missing/blank projections: {int(badp.sum())}")
        if badp.any():
            print(df.loc[badp, [c for c in ["name", pos_col, pcol] if c]].head(10).to_string(index=False))
    if scol:
        bads = df[scol].isna() | (df[scol].astype(str).str.strip() == "") | (df[scol].astype(str).str.lower().isin(["nan","none"]))
        print(f"Missing/blank salaries: {int(bads.sum())}")
        if bads.any():
            print(df.loc[bads, [c for c in ["name", pos_col, scol] if c]].head(10).to_string(index=False))

if __name__ == "__main__":
    main(sys.argv)
