#!/usr/bin/env python3
# scripts/fix_lineup_columns.py
import sys, os, csv

TARGET_HEADER = [
    "QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST",
    "Salary","Fpts Proj","Fpts Used","Fpts Act","Ceiling",
    "Own. Sum","Own. Product","STDDEV","Players vs DST","Stack"
]

def _normalize_header(cols):
    out, rb_i, wr_i = [], 0, 0
    for c in cols:
        if c.startswith("RB"): rb_i += 1; out.append(f"RB{rb_i}")
        elif c.startswith("WR"): wr_i += 1; out.append(f"WR{wr_i}")
        else: out.append(c)
    if len(out) == len(TARGET_HEADER): out = TARGET_HEADER
    return out

def fix_file(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f); rows = list(reader)
    if not rows: raise SystemExit("Empty file.")
    header = _normalize_header(rows[0])
    fixed = [header]
    for i, r in enumerate(rows[1:], start=2):
        if len(r) == 18:  # assume missing DST at index 8
            r = r[:8] + [""] + r[8:]
        fixed.append(r)
    out = os.path.splitext(path)[0] + "_fixed.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(fixed)
    print(f"✅ wrote {out}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/fix_lineup_columns.py /path/to/optimal_lineups.csv")
        sys.exit(2)
    fix_file(sys.argv[1])
