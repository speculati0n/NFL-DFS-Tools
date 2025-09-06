#!/usr/bin/env python3
# Usage:
#   python scripts/diagnose_sim_ids.py /path/to/player_ids.csv
# Prints counts the simulator will see after the flexible loader.

import sys
from player_ids_flex import load_player_ids_flex

def main(argv):
    if len(argv) != 2:
        print("Usage: python scripts/diagnose_sim_ids.py /path/to/player_ids.csv")
        sys.exit(2)
    df = load_player_ids_flex(argv[1])
    pos_counts = df["Position"].value_counts().to_dict()
    dst = df[df["Position"]=="DST"]
    print("POS counts:", pos_counts)
    print("DST rows:", len(dst))
    if "TeamAbbrev" in df.columns:
        miss_team = dst["TeamAbbrev"].isna().sum()
        print("DST rows missing TeamAbbrev:", int(miss_team))

if __name__ == "__main__":
    main(sys.argv)
