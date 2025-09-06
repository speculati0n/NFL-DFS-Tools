import sys, csv

REQUIRED_COLUMNS = {"Name", "ID", "Position", "Team", "Salary", "Fpts"}


def check_file(path: str) -> int:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("No header found")
            return 1
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            print(f"Missing columns: {sorted(missing)}")
            return 1
        rows = list(reader)
        if not rows:
            print("No player rows found")
            return 1
    print("✅ projections file looks good")
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_projections.py projections.csv")
        sys.exit(2)
    sys.exit(check_file(sys.argv[1]))
