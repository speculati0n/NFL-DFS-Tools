from collections import Counter

SLOTS = ["QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST"]


def lineup_key(lineup_dict):
    """Canonical key independent of slot order for non-DST skill players."""
    ids = []
    for s in SLOTS:
        pid = lineup_dict.get(f"{s}_id") or lineup_dict.get(s)
        if pid is not None:
            ids.append(str(pid))
    ids.sort()
    return tuple(ids)


def jaccard_similarity(a_ids, b_ids):
    A, B = set(a_ids), set(b_ids)
    if not A and not B:
        return 0.0
    return len(A & B) / float(len(A | B))

