from typing import Dict, Tuple, List
from collections import Counter

ROSTER = ['QB','RB1','RB2','WR1','WR2','WR3','TE','FLEX','DST']

# Exclusive precedence (most specific → least). Keep in sync with analysis script.
BUCKET_PRECEDENCE: List[str] = [
    'QB+WR+WR+OppWR+OppWR','QB+WR+WR+OppWR','QB+WR+WR+OppTE',
    'QB+WR+WR+TE','QB+WR+WR+RB','QB+WR+WR+WR',
    'QB+WR+OppWR','QB+WR+OppTE','QB+TE+OppWR','QB+RB+OppWR',
    'QB+WR+TE','QB+WR+RB','QB+TE','QB+RB','QB+WR',
    'WR+WR same-team','WR+TE same-team','RB+WR same-team','RB+TE same-team','RB+DEF same-team',
    'WR vs OppWR','TE vs OppWR','WR vs OppTE','RB vs OppWR'
]

def _team(row, slot): return row.get(f"{slot}_team")
def _opp(row,  slot): return row.get(f"{slot}_opp")
def _pos(row,  slot): return row.get(f"{slot}_pos")

def _comb(n: int, r: int) -> int:
    if r < 0 or r > n: return 0
    if r in (0,1): return n if r==1 else 1
    # small r only (r <= 3 here)
    if r == 2: return n*(n-1)//2
    if r == 3: return n*(n-1)*(n-2)//6
    return 0

def compute_features(row: Dict) -> Dict[str, int | str]:
    # Any vs DST (player on the same team as DST's opponent)
    dst_team = _team(row, 'DST')
    dst_opp  = _opp(row,  'DST')
    any_vs = 0
    for s in ['QB','RB1','RB2','WR1','WR2','WR3','TE','FLEX']:
        t, o = _team(row, s), _opp(row, s)
        if t and o and dst_team and dst_opp and (t == dst_team and o == dst_team):
            # ignore same-team; focus on "player facing his own DST"
            pass
        if t and o and dst_team and dst_opp and (t == dst_opp):  # player on team opposing DST
            any_vs += 1

    # Double TE
    te_count = int(_pos(row,'TE') == 'TE') + int(_pos(row,'FLEX') == 'TE')
    double_te = 1 if te_count >= 2 else 0

    # FLEX position
    fpos = _pos(row,'FLEX') or ''
    return {
        "feat_any_vs_dst": any_vs,
        "feat_double_te": double_te,
        "flex_pos": fpos,
        "flex_is_wr": 1 if fpos == 'WR' else 0,
        "flex_is_rb": 1 if fpos == 'RB' else 0,
        "flex_is_te": 1 if fpos == 'TE' else 0
    }

def compute_presence_and_counts(row: Dict) -> Tuple[Dict[str,int], Dict[str,int]]:
    flags: Dict[str,int] = {}
    counts: Dict[str,int] = {}

    qb_t, qb_o = _team(row,'QB'), _opp(row,'QB')
    wr_slots = [s for s in ['WR1','WR2','WR3'] if _team(row,s)]
    rb_slots = [s for s in ['RB1','RB2'] if _team(row,s)]

    wr_t = [_team(row,s) for s in wr_slots]
    wr_o = [_opp(row,s)  for s in wr_slots]
    rb_t = [_team(row,s) for s in rb_slots]
    rb_o = [_opp(row,s)  for s in rb_slots]

    te_t, te_o = _team(row,'TE'), _opp(row,'TE')
    dst_t = _team(row,'DST')

    n_wr_same = sum(1 for t in wr_t if qb_t and t == qb_t)
    n_wr_opp  = sum(1 for t in wr_t if qb_o and t == qb_o)
    n_rb_same = sum(1 for t in rb_t if qb_t and t == qb_t)
    n_rb_opp  = sum(1 for t in rb_t if qb_o and t == qb_o)
    te_same   = 1 if (te_t and qb_t and te_t == qb_t) else 0
    te_opp    = 1 if (te_t and qb_o and te_t == qb_o) else 0

    def set_(k, pres, cnt): flags[k] = 1 if pres else 0; counts[k] = int(cnt)

    # Same-team stacks
    set_('QB+WR', n_wr_same>=1, n_wr_same)
    set_('QB+WR+WR', n_wr_same>=2, _comb(n_wr_same,2))
    set_('QB+WR+WR+WR', n_wr_same>=3, _comb(n_wr_same,3))
    set_('QB+TE', te_same==1, te_same)
    set_('QB+RB', n_rb_same>=1, n_rb_same)
    set_('QB+WR+TE', n_wr_same>=1 and te_same==1, n_wr_same*te_same)
    set_('QB+WR+RB', n_wr_same>=1 and n_rb_same>=1, n_wr_same*n_rb_same)
    set_('QB+RB+TE', n_rb_same>=1 and te_same==1, n_rb_same*te_same)
    set_('QB+WR+WR+TE', n_wr_same>=2 and te_same==1, _comb(n_wr_same,2)*te_same)
    set_('QB+WR+WR+RB', n_wr_same>=2 and n_rb_same>=1, _comb(n_wr_same,2)*n_rb_same)

    # Bring-backs
    set_('QB+OppWR', n_wr_opp>=1, n_wr_opp)
    set_('QB+OppTE', te_opp==1, te_opp)
    set_('QB+OppRB', n_rb_opp>=1, n_rb_opp)
    set_('QB+WR+OppWR', n_wr_same>=1 and n_wr_opp>=1, n_wr_same*n_wr_opp)
    set_('QB+WR+OppTE', n_wr_same>=1 and te_opp==1, n_wr_same*te_opp)
    set_('QB+WR+OppRB', n_wr_same>=1 and n_rb_opp>=1, n_wr_same*n_rb_opp)
    set_('QB+TE+OppWR', te_same==1 and n_wr_opp>=1, te_same*n_wr_opp)
    set_('QB+TE+OppTE', te_same==1 and te_opp==1, te_same*te_opp)
    set_('QB+RB+OppWR', n_rb_same>=1 and n_wr_opp>=1, n_rb_same*n_wr_opp)
    set_('QB+RB+OppTE', n_rb_same>=1 and te_opp==1, n_rb_same*te_opp)
    set_('QB+WR+WR+OppWR', n_wr_same>=2 and n_wr_opp>=1, _comb(n_wr_same,2)*n_wr_opp)
    set_('QB+WR+WR+OppTE', n_wr_same>=2 and te_opp==1, _comb(n_wr_same,2)*te_opp)
    set_('QB+WR+WR+OppWR+OppWR', n_wr_same>=2 and n_wr_opp>=2, _comb(n_wr_same,2)*_comb(n_wr_opp,2))

    # Minis
    wr_counts = Counter([t for t in wr_t if t])
    rb_counts = Counter([t for t in rb_t if t])
    wrwr_pairs = sum(_comb(c,2) for c in wr_counts.values())
    set_('WR+WR same-team', wrwr_pairs>=1, wrwr_pairs)
    rbwr_pairs = sum(rb_counts.get(t,0) for t in wr_counts)  # pairs across RB/WR on same team
    set_('RB+WR same-team', rbwr_pairs>=1, rbwr_pairs)
    wrte_pairs = wr_counts.get(te_t,0) if te_t else 0
    set_('WR+TE same-team', wrte_pairs>=1, wrte_pairs)
    rbte_pairs = rb_counts.get(te_t,0) if te_t else 0
    set_('RB+TE same-team', rbte_pairs>=1, rbte_pairs)
    rbdef_pairs = sum(1 for t in rb_t if t and t == dst_t)
    set_('RB+DEF same-team', rbdef_pairs>=1, rbdef_pairs)

    # vs minis
    # WR vs OppWR
    wr_vs_wr = 0
    for i in range(len(wr_slots)):
        for j in range(i+1,len(wr_slots)):
            if wr_t[i] and wr_o[i] and wr_t[j] and wr_o[j] and wr_t[i]==wr_o[j] and wr_t[j]==wr_o[i]:
                wr_vs_wr += 1
    set_('WR vs OppWR', wr_vs_wr>=1, wr_vs_wr)

    te_vs_wr = 0
    if te_t and te_o:
        for j in range(len(wr_slots)):
            if wr_t[j] and wr_o[j] and te_t==wr_o[j] and wr_t[j]==te_o:
                te_vs_wr += 1
    set_('TE vs OppWR', te_vs_wr>=1, te_vs_wr)

    wr_vs_te = 0
    if te_t and te_o:
        for j in range(len(wr_slots)):
            if wr_t[j] and wr_o[j] and wr_t[j]==te_o and te_t==wr_o[j]:
                wr_vs_te += 1
    set_('WR vs OppTE', wr_vs_te>=1, wr_vs_te)

    rb_vs_wr = 0
    for i in range(len(rb_slots)):
        for j in range(len(wr_slots)):
            if rb_t[i] and rb_o[i] and wr_t[j] and wr_o[j] and rb_t[i]==wr_o[j] and wr_t[j]==rb_o[i]:
                rb_vs_wr += 1
    set_('RB vs OppWR', rb_vs_wr>=1, rb_vs_wr)

    return flags, counts

def classify_bucket(flags: Dict[str,int]) -> str:
    for k in BUCKET_PRECEDENCE:
        if flags.get(k,0) == 1:
            return k
    return "No Stack"
