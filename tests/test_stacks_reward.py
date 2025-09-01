import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from dfs.stacks import compute_presence_and_counts, classify_bucket, compute_features
from dfs.rl_reward import compute_reward


def test_stack_flags_and_reward():
    row = {
        'QB_team':'A','QB_opp':'B','QB_pos':'QB','QB':'QB1',
        'RB1_team':'A','RB1_opp':'B','RB1_pos':'RB','RB1':'RB_A',
        'RB2_team':'C','RB2_opp':'D','RB2_pos':'RB','RB2':'RB_C',
        'WR1_team':'A','WR1_opp':'B','WR1_pos':'WR','WR1':'WR_A1',
        'WR2_team':'B','WR2_opp':'A','WR2_pos':'WR','WR2':'WR_B',
        'WR3_team':'Z','WR3_opp':'Q','WR3_pos':'WR','WR3':'WR_Z',
        'TE_team':'E','TE_opp':'F','TE_pos':'TE','TE':'TE_E',
        'FLEX_team':'X','FLEX_opp':'Y','FLEX_pos':'WR','FLEX':'WR_X',
        'DST_team':'M','DST_opp':'N','DST_pos':'DST','DST':'DST_M',
        'salary':49900,
        'score':100,
        'projections_proj':100,
    }
    flags, counts = compute_presence_and_counts(row)
    assert flags['QB+WR'] == 1
    assert flags['QB+WR+OppWR'] == 1
    assert counts['QB+WR+OppWR'] == 1
    bucket = classify_bucket(flags)
    assert bucket == 'QB+WR+OppWR'
    feats = compute_features(row)
    assert feats['flex_pos'] == 'WR'
    assert feats['flex_is_wr'] == 1
    cfg_low = {'proj':0.0,'salary_util':0.0,'qb_wr_bonus':1.0,'bringback_bonus':0.0}
    cfg_high = {'proj':0.0,'salary_util':0.0,'qb_wr_bonus':5.0,'bringback_bonus':0.0}
    r_low = compute_reward(row, cfg_low)
    r_high = compute_reward(row, cfg_high)
    assert r_high > r_low
