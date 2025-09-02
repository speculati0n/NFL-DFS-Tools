import os
import glob
import pandas as pd
from datetime import datetime

ROSTER_SLOTS = ['QB','RB1','RB2','WR1','WR2','WR3','TE','FLEX','DST']

# --- lineup key (ordered, stable) ---
def _lineup_key(df_like) -> pd.Series:
    def key_from_row(r):
        return '|'.join(str(r.get(s, '')).strip() for s in ROSTER_SLOTS)
    if isinstance(df_like, pd.Series):
        return key_from_row(df_like)
    return df_like.apply(key_from_row, axis=1)

def _date_to_filename(date_like: str) -> str:
    """
    Accept 'YYYY-MM-DD' or mm/dd/yyyy or Timestamp and return the exact filename
    used in data/historical: Aggregated_Lineup_Stats_YYYY-MM-DD_stack.csv
    """
    if isinstance(date_like, str):
        try:
            dt = datetime.strptime(date_like[:10], "%Y-%m-%d")
        except ValueError:
            dt = datetime.strptime(date_like[:10], "%m/%d/%Y")
    else:
        dt = pd.to_datetime(date_like)
    return f"Aggregated_Lineup_Stats_{dt.strftime('%Y-%m-%d')}_stack.csv"

# --- flexible column aliasing ---
ALIASES = {
    'rank': ['rank', 'Rank', 'RANK', 'contest_rank'],
    'amount_won': ['amount_won', 'Amount Won', 'amountWon', 'Winnings', 'winnings', 'Payout', 'payout'],
    'contest_id': ['Contest ID', 'ContestID', 'contest_id', 'ContestId'],
    'field_size': ['field_size', 'Field Size', 'maximumEntries'],
    'entries_per_user': ['maximumEntriesPerUser', 'maxEntriesPerUser', 'entries_per_user'],
    'entry_fee': ['entryFee', 'entry_fee', 'Entry Fee'],
    'contest_name': ['Contest Name', 'contest_name', 'Contest name', 'contestName'],
    'score': [
        'score',
        'Score',
        'dk_points',
        'DK Points',
        'points',
        'Points',
        'lineup_points',
        'lineupPoints',
        'FPTS',
        'fpts',
        'total_points',
        'totalPoints',
    ],
}


def standardize_scoreboard_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with common leaderboard columns renamed.

    The :data:`ALIASES` map above contains canonical column names and a list of
    possible variants seen in different data sources.  This helper applies the
    mapping so that downstream code can rely on columns like ``score``,
    ``rank``, ``amount_won``, ``field_size`` and friends regardless of the
    original header names.
    """

    ren = {}
    for canonical, opts in ALIASES.items():
        for c in opts:
            if c in df.columns:
                ren[c] = canonical
                break
    return df.rename(columns=ren) if ren else df

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    # normalize aliases
    ren = {}
    for canonical, opts in ALIASES.items():
        for c in opts:
            if c in df.columns:
                ren[c] = canonical
                break
    if ren:
        df = df.rename(columns=ren)

    # make sure roster slots exist (repair lowercased)
    for s in ROSTER_SLOTS:
        if s not in df.columns:
            lc = s.lower()
            if lc in df.columns:
                df = df.rename(columns={lc: s})
    return df

def load_outcomes_for_date(base_dir: str, date_like: str) -> pd.DataFrame:
    """
    Search recursively under base_dir for the given date file:
      Aggregated_Lineup_Stats_YYYY-MM-DD_stack.csv
    Return slim DF with normalized columns + roster slots + lineup key.
    """
    fname = _date_to_filename(date_like)
    fps = glob.glob(os.path.join(base_dir, "**", fname), recursive=True)
    if not fps:
        return pd.DataFrame()

    keeps = (['contest_id','rank','amount_won','field_size','entries_per_user','entry_fee','contest_name'] + ROSTER_SLOTS)
    out = []
    for fp in fps:
        try:
            df = pd.read_csv(fp, low_memory=False)
        except Exception:
            continue
        df = _ensure_cols(df)

        # require roster slots to build key
        if not all(s in df.columns for s in ROSTER_SLOTS):
            continue

        slim = df[[c for c in keeps if c in df.columns]].copy()
        # lineup key
        slim['__lineup_key'] = _lineup_key(slim)
        out.append(slim)

    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True).drop_duplicates()

def attach_historical_outcomes(
    generated_df: pd.DataFrame,
    date_like: str,
    base_dir: str
) -> pd.DataFrame:
    """
    Merge historical outcomes + contest metadata into generated_df.
    If generated_df has 'Contest ID' already, normalize it to 'contest_id' first.
    """
    if generated_df.empty:
        return generated_df

    hist = load_outcomes_for_date(base_dir, date_like)
    g = generated_df.copy()

    # Normalize generated_df columns (Contest ID & Contest Name may exist already)
    if 'Contest ID' in g.columns and 'contest_id' not in g.columns:
        g = g.rename(columns={'Contest ID':'contest_id'})
    if 'Contest Name' in g.columns and 'contest_name' not in g.columns:
        g = g.rename(columns={'Contest Name':'contest_name'})

    g['__lineup_key'] = _lineup_key(g)

    # Compose the columns we will expose
    expose_cols = ['contest_rank','amount_won','field_size','entries_per_user','entry_fee','contest_name','matches_found']

    # Stash any existing columns so we can preserve pre-computed results
    existing = {c: g[c] for c in expose_cols if c in g.columns}
    g = g.drop(columns=list(existing.keys()), errors='ignore')

    if hist.empty:
        # No historical data → restore existing columns (if any) and
        # populate missing ones with NA
        for c, series in existing.items():
            g[c] = series
        for c in expose_cols:
            if c not in g.columns:
                g[c] = pd.NA
        return g.drop(columns=['__lineup_key'])

    has_cid = 'contest_id' in g.columns and 'contest_id' in hist.columns

    if has_cid:
        cols = [
            'contest_id',
            '__lineup_key',
            'rank',
            'amount_won',
            'field_size',
            'entries_per_user',
            'entry_fee',
            'contest_name',
        ]
        cols = [c for c in cols if c in hist.columns]
        merged = g.merge(hist[cols], on=['contest_id', '__lineup_key'], how='left')
        if 'rank' in merged.columns:
            merged = merged.rename(columns={'rank': 'contest_rank'})
        else:
            merged['contest_rank'] = pd.NA
        # ensure expected columns exist even if absent in historical data
        for missing in ['amount_won', 'field_size', 'entries_per_user', 'entry_fee', 'contest_name']:
            if missing not in merged.columns:
                merged[missing] = pd.NA
        merged['matches_found'] = (~merged['contest_rank'].isna()).astype(int)
        for c, series in existing.items():
            if c in merged.columns:
                merged[c] = merged[c].combine_first(series)
            else:
                merged[c] = series
        return merged.drop(columns=['__lineup_key'])

    # No Contest ID → reduce duplicates by best rank, sum amount_won
    cols = [
        '__lineup_key',
        'rank',
        'amount_won',
        'field_size',
        'entries_per_user',
        'entry_fee',
        'contest_name',
        'contest_id',
    ]
    cols = [c for c in cols if c in hist.columns]
    tmp = g.merge(hist[cols], on='__lineup_key', how='left')

    def _reduce(group):
        best_rank = group['rank'].min() if 'rank' in group and group['rank'].notna().any() else pd.NA
        amt = group['amount_won'].fillna(0).sum() if 'amount_won' in group and group['amount_won'].notna().any() else pd.NA
        fs = group['field_size'].dropna().max() if 'field_size' in group and group['field_size'].notna().any() else pd.NA
        epu = group['entries_per_user'].dropna().max() if 'entries_per_user' in group and group['entries_per_user'].notna().any() else pd.NA
        fee = group['entry_fee'].dropna().max() if 'entry_fee' in group and group['entry_fee'].notna().any() else pd.NA
        # prefer the most frequent contest_name in ties
        cname_series = group['contest_name'].dropna() if 'contest_name' in group else pd.Series()
        cname = cname_series.mode().iat[0] if len(cname_series) else pd.NA
        matches = group['contest_id'].nunique(dropna=True) if 'contest_id' in group else 0
        return pd.Series({
            'contest_rank': best_rank,
            'amount_won': amt,
            'field_size': fs,
            'entries_per_user': epu,
            'entry_fee': fee,
            'contest_name': cname,
            'matches_found': matches,
        })

    reduced = (
        tmp.reset_index()
        .groupby('index', dropna=False)
        .apply(_reduce)
        .reset_index()
        .set_index('index')
    )
    out = g.join(reduced, how='left').drop(columns=['__lineup_key'])
    for c, series in existing.items():
        if c in out.columns:
            out[c] = out[c].combine_first(series)
        else:
            out[c] = series
    return out
