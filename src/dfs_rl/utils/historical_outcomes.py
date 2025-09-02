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
    'amount_won': ['amount_won','Amount Won','amountWon','Winnings','winnings','Payout','payout'],
    'contest_id': ['Contest ID','ContestID','contest_id','ContestId'],
    'field_size': ['field_size','Field Size','maximumEntries'],
    'entries_per_user': ['maximumEntriesPerUser','maxEntriesPerUser','entries_per_user'],
    'entry_fee': ['entryFee','entry_fee','Entry Fee'],
    'contest_name': ['Contest Name','contest_name','Contest name','contestName'],
}

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

    # If we have no historical data, make sure the expected columns exist but do
    # not clobber any pre-existing values (e.g. an Arena run may already have
    # contest ranks from the full contest CSV).
    if hist.empty:
        for c in expose_cols:
            if c not in g.columns:
                g[c] = 0 if c == 'matches_found' else pd.NA
        if 'matches_found' in g.columns:
            g['matches_found'] = g['matches_found'].fillna(0)
        return g.drop(columns=['__lineup_key'])

    has_cid = 'contest_id' in g.columns and 'contest_id' in hist.columns

    if has_cid:
        hist_cols = [
            'contest_id',
            '__lineup_key',
            'rank',
            'amount_won',
            'field_size',
            'entries_per_user',
            'entry_fee',
            'contest_name',
        ]
        merged = g.merge(
            hist[hist_cols],
            on=['contest_id', '__lineup_key'],
            how='left',
            suffixes=('', '_hist'),
        )

        # Historical rank takes precedence, falling back to any existing values
        merged['contest_rank'] = merged['rank'].combine_first(merged.get('contest_rank'))
        merged.drop(columns=['rank'], inplace=True)

        for c in ['amount_won','field_size','entries_per_user','entry_fee','contest_name']:
            hist_col = f"{c}_hist"
            if hist_col in merged.columns:
                merged[c] = merged[hist_col].combine_first(merged.get(c))
                merged.drop(columns=[hist_col], inplace=True)

        merged['matches_found'] = (~merged['contest_rank'].isna()).astype(int)
        return merged.drop(columns=['__lineup_key'])

    # No Contest ID -> reduce duplicates by best rank, sum amount_won
    hist_cols = [
        '__lineup_key',
        'rank',
        'amount_won',
        'field_size',
        'entries_per_user',
        'entry_fee',
        'contest_name',
        'contest_id',
    ]
    tmp = g.merge(
        hist[hist_cols],
        on='__lineup_key',
        how='left',
        suffixes=('', '_hist'),
    )

    def _reduce(group):
        best_rank = group['rank_hist'].min() if group['rank_hist'].notna().any() else pd.NA
        amt = group['amount_won_hist'].fillna(0).sum() if group['amount_won_hist'].notna().any() else pd.NA
        fs = group['field_size_hist'].dropna().max() if group['field_size_hist'].notna().any() else pd.NA
        epu = group['entries_per_user_hist'].dropna().max() if group['entries_per_user_hist'].notna().any() else pd.NA
        fee = group['entry_fee_hist'].dropna().max() if group['entry_fee_hist'].notna().any() else pd.NA
        # prefer the most frequent contest_name in ties
        cname = group['contest_name_hist'].dropna()
        cname = cname.mode().iat[0] if len(cname) else pd.NA
        matches = group['contest_id_hist'].nunique(dropna=True)
        return pd.Series({
            'contest_rank': best_rank,
            'amount_won': amt,
            'field_size': fs,
            'entries_per_user': epu,
            'entry_fee': fee,
            'contest_name': cname,
            'matches_found': matches
        })

    reduced = (tmp.reset_index()
                 .groupby('index', dropna=False)
                 .apply(_reduce)
                 .reset_index()
                 .set_index('index'))

    out = g.join(reduced, how='left', rsuffix='_hist')

    for c in expose_cols:
        hist_col = f"{c}_hist"
        if hist_col in out.columns:
            out[c] = out[hist_col].combine_first(out.get(c))
            out.drop(columns=[hist_col], inplace=True)

    out['matches_found'] = out['matches_found'].fillna(0).astype(int)
    return out.drop(columns=['__lineup_key'])
