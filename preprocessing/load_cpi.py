import pandas as pd
def load_cpi():
    df = pd.read_csv('data/CPI_MONTHLY.csv', skiprows=25, nrows=359)
    df['date'] = pd.to_datetime(df['date'])
    df.rename(columns={'date': 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)
    return df

def load_revision_dates():
    pd.set_option('future.no_silent_downcasting', True)
    df = pd.read_csv('data/CPI_MONTHLY.csv', skiprows=387, nrows=429-387)
    df['date'] = pd.to_datetime(df['date'])
    df.rename(columns={'date': 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)
    df.replace('R', 1, inplace=True)
    df = df.infer_objects(copy=False)
    # df.fillna(0.0, inplace=True)
    return df

def merged_cpi_and_revisions():
    cpi_df = load_cpi()
    revisions_df = load_revision_dates()
    merged_df = cpi_df.merge(revisions_df, on='Date', how='left', suffixes=('', '_revision'))
    return merged_df