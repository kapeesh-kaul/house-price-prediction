import pandas as pd

def load_rate():
    df = pd.read_csv('data/Prime-Rate-History.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.resample('D').interpolate(method='linear')
    return df