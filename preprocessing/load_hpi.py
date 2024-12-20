import pandas as pd

def load_hpi():
    xls = pd.ExcelFile('data/house_price_index.xlsx')
    sheets = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        df = df.rename(columns={df.columns[0]: 'Date'})
        df = df.add_prefix(f'{sheet_name}_')
        df = df.rename(columns={f'{sheet_name}_Date': 'Date'})
        
        df.index = pd.to_datetime(df['Date'])
        df = df.drop(columns=['Date'])
        df = df.resample('D').interpolate(method='linear')
        
        sheets[sheet_name] = df
    return sheets

def hpi_combined():
    sheets = load_hpi()
    combined_df = pd.DataFrame()
    for sheet_name, df in sheets.items():
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = combined_df.join(df, on='Date', how='outer', rsuffix=f'_{sheet_name}')
    return combined_df
