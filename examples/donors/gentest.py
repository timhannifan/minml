import pandas as pd

def run():
    df = pd.read_csv('data_small.csv')
    print( print(df.iloc[0]))
    return df
