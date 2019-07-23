import pandas as pd
import numpy as np

def read_serie(file, col):
    df = pd.read_csv(file,
                     usecols = [col]
                    )
    serie = df[col]
    return serie

def read_columns(file, col):
    df = pd.read_csv(file,
                     usecols = col
                    )
    return df

def to_list_missing(pandas_df, do_print = False):
    na_count = pandas_df.isna().sum(axis = 0)
    missing  = na_count[na_count > 0].index.tolist()
        
    if do_print:
        print('Missing Features ... ')
        for f in missing:
            print(('- ' + f).ljust(19) + ': %d' % na_count[f])
    
    return missing

def fill_mean(pandas_df, cols):
    
    df = pandas_df.copy(deep = True)
    
    for c in cols:
        m  = df[c].mean()
        df = df[c].fillna(m)
        
    return df

def fill_empty(pandas_df, cols):
    
    df = pandas_df.copy(deep = True)
    
    for c in cols:
        df = df[c].fillna('')
        
    return df

def fill_zero(pandas_df, cols):
    
    df = pandas_df.copy(deep = True)
    
    for c in cols:
        df = df[c].fillna()
        
    return df