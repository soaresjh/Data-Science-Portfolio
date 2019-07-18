import numpy as np
import pandas as pd

from titanic import helper as h

### --------------------------------------------------------------------
def fill_mean(pandas_serie):
    mean_value   = pandas_serie.mean()
    filled_serie = pandas_serie.fillna(mean_value)
    
    return filled_serie

### --------------------------------------------------------------------
def fill_commum(pandas_serie):
    top_occurence = pandas_serie.describe()['top']
    filled_serie  = pandas_serie.fillna(top_occurence)
    
    return filled_serie

### --------------------------------------------------------------------
def fill_zero(pandas_serie):
    filled_serie = pandas_serie.fillna(0)
    return filled_serie

### --------------------------------------------------------------------
def fill_unknow(pandas_serie):
    filled_serie  = pandas_serie.fillna('UNKNOW')
    return filled_serie

### --------------------------------------------------------------------
def interval_arange(pandas_df, column, n_bins = 8):
    
    _, list_bins = pd.cut(pandas_df[column].values, 
                          bins = n_bins, 
                          retbins=True)
    
    binarized = pandas_df[column].apply(h.col_bins, 
                                        args = (list_bins,)
                                       )
    pandas_df[column] = binarized
    
    return pandas_df

### --------------------------------------------------------------------
def group_arange(pandas_df, column, list_group, out_group = 'unknow'):
    
    grouped = pandas_df[column].apply(h.col_group,
                                      args = (list_group,
                                              out_group)
                                     )
    pandas_df[column] = grouped
    
    return pandas_df
    
### --------------------------------------------------------------------
def upper_df(pandas_df):
    list_col = pandas_df.dtypes[pandas_df.dtypes==object].axes[0]
    upper_df = pandas_df.copy()
    
    for column in list_col:
        upper_df[column] = upper_df[column].str.upper()
    
    return upper_df
    
### --------------------------------------------------------------------
def to_list_categorical(pandas_df, do_print = False):
    
    obj_features = []
    for feat in pandas_df.columns:
        dt = pandas_df[feat].dtype.name
        if any([dt == 'category', dt == 'object']):
            obj_features += [feat]
    
    if do_print:
        obj_describe = []
        for feat in obj_features:
            l = len(pandas_df[feat].unique())
            obj_describe.append([feat,l])

        obj_describe.sort(key = lambda x: x[1], reverse = True)
    
        print('Object Features:', 
              *obj_describe, sep='\n- ')
    
    return obj_features

### --------------------------------------------------------------------
def to_list_missing(pandas_df, do_print = False):
    missing_features = pandas_df.columns[pandas_df.isna().sum()>0]
    missing_features = list(missing_features)
    
    if do_print:
        mis_describe = []
        for feat in missing_features:
            l = pandas_df[feat].isna().sum()
            mis_describe.append([feat,l])

        mis_describe.sort(key = lambda x: x[1], reverse = True)    

        print('Missing Features:', 
              *mis_describe, sep='\n- ')
    
    return missing_features


### --------------------------------------------------------------------
### --------------------------------------------------------------------
