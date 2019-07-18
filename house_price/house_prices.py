import numpy as np
import pandas as pd


def to_list_obj(DataFrame, do_print = False):
    features = DataFrame.dtypes[DataFrame.dtypes==object].index
    features = list(features)
    if do_print:
        print('Object Features: ', *features, sep = '\n-')
    return [features, len(features)]

def to_list_miss(DataFrame, do_print = False):
    features = DataFrame.isna().sum()[DataFrame.isna().sum()>0].index
    features = list(features)
    if do_print:
        print('Missing Features: ', *features, sep = '\n-')
    return [features, len(features)]