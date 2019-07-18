import pandas as pd
import numpy as np
from titanic.upper import *

### --------------------------------------------------------------------
def read_csv_col(file, col_list, d_type = None):
    
    df = pd.read_csv(file,
                     usecols = col_list,
                     dtype = d_type
                    )
    return df

### --------------------------------------------------------------------
def merge(list_list):
    
    merged_list = []
    
    for inner_list in list_list:
        for element in inner_list:
            if element not in merged_list:
                merged_list += [element]
    
    return merged_list

### --------------------------------------------------------------------
def join(list_list):
    joined_list = []
    
    for inner_list in list: 
        joined_list += inner_list
    
    return joined_list
    
### --------------------------------------------------------------------
def upper_case(arg):
    
    type_str = type(arg).__name__
    
    if type_str not in support_upper:
        print('unable to upper case')
        return arg
    
    upper_func = dict_func[type_str]
    uppered = upper_func(arg)
    
    return uppered

#### -------------------------------------------------------------------
def sectionizer(message, size = 26):
    print(message.ljust(size,'.'))
    
#### -------------------------------------------------------------------
def filter_list(raw, nasty):
    
    filtered = raw
    
    for element in nasty:
        if element in filtered:
            filtered.remove(element)
    
    return filtered

### --------------------------------------------------------------------
def uniques(listed_values):
    unique_list = [];
    for value in listed_values:
        if value not in unique_list:
            unique_list.append(value)
    return unique_list   