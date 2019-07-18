import numpy as np
import pandas as pd

### --------------------------------------------------------------------
def col_bins(df_element, list_bonds):
    
    is_in_bin = 0
    for bound in list_bonds:
        if df_element <= bound:
            return is_in_bin
        is_in_bin += 1
    
    final_bin = len(list_bonds)
    return final_bin

### --------------------------------------------------------------------
def col_group(df_element, list_group, out_group = 'UNKNOW'):
    
#     print('list_group = ', end = '')
#     print(list_group)
    
    for group in list_group:
#         print('group = ', end = '')
#         print(group)
#         print('df_elemnet = ', end = '')
#         print(df_element)
        if group in df_element:
            return group
        
    return out_group
    
### --------------------------------------------------------------------
### --------------------------------------------------------------------
### --------------------------------------------------------------------