import numpy as np
import re

def uniques(listed_values):
    unique_list = [];
    for value in listed_values:
        if value not in unique_list:
            unique_list.append(value)
    return unique_list

def arange_matrix(list_or_array):
    list_elements = []
    n = 0
    for a in list_or_array:

        s = re.sub('[\[\]\']','',a).replace(', ', ',').split(',')
        n = len(s) if len(s) > n else n
        s_aux = []
        for e in s:
            if e not in s_aux:
                s_aux.append(e)
        list_elements+= [s_aux]

    m = len(list_elements)
    
    matrix_elements = np.empty([m, n]).astype(str)
    
    for i in range(m):
        for j in range(n):
            try:
                matrix_elements[i][j] = list_elements[i][j]
            except:
                matrix_elements[i][j] = None
    
    return matrix_elements

def to_list_all(matrix_2d):
    
    try:
        m, n = matrix_2d.shape
    except:
        m, = matrix_2d.shape
        n  = 1
    uniques = []
    
    for i in range(m):
        for j in range(n):
            elemente = matrix_2d[i][j]
            if elemente not in uniques:
                uniques.append(elemente)
                
    return uniques
    
    
