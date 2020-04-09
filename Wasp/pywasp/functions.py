from pyspark.sql import functions as F, types as T

def server_finder(column):
    column = F.lower(column)
    regex_exp = r'(apache|nginx|microsoft)([^\/]+|)(\/|)((\d+\.|\d+\b|)+)'
    
    server  = F.regexp_extract(column, regex_exp, 1)
    version = F.regexp_extract(column, regex_exp, 4)
    
    server = F.when(server == '', 'other')                            \
              .otherwise(server)
    
    column = F.when(version == '', server)                            \
              .otherwise(F.concat_ws('/', server, version))
    
    return column

def is_in_list(column, listed):
    column = F.lower(column)

    column = F.when(~column.isin(listed),'other')                      \
              .otherwise(column)
    
    return column

def ith_(v, i):
    try:
        return float(v[i])
    except ValueError:
        return None

get_ith = F.udf(ith_, T.FloatType())
