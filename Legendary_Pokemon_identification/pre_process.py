import numpy as np
import pandas as pd

from pokemon import spark  as S,       \
             import pandas as P

from pokemon.tools import uniques,     \
                          to_list_all, \
                          arange_matrix

import findspark
findspark.init()
from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext('local[*]')
sqlContext = SQLContext(sc)

path = 'data'
file = 'Pokemon.csv'

features = ['attack', 'capture_rate', 'defense',
            'height_m', 'hp', 'sp_attack', 
            'sp_defense',  'speed', 'weight_kg']

pandas_df = P.read_columns(path+'/'+file, features)

### --------------------------------------------------------------------
array_abilities  = P.read_serie(path+'/'+file,'abilities').values
matrix_abilities = arange_matrix(array_abilities)

pandas_abilities = pd.DataFrame(matrix_abilities, 
                                columns = ['abilitie_1', 'abilitie_2',
                                           'abilitie_3', 'abilitie_4',
                                           'abilitie_5', 'abilitie_6']
                               )

### -------------------------------------------------------------------                       
array_type   = P.read_serie(path+'/'+file,'type1').values

list_types   = to_list_all(array_type)
list_ability = to_list_all(matrix_abilities)

pandas_df = P.fill_mean(pandas_df, ['height_m', 'weight_kg'])

x = sqlContext.createDataFrame(pandas_df)

### -------------------------------------------------------------------
x_abil = sqlContext.createDataFrame(pandas_abilities)
x_abil = S.count_encoder(x_abil, list_ability)

### -------------------------------------------------------------------
x_type = S.read_column(path+'/'+file, 
                       sqlContext,
                       ['type1', 'type2'])
x_type = S.counter_encoder(x_type, list_types)

### -------------------------------------------------------------------
x      = x.withColumn('id',F.monotonically_increasing_id())
x_type = x_type.withColumn('id',F.monotonically_increasing_id())
x_abil = x_abil.withColumn('id',F.monotonically_increasing_id()) 

x.write.csv(path + '/encoded_' + file, 
            header = True
           )