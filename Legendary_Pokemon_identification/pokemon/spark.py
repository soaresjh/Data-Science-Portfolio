

### --------------------------------------------------------------------
def read_column(file_path, sql, coluns):
    df_spark = sql.read.csv(file_path, 
                            header      = True, 
                            inferSchema = True,
                           )
    df_columns = df_spark.select(*columns)
    
    return df_columns

### --------------------------------------------------------------------
def counter_encoder(spark_df, category_list, prefix = ''):
    list_feature = spark_df.columns
    spark_df = spark_df.na.fill('None')
    for cat in category_list:
        spark_df = spark_df.withColumn(prefix+'_'+cat, 
                                           sum(F.col(f)\
                                                   .contains(cat)\
                                                   .cast('int')\
                                               for f in list_feature
                                           )
                                      )
        
    col_class = [prefix+'_'+cat for cat in category_list]
    df = spark_df.select(col_class)

    return df

