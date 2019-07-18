print('-------> INITIALIZING <-------')

import pandas as pd
import numpy as np
import time
import sys

from titanic.tools import read_csv_col,                                \
                          merge,                                       \
                          upper_case,                                  \
                          sectionizer

from titanic.dataframe import fill_mean,                               \
                              fill_commum,                             \
                              fill_zero,                               \
                              fill_unknow,                             \
                              interval_arange,                         \
                              group_arange,                            \
                              to_list_categorical,                     \
                              to_list_missing,                         \
                              upper_df

sectionizer('LOADING  - ', 65)
### File ---------------------------------------------------------------

args = sys.argv

path = str(args[1]) 
file = str(args[2])

data = pd.read_csv(path + '/' + file)
print('FILE NAME')
print('----> '+ path + '/' + file + ' <----')

print(' - raw dataframe')
missin_features   = to_list_missing(data, do_print = True)
print('')
categorical_features = to_list_categorical(data, do_print = True)
del data

sectionizer('PROCESSING  - ', 65)
processed = []

sectionizer('  - Age - ', 65)
#### Age ---------------------------------------------------------------

x_age        = read_csv_col(path + '/' + file, 
                            ['Age'], 
                            np.float64);
x_age['Age'] = fill_mean(x_age['Age'])

n_bins = 8
x_age = interval_arange(x_age, 'Age', n_bins)
processed += ['Age']

sectionizer('  - Fare - ', 65)
#### Age ---------------------------------------------------------------

x_fare         = read_csv_col(path + '/' + file, 
                            ['Fare'], 
                            np.float64);
x_fare['Fare'] = fill_mean(x_fare['Fare'])

n_bins = 10
x_fare = interval_arange(x_fare, 'Fare', n_bins)
processed += ['Fare']

sectionizer('  - Embarker - ', 65)
#### Embarked -----------------------------------------------------------

x_embarked             = read_csv_col(path + '/' + file, 
                                      ['Embarked'],
                                      str);
x_embarked['Embarked'] = fill_commum(x_embarked['Embarked'])
x_embarked = upper_df(x_embarked)

x_embarked.replace({'C':'Cherbourg', 'S':'Southampton', 'Q':'Queenstown'}, 
                   inplace = True)
x_embarked = upper_df(x_embarked)
processed += ['Embarked']

sectionizer('  - Cabin - ', 65)
#### Cabin -------------------------------------------------------------

x_cabin = read_csv_col(path + '/' + file, 
                       ['Cabin'],
                       str)
x_cabin = upper_df(x_cabin)
x_cabin['Cabin'] = fill_unknow(x_cabin['Cabin'])

flours = ('A','B','C','D','E','F','G');

x_cabin = group_arange(x_cabin, 'Cabin', flours)
processed += ['Cabin']

sectionizer('  - Title - ', 65)
#### Name --------------------------------------------------------------

x_name = read_csv_col(path + '/' + file, 
                      ['Name'], 
                      str)
x_name = upper_df(x_name)

title = ['Mr', 'Mrs','Ms','Mme','Miss', 'Mlle',
         'Master','Dr','Don','Countess',
         'Major','Cap','Col',
         'Rev']
title = upper_case(title)

x_name = group_arange(x_name, 'Name', title, out_group = 'MR')

dict_title = {'Mrs':'Ms', 'Mme':'Ms', 'Miss':'Ms', 'Mlle':'Ms',
              'Master':'Nobil','Dr':'Nobil',
              'Don':'Nobil', 'Countess':'Nobil',
              'Major':'Arm', 'Cap':'Arm', 'Col':'Arm'}
dict_title = upper_case(dict_title)

x_name['Name'].replace(dict_title, inplace = True)
processed += ['Name']

sectionizer('  - Sex - ', 65)
#### Sex ---------------------------------------------------------------
x_sex = read_csv_col(path + '/' + file, 
                     ['Sex'],
                     "category")
x_sex = upper_df(x_sex)
processed += ['Sex']

sectionizer('  - Ticket - ', 65)
#### Ticker ------------------------------------------------------------
print('*TICKET COLUMN IS BEING DROPED')
processed += ['Ticket']

sectionizer(' - Final dataset - ', 65)
### Data ---------------------------------------------------------------

drop_col = missin_features + processed
drop_col = drop_col+['PassengerId', 'Survived']
    
x = pd.read_csv(path + '/' + file,
                usecols = lambda x: x not in drop_col
               );
    
x = pd.concat([x,
               x_sex,
               x_age,
               x_fare,
               x_name,
               x_cabin,
               x_embarked,
              ], axis = 1);

pd.set_option('display.max_columns', 40)
pd.set_option('expand_frame_repr', False)

categorical_features = to_list_categorical(x, do_print = False)
print('  -> Numerical Features')
print(x.describe())
print('  -> Categorical Features')
print(x[categorical_features].describe())

sectionizer('SAVING  - ', 65)
now = time.strftime("md%m%d_H%H%M_")

try:
    y = read_csv_col(path + '/' + file, ['Survived'], int);
    y.to_csv(path + '/' + now +'target_'+   file, index = False)
    print('*training dataset')
except:
    print('*predicting dataset')
x.to_csv(path + '/' + now +'pre_feat_'+ file, index = False)

print('DONE !!!')
print('FILE NAME')
print('----> '+ now +'pre_feat_'+ file + ' <----')