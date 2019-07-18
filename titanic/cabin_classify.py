import pandas as pd
import numpy as np
import time
import sys

from titanic.tools import read_csv_col,                                \
                          sectionizer

from sklearn.linear_model import LogisticRegression

sectionizer('LOADING  - ', 65)
### File ---------------------------------------------------------------
args = sys.argv

path = str(args[1]) 
file = str(args[2])
when = str(args[3])

x = read_csv_col(path + '/' + when + '_pre_feat_'+ file, 
                 ['Pclass','SibSp','Parch','Fare','Cabin'])
print('FILE NAME')
print('----> '+ path + '/' + when + '_pre_feat_'+ file + ' <----')

x_train = x.loc[x['Cabin'] != 'unknow']
y_train = x_train['Cabin'].values
x_train = x_train.drop('Cabin', axis = 1)

x_classify = x.loc[x['Cabin'] == 'unknow'].drop('Cabin', 
                                                axis = 1)
sectionizer('MODELING  - ', 65)
### Model --------------------------------------------------------------
csf = LogisticRegression(max_iter    = 10000,
                         solver      ='lbfgs', 
                         multi_class ='multinomial')

sectionizer('PREDICTING  - ', 65)
### Prediction ---------------------------------------------------------
csf.fit(x_train, y_train)
y_classify = csf.predict(x_classify)

x_train['Cabin']    = y_train
x_classify['Cabin'] = y_classify

x_cabin = pd.concat([x_train, x_classify]).sort_index()
x_cabin = x_cabin['Cabin']

sectionizer('SAVING  - ', 65)
### Save ---------------------------------------------------------------
x = pd.read_csv(path + '/' + when + '_pre_feat_'+ file).sort_index()
x['Cabin'] = x_cabin

x.to_csv(path + '/' + when +'_features_'+ file, index = False)
print('DONE!!!')
print('FILE NAME')
print('----> '+ when +'_features_'+ file + ' <----')