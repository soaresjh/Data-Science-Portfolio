### - libraries ####
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split

from model import model_build

from titanic.model_evaluation import auc_roc

from titanic.dataframe import to_list_categorical
from titanic.tools import filter_list,                                \
                          sectionizer

import sys
from pickle import dump 
from os import mkdir

sectionizer('LOADING  - ', 65)
#----------------------------------------------------------------------
args = sys.argv

path = str(args[1]) 
file = str(args[2])
when = str(args[3])

x = pd.read_csv(path + '/' + when + '_features_'+ file)
y = pd.read_csv(path + '/' + when + '_target_'+ file)\
        ['Survived'].values
print('FILES NAME')
print('----> '+ path + '/' + when + '_features_'+ file + ' <----')
print('----> '+ path + '/' + when + '_target_'+ file + '   <----')


features = list(x.columns)
cat = to_list_categorical(x, do_print = False)
num = filter_list(raw = features, nasty = cat)

sectionizer('MODELING ', 65)
#----------------------------------------------------------------------
model = model_build(categorical_feat = cat, 
                    numerical_feat   = num)

sectionizer(' - Train/Test Spliting ', 65)
#----------------------------------------------------------------------
x[num] = x[num].astype(float)
x_train, x_test,\
y_train, y_test = train_test_split(x,y,
                                   test_size = 0.25,
                                   random_state = 26)

sectionizer(' -- Training Model ', 65)
#----------------------------------------------------------------------
model.fit(x_train,y_train)

sectionizer('EVALUATING ', 65)
y_prob = model.predict_proba(x_test)
auc = auc_roc(model, y_prob, y_test)
print('Area Under the Curve = %0.3f' % auc)

sectionizer('SAVING MODEL ', 65)
try: 
    mkdir('model')
    print('model Directory created')
except: 
    print('model Directory already existent')

now = time.strftime("md%m%d_H%H%M_")
pkl = open('model/trained_model_'+now+file.split('.')[0]+'.pkl','wb')
dump(model,pkl)
pkl.close();
print('DONE!!!')
print('MODEL NAME')
print('----> model/trained_model_'+now+file.split('.')[0]+'.pkl <----')
