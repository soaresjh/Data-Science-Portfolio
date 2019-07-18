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
from pickle import load 
from os import mkdir

sectionizer('LOADING  - ', 65)
#----------------------------------------------------------------------
args = sys.argv

path = str(args[1]) 
file = str(args[2])
when = str(args[3])
when_model = str(args[5])
file_model = str(args[4])

x = pd.read_csv(path + '/' + when + '_features_'+ file)

print('FILES NAME')
print('----> '+ path + '/' + when + '_features_'+ file + ' <----')

features = list(x.columns)
cat = to_list_categorical(x, do_print = False)
num = filter_list(raw = features, nasty = cat)

sectionizer('MODELING ', 65)
#----------------------------------------------------------------------
pkl = open('model/trained_model_'+when_model+'_'+file_model+'.pkl','rb')
model = load(pkl)
pkl.close();

x[num] = x[num].astype(float)
y = model.predict(x)
x['Survived'] = y

now = time.strftime("md%m%d_H%H%M_")
x.to_csv(path +'/'+ now + 'predicted_registration.csv')
print('DONE!!!')
print('PREDICTION NAME')
print('----> '+path +'/'+ now + 'predicted_registration.csv'+ '<----')
