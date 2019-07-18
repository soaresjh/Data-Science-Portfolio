from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing     import OneHotEncoder
import numpy as np

from titanic.tools import uniques

class OneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, categories_list = [], 
                 categories_number = [], 
                 droped = []):
        self.categories_list   = categories_list;
        self.categories_number = categories_number;

    def transform(self, x, y = None):
        
        if not isinstance(x, np.ndarray):
            try:
                x = x.values
            except:
                print('Input data X is neither pandas.Dataframe nor numpy.ndarray')
                return None
        
        observation, _   = x.shape
        categories_whole = sum(self.categories_number)
        
        encoded = np.zeros([observation, categories_whole])
                
        i = 0
        j = 0
        for cats in self.categories_list:
            for c in cats:
                array = x[:,i] == c
                array = array.astype(int)
                encoded[:, j] = array
                j += 1
            i += 1
        
        return encoded

    def fit(self, x, y = None):
        
        if not isinstance(x, np.ndarray):
            try:
                x = x.values
            except:
                print('Input data X is neither pandas.Dataframe nor numpy.ndarray')
                return self
        
        n_row, n_feat = x.shape
        list_cat = []
        num_cat  = []
        to_drop  = []
        
        for i in range(n_feat):
            categories = uniques(x[:,i])
            to_drop   += [categories[-1]]
            categories.remove(categories[-1])
            num_cat   += [len(categories)]
            list_cat  += [categories]
        
        self.categories_list     = list_cat
        self.categories_number = num_cat
        self.droped            = to_drop
        
        return self