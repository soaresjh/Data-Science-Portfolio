from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline

from titanic.estimator         import OneHotEncoder
from sklearn.preprocessing     import StandardScaler
from mlxtend.feature_selection import ColumnSelector

#### -------------------------------------------------------------------

def Featurizer(categorical, numerical):
    featurizer = FeatureUnion([
                        ('encode_categorical', Pipeline([
                         ('select_columns', ColumnSelector(categorical)),
                         ('one_hot_encode',  OneHotEncoder())
                        ])),
                        ('encode_normalize', Pipeline([
                            ('select_columns', ColumnSelector(numerical)),
                            ('one_hot_encode', StandardScaler())
                        ]))
                    ])
    return featurizer