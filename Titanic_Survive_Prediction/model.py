from sklearn.pipeline import Pipeline

from titanic.pipeline import Featurizer

from sklearn.feature_selection import RFECV 

from sklearn.linear_model import LogisticRegression


def model_build(categorical_feat,
                numerical_feat,
                clf = LogisticRegression(solver   = 'sag', 
                                         max_iter = 1000),
                scr = 'roc_auc'
               ):
    
    featurizer = Featurizer(categorical_feat, numerical_feat)
    
    recursive_elimination = RFECV(clf, n_jobs = -1, cv = 3, 
                                  scoring = scr, step = 1)
    
    model = Pipeline([
        ('modeling', Pipeline([
            ('featurize', featurizer),
            ('reduction', recursive_elimination)
        ])),
        ('classify', clf)
    ])
    
    return model