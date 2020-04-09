from sklearn.metrics import roc_curve,    \
                            roc_auc_score, \
                            classification_report, \
                            precision_recall_fscore_support \

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn

import itertools

### --------------------------------------------------------------------
def roc_auc(y_prob, y_test):
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    thresholds[thresholds > 1] = 1

    plot_tone = plt.cm.jet(thresholds)

    cor_map = plt.cm.ScalarMappable(cmap=plt.cm.jet_r)
    cor_map.set_array(thresholds)

    f, ax = plt.subplots(figsize=(8, 5))

    ax.plot(fpr, tpr, linestyle='--', c = 'k', zorder=1)
    ax.scatter(fpr, tpr, c = plot_tone, zorder=2)
    cb = f.colorbar(cor_map)
    cb.ax.tick_params(labelsize = 14)
    cb.set_label('probability of complaint', fontsize = 16, color = 'k')
    ax.plot([0, 1], [0, 1], linestyle='dotted')
    plt.xlabel('False Positive Rate', fontsize = 18, color='black')
    plt.ylabel('True Positive Rate', fontsize = 18, color='black')
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    auc = roc_auc_score(y_test, y_prob)

    return auc, f


### --------------------------------------------------------------------
def plot_roc_auc(y_prob, y_test):

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    thresholds[thresholds > 1] = 1

    plot_tone = plt.cm.jet(thresholds)

    cor_map = plt.cm.ScalarMappable(cmap=plt.cm.jet_r)
    cor_map.set_array(thresholds)

    f, ax = plt.subplots(figsize=(8, 5))

    ax.plot(fpr, tpr, linestyle='--', c = 'k', zorder=1);
    ax.scatter(fpr, tpr, c = plot_tone, zorder=2);
    cb = f.colorbar(cor_map)
    cb.ax.tick_params(labelsize = 14)
    cb.set_label('probability of complaint', fontsize = 16, color = 'k')
    ax.plot([0, 1], [0, 1], linestyle='dotted')
    plt.xlabel('False Positive Rate', fontsize = 18, color='black')
    plt.ylabel('True Positive Rate', fontsize = 18, color='black')
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    auc = roc_auc_score(y_test, y_prob)

    return auc, f

def report(model):
    y_pred = [i['prediction'] for i in model.select('prediction').collect()]
    y_test = [i['label'] for i in model.select('label').collect()]

    print('Classification Report\n')
    print(classification_report(y_test, y_pred))

    p, r, f, s = precision_recall_fscore_support(y_test, y_pred, labels=[1])

    return {'precision': p[0], 'recall': r[0], 'fscore': f[0], 'support': s[0]}

def feature_importance(model_pipeline, test_df, 
                       do_print = True,
                       drop     = [],
                       n_feat   = 15):

    array_coef = np.asarray(model_pipeline.stages[-1].coefficients)

    array_features = test_df.select('features').collect()

    h = len(array_features)
    l = array_features[0][0].size
    matrix_features = np.zeros([h,l])

    i = 0
    for af in array_features:
        matrix_features[i] = af['features'].toArray()
        i += 1

    array_variance = np.var(matrix_features, axis = 0)
    factor = array_coef*array_variance

    list_col = model_pipeline.stages[0].getInputCols().copy()

    for i in range(5):
        feat_name = model_pipeline.stages[1].stages[i+1].getInputCol()
        feat_cat = model_pipeline.stages[1].stages[i+1].labels

        col_name = [feat_name +'_'+ c for c in feat_cat]

        list_col += col_name + [feat_name + '_unknow']

    drop.sort(reverse = True)
    for i in drop:
        del list_col[i-1]

    df = pd.DataFrame({'feature' : list_col,
                    'variance': array_variance})
    # df.drop(drop, inplace = True)
    
    array_variance = df['variance'].values
    
    factor = array_coef*array_variance
    
    df['factor'] = factor
    df['module'] = abs(factor)
                      
    df = df.sort_values('module', ascending = False)

    fig = None
    if do_print:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 0.5*n_feat))
        sbn.barplot(
                    x = 'factor',
                    y = 'feature' , 
                    data = df.head(n_feat),
                    palette="Blues_d"
                    )

        plt.xlabel('Importance Factor', fontsize = 18, color='black')
        plt.ylabel('', fontsize = 1, color='black')
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
    
    return df, fig


def plot_feature_importance(df,
                       drop     = [],
                       n_feat   = 15):

    df = df.sort_values('module', ascending = False)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 0.5*n_feat))
    sbn.barplot(
                x = 'factor',
                y = 'feature' , 
                data = df.head(n_feat),
                palette="Blues_d"
                )

    plt.xlabel('Importance Factor', fontsize = 18, color='black')
    plt.ylabel('', fontsize = 1, color='black')
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    
    return df, fig

def plot_confusion_matrix(cm,
                          cmap=None,
                          normalize=True):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    title='Confusion matrix',

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()