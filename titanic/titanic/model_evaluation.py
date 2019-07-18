import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.metrics import confusion_matrix, \
                            roc_curve, \
                            roc_auc_score, \
                            classification_report

def auc_roc(model, y_prob, y_test):
    probs  = y_prob[:, 1]

    auc = roc_auc_score(y_test, probs)

    fpr, tpr, thresholds = roc_curve(y_test, probs)
    thresholds[thresholds > 1] = 1

    plot_tone = plt.cm.jet(thresholds)

    cor_map = plt.cm.ScalarMappable(cmap=plt.cm.jet_r)
    cor_map.set_array(thresholds)

    f, ax = plt.subplots(figsize=(8, 5))

    plt.plot(fpr, tpr, linestyle='--', c = 'k', zorder=1);
    plt.scatter(fpr, tpr, c = plot_tone, zorder=2);
    cb = f.colorbar(cor_map)
    cb.ax.tick_params(labelsize = 14)
    cb.set_label('Threshold Probability', fontsize = 16, color = 'k')
    plt.plot([0, 1], [0, 1], linestyle='dotted')
    plt.xlabel('False Positive Rate', fontsize = 18, color='black')
    plt.ylabel('True Positive Rate', fontsize = 18, color='black')
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    plt.show()
    
    return auc

# is_in_model = model.named_steps['modeling'].steps[1][1].support_