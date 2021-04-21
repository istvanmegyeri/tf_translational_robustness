from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np


def avg_auroc(y, p):
    aucs = []
    for i in range(y.shape[1]):
        fpr, tpr, thresholds = roc_curve(y[:, i], p[:, i])
        v = auc(fpr, tpr)
        aucs.append(v)
    return np.nanmean(aucs)


def avg_auprc(y, p):
    aucs = []
    for i in range(y.shape[1]):
        precision, recall, thresholds = precision_recall_curve(y[:, i], p[:, i])
        v = auc(recall, precision)
        aucs.append(v)
    return np.nanmean(aucs)
