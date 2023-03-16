import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy import interp

def Binary(y_scores):
    y_preds=[]
    for i in range(len(y_scores)):
        y_score= y_scores[i]
        y_pred= np.zeros(y_score.shape[0])
        for j in range(y_score.shape[0]):
            if y_score[j][0] > y_score[j][1]:
                y_pred[j]=0
            else:
                y_pred[j]=1
        y_preds.append(y_pred)
    return y_preds

def b(y):
    a=[]
    for i in y[:,1]:
        if i>=0.5:
            a.append(1)
        else:
            a.append(0)
    return np.array(a)

def AccAuc(y_scores, test_ys):
    Rtpr=[]
    Rfpr=[]
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 99)
    accs = []
    for i in range(len(y_scores)):
        y_score = y_scores[i]
        test_y=test_ys[i]
        fpr, tpr, thresholds = roc_curve(test_y, y_score[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        Rfpr.append(fpr)
        Rtpr.append(tpr)
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        y_label = b(y_score)
        acc = accuracy_score(test_y, y_label)
        aucs.append(roc_auc)
        accs.append(acc)
        i += 1
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    return [mean_acc, std_acc], [mean_auc, std_auc], Rtpr, Rfpr

def classfyIndex(y_preds, test_ys ):
    true=[]
    pred=[]
    for i in range(len(y_preds)):
        true.extend(test_ys[i])
        pred.extend(y_preds[i])
    p=precision_score(np.array(true), np.array(pred))
    r=recall_score(np.array(true), np.array(pred))
    f=f1_score(np.array(true), np.array(pred))
    return p, r, f


def specificityCalc(y_preds, test_ys):
    true=[]
    pred=[]
    for i in range(len(y_preds)):
        true.extend(test_ys[i])
        pred.extend(y_preds[i])
    tn, fp, fn, tp = confusion_matrix(np.array(true),np.array(pred)).ravel()
    Condition_negative = tn +fp + 1e-6
    Specificity = tn / Condition_negative
    return Specificity