import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import  roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold


def SaveFea(list_variable,path):
    file = open(path,'w')
    for i in list_variable:
        file.write(str(i)+'\n')
    file.close()



def SVMAUC(data, y):
    X = data
    kf = KFold(n_splits=25)
    rr = np.zeros([25, 1], dtype='float32')
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        train_X, train_y = X[train_index], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        clf = OneVsRestClassifier(SVC(kernel="linear", probability=True))
        clf.fit(train_X, train_y)
        y_scores = clf.predict_proba(test_X)
        rr[i] = y_scores
    return roc_auc_score(y, rr)

def auc1Fea(featureTest, radiomics, y):
    data = pd.DataFrame(radiomics, columns=[featureTest]).values
    auc =  SVMAUC(data,y)
    return auc

def DeleteCors(radiomics,feature,y, Ryuzhi=0.85):
    df=pd.DataFrame(radiomics,columns=feature)
    rdata = df.values
    R = np.abs(np.corrcoef(rdata, rowvar=False))
    R_tri = np.triu(R, k=1)
    index = np.argwhere(((R_tri >= Ryuzhi) | (R_tri <= -Ryuzhi)) )
    while index.shape[0]!=0:
        highCor= set(index[:,0])|set(index[:,1])
        LowCorId= [i for i in range(R.shape[0]) if i not in highCor]
        title = list(df.columns)
        keeps=[]
        for i in index:
            featureTest0=title[i[0]]
            auc0= auc1Fea(featureTest0, radiomics,y)
            featureTest1=title[i[1]]
            auc1=auc1Fea(featureTest1, radiomics,y)
            if auc0>auc1:
                keeps.append(featureTest0)
            elif auc0<auc1:
                keeps.append(featureTest1)
        for j in LowCorId:
            keeps.append(title[j])
        keeps=set(keeps)
        df = pd.DataFrame(radiomics, columns=keeps)
        rdata=df.values
        R = np.abs(np.corrcoef(rdata, rowvar=False))
        R_tri = np.triu(R, k=1)
        index = np.argwhere(((R_tri >= Ryuzhi) | (R_tri <= -Ryuzhi)) & (R_tri != 1))
    return keeps
