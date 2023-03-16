from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from ReliefF import *

def chooseFea(name, radiomics, y):
    if name=='RFE':
        feature=RFE(radiomics,y)
    elif name=='ReliefF':
        feature=ReliefF(radiomics, y )
    elif name=='LASSO':
        feature=LASSO(radiomics, y)
    return feature


def RFE(radiomics,y):
    model_Linear_SVM = SVC(kernel='linear', probability=True)
    rfecv = RFECV(estimator=model_Linear_SVM)
    rfecv = rfecv.fit(radiomics, y)
    feature= radiomics.columns[rfecv.support_]
    return feature

def ReliefF(radiomics,y):
    radiomics.insert(radiomics.shape[1], 'LABEL', y)
    f = Relief(radiomics, 1, 0.01, 1)
    f.reliefF()
    feature=f.get_final()
    feature=list(feature.index)
    return feature

def LASSO(radiomics, y):
    lasso = Lasso()
    lasso.fit(radiomics.values, y)
    mask = lasso.coef_ != 0
    new_reg_data = radiomics.iloc[:, mask]
    feature=new_reg_data.columns
    return feature

