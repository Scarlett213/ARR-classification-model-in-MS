from Classifier import *
import pandas as pd
import os
from sklearn.model_selection import KFold
import numpy as np
from options import ClassificaitonOptions
from func import *

if __name__ =='__main__':
    options = ClassificaitonOptions()
    opts=options.parse()
    path=opts.csv_path
    radiomics = pd.read_csv(path)
    path_ARR= opts.ARR_path
    y = pd.read_excel(path_ARR)
    pathtxt =opts.save_path
    featureSelCor = np.loadtxt(pathtxt, dtype=np.str)
    cls = opts.classifiers
    kf = KFold(n_splits=5, shuffle=True)
    model=chooseModel(cls)
    x=pd.DataFrame(radiomics,columns=featureSelCor).values
    y_scores, test_ys=Class(model, x, y, kf)
    y_preds=Binary(y_scores)
    acc, auc, Rtpr, Rfpr =AccAuc(y_scores, test_ys)
    precision, recall, f1= classfyIndex(y_preds, test_ys)
    Specificity= specificityCalc(y_preds, test_ys)
