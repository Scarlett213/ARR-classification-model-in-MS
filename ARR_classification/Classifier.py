from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
def chooseModel(name):

    if name=='SVM':
        model=SVC(probability=True)
    elif name=='RF':
        model= RandomForestClassifier()
        RandomForestClassifier.n_classes_=2
    elif name=='KNN':
        model= KNeighborsClassifier()
    elif name=='LR':
        model= LogisticRegression()
    return model

def Class(model,X,y,kf):
    y_scores=[]
    test_ys=[]
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        train_X, train_y = X[train_index], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        model.fit(train_X, train_y)
        y_score = model.predict_proba(test_X)
        y_scores.append(y_score)
        test_ys.append(test_y)
    return y_scores, test_ys

