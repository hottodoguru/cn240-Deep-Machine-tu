import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , roc_curve,plot_roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
from matplotlib.patches import Patch
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import pickle


dataset = "other_vs_non_other.csv"
feature = ["cdr" , "exudate" , "label"]
glaucoma_data = pd.read_csv(dataset , names = feature)

skf = StratifiedKFold(n_splits=5)

#### GaussianProcessClassifier
model = []
tprs = []
aucs = []
fold_no = 1
kernel = 1.0 * RBF(1.0)
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
gpc = GaussianProcessClassifier(kernel=kernel,random_state=0)

label = glaucoma_data.label.values
X_data = glaucoma_data.drop('label',axis=1)


trainData,testData,trainType,testType = train_test_split(X_data,label,test_size=0.2,random_state=100)
for train,test in skf.split(trainData,trainType):
    x_train = trainData.values[train]
    y_train = trainType[train]
    X_test = trainData.values[test]
    y_test = trainType[test]
    gpc.fit(x_train,y_train)
    
    ############# viz plot 
    viz = plot_roc_curve(gpc, X_test, y_test,
                         name='ROC fold {}'.format(fold_no),
                         ax=ax)

    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    # Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, gpc.predict_proba(X_test)[:,1])
    # Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,gpc.predict(X_test))
    # Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (y_test, auc))
    conf_matrix = confusion_matrix(y_test,gpc.predict(X_test))
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    Specificity = TN/(TN+FP)
    sensitivity = TP / (TP + FN) ##recall
    Precision = TP / (TP+FP)
    F1 = 2*((Precision * sensitivity)/(Precision + sensitivity))
    
    print('Fold Number : %i ' %fold_no )
    print('Accuracy : ',Accuracy)
    print('Specificity : ',Specificity)
    print('Sensitivity : ',sensitivity)
    print("Precision: ",Precision )
    print("F1-Score: ", F1)
    
    print("FP: ",FP)
    print("FN: ",FN)
    print("TP: ",TP)
    print("TN: ",TN,'\n')

    
    filename = "other_model" + str(fold_no) + ".sav"
    pickle.dump(gpc, open(filename, 'wb'))
    fold_no += 1

### KNN
model = []
tprs = []
aucs = []
fold_no = 1
kernel = 1.0 * RBF(1.0)
mean_fpr = np.linspace(0, 1, 100)
i = 0
fig, ax = plt.subplots()
knn_clf=KNeighborsClassifier(n_neighbors=2)
label = glaucoma_data.label.values
X_data = glaucoma_data.drop('label',axis=1)

trainData,testData,trainType,testType = train_test_split(X_data,label,test_size=0.2,random_state=100)
for train,test in skf.split(trainData,trainType):
    x_train = trainData.values[train]
    y_train = trainType[train]
    X_test = trainData.values[test]
    y_test = trainType[test]
    gpc.fit(x_train,y_train)
    
    ############# viz plot 
    viz = plot_roc_curve(knn_clf, X_test, y_test,
                         name='ROC fold {}'.format(fold_no),
                         ax=ax)

    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    # Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, knn_clf.predict_proba(X_test)[:,1])
    # Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,knn_clf.predict(X_test))
    # Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (y_test, auc))
    conf_matrix = confusion_matrix(y_test,knn_clf.predict(X_test))
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    Specificity = TN/(TN+FP)
    sensitivity = TP / (TP + FN) ##recall
    Precision = TP / (TP+FP)
    F1 = 2*((Precision * sensitivity)/(Precision + sensitivity))
    print('Fold Number : %i ' %fold_no )
    print('Accuracy : ',Accuracy)
    print('Specificity : ',Specificity)
    print('Sensitivity : ',sensitivity)
    print("Precision: ",Precision )
    print("F1-Score: ", F1)
    
    print("FP: ",FP)
    print("FN: ",FN)
    print("TP: ",TP)
    print("TN: ",TN,'\n')

    filename = "other_model" + str(fold_no) + ".sav"
    pickle.dump(gpc, open(filename, 'wb'))
    fold_no += 1
