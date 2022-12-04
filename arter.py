#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 18:01:37 2022

@author: okanincemd
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import cross_val_score, cross_validate
import warnings
warnings.filterwarnings('ignore')

clin = pd.read_excel("......xlsx")
df = pd.read_excel("........xlsx")

x = df.iloc[:,:-1]
y = df.iloc[:,-1]


#%% Time
from time import time

#%% Preprocessing

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = sc.fit_transform(x)


#%% Combination with Clinical Features

clin_df = pd.concat([X1,clin], axis = 1)
clin_df.drop("Response", axis = 1, inplace = True)

#%% Train-test split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size= 0.33, 
                                                 random_state= 42)

X3_train,X3_test,y3_train,y3_test = train_test_split(clin_df,y,
                                                 test_size= 0.33,
                                                 random_state=42)

#%% SVM / X1
from sklearn.metrics import classification_report , confusion_matrix, roc_auc_score 
from sklearn.svm import SVC

svm1 = SVC(C = 1.7, kernel  ="rbf", gamma = "auto", shrinking = False,
         max_iter = -1, random_state = 42, probability=True )

t_svm0 = time()
svm1.fit(X_train,y_train)
svm_time = time() - t_svm0

y_pred1 = svm1.predict_proba(X_test)

print("********* SVM X1 Train Score ************")
print(classification_report(y_train, svm1.predict(X_train)))
print(confusion_matrix(y_train, svm1.predict(X_train)))
print("********* SVM X1 AUC Score ************")
print(roc_auc_score(y_train, svm1.predict(X_train)))
print("******************")
print("******** SVM X1  TEST TEST ********")
print(classification_report(y_test, svm1.predict(X_test)))
print("********** SVM X1 Confusion Matrix ***********")
print(confusion_matrix(y_test, svm1.predict(X_test)))
print("********* SVM X1 AUC Score ************")
print(roc_auc_score(y_test,y_pred1[:,1]))

#%% SVM / X3
from sklearn.metrics import classification_report , confusion_matrix, roc_auc_score 
from sklearn.svm import SVC

svm3 = SVC(C = 4, kernel  ="rbf", gamma = "auto", shrinking = False,
         max_iter = -1, random_state = 42, probability=True )

t_svm0 = time()
svm3.fit(X3_train,y3_train)
svm3_time = time() - t_svm0

y_pred3 = svm3.predict_proba(X3_test)

print("********* SVM X1 Train Score ************")
print(classification_report(y3_train, svm3.predict(X3_train)))
print("******************")
print("******** SVM X1  TEST TEST ********")
print(classification_report(y3_test, svm3.predict(X3_test)))
print("********** SVM X1 Confusion Matrix ***********")
print(confusion_matrix(y3_test, svm3.predict(X3_test)))
print("********* SVM X1 AUC Score ************")
print(roc_auc_score(y3_test,y_pred3[:,1]))

#%% viz
from sklearn.metrics import RocCurveDisplay, auc

fig, ax = plt.subplots(figsize = (12,8))

viz1 = RocCurveDisplay.from_estimator(svm1, X_test,y_test,
                                      name = "Radiomics features",
                                      alpha = .8,
                                      lw = 1,
                                      ax = ax)

viz2 = RocCurveDisplay.from_estimator(svm3, X3_test, y3_test,
                                      name = "w/Clinical features",
                                      alpha = .8,
                                      lw = 1,
                                      ax = ax)

ax.plot([0,1],[0,1], linestyle = "--",
        lw = 2, color = "b", alpha = 0.8)

ax.set(
       xlim = [-0.05,1.05],
       ylim = [-0.05,1.05])

ax.set_title("CE-T1 Predictions", fontsize = 15)
ax.set_xlabel("1-Specificity", fontsize = 15)
ax.set_ylabel("Sensitivity", fontsize = 15)
plt.legend(loc = "lower right")
plt.savefig("CE-T1 ROC.png", format = "png", dpi = 300)
plt.show()



#%% Calibration plot

from sklearn.calibration import calibration_curve

prob_true1, prob_pred1 = calibration_curve(y_test, y_pred1[:,1], n_bins=5)
prob_true2, prob_pred2 = calibration_curve(y3_test, y_pred3[:,1], n_bins=5)


fig,ax = plt.subplots(figsize = (12,8))

ax.plot([0,1],[0,1], linestyle = "--", lw =2, 
        label = "Perfectly Calibrated",
        color = "b", alpha = .8)

ax.set(
       xlim = [-0.05,1.05],
       ylim = [-0.05,1.05])

ax.plot(prob_pred1, prob_true1, marker = ".", 
        label = "Radiomics features", lw = 1, alpha = .8)
ax.plot(prob_pred2, prob_true2, marker = ".",
        label = "w/Clinical Features", lw = 1, alpha = .8)

ax.set_title("CE-T1 Calibration Plot", fontsize = 15)
ax.set_xlabel("Average Predicted Probability", fontsize = 15)
ax.set_ylabel("Ratio of Positives", fontsize = 15)
plt.legend(loc = "lower right")
plt.savefig("Arter Calibration.png", format = "png", dpi = 300)
plt.show()








