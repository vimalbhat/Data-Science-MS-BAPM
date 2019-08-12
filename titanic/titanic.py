# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:00:27 2019

@author: Vimal Bhat
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

fileData=pd.read_csv("train.csv")
#majority empty values
fileData.drop(columns=["Cabin"],inplace=True)
#dropping empty 2 values
fileData.dropna(axis=0,subset=["Embarked"],inplace=True)

#imputing age
from sklearn.impute import SimpleImputer
simpleImputer=SimpleImputer(missing_values=np.nan, strategy='mean')
fileData.loc[:,["Age"]]=simpleImputer.fit_transform(fileData.loc[:,["Age"]])






y=fileData.iloc[:,1].values
X=fileData.iloc[:,[2,4,5,6,7,9,10]].values

#Preprocessing Test Data
fileData_test=pd.read_csv("test.csv")
fileData_test.drop(columns=["Cabin"],inplace=True)
#dropping empty 2 values
#fileData_test.dropna(axis=0,subset=["Embarked","Fare"],inplace=True)
#imputing age
from sklearn.impute import SimpleImputer
simpleImputer=SimpleImputer(missing_values=np.nan, strategy='mean')
fileData_test.loc[:,["Age"]]=simpleImputer.fit_transform(fileData_test.loc[:,["Age"]])
#y_test=fileData.iloc[:,1].values
X_test=fileData_test.iloc[:,[1,3,4,5,6,8,9]].values

#pd.isnull(fileData_test).sum()
#fileData_test.isnull().sum()

from sklearn.preprocessing import LabelEncoder
labelEncoder=LabelEncoder()
labelEncoder.fit(X[:,[1]])
X[:,1]=labelEncoder.transform(X[:,1])
labelEncoder1=LabelEncoder()
labelEncoder1.fit(X[:,[6]])
X[:,6]=labelEncoder1.transform(X[:,6])
labelEncoder1.fit_transform

#test Data

labelEncoder=LabelEncoder()
labelEncoder.fit(X_test[:,[1]])
X_test[:,1]=labelEncoder.transform(X_test[:,1])
labelEncoder1=LabelEncoder()
labelEncoder1.fit(X_test[:,[6]])
X_test[:,6]=labelEncoder1.transform(X_test[:,6])



from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=0)


import sys
sys.path.insert(0,"D:/Udemy/Machine Learning A-Z New/Custom/parallelProcessingModel.py")

import parallelProcessingModel as ppm
pas=ppm.machineModelParallel()
func=classifier.predict
pas.addModel(func,X_test)
from multiprocessing import Process
def test(a):
    return False

p1=Process(target=func,args=(X_test,))

abs=p1.start()

classifier.fit(X,y)
pd.isnull(fileData).sum()
pas.argsForModel
pas.executeModel[0]

df=pd.DataFrame(X_test)
df.reset_index()
y_pred=classifier.predict(X_test)
X_test=X_test.astype('float64') 
submission=np.array(fileData_test_res.iloc[:,0].values)
submission=submission.reshape(-1,1)
y_pred=y_pred.reshape(-1,1)
newArray=np.append(submission,y_pred,axis=1)

pd.DataFrame(data=newArray).to_csv("submission.csv")

fileData_test_res=pd.read_csv("gender_submission.csv")
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(fileData_test_res.iloc[:,1].values,y_pred)


