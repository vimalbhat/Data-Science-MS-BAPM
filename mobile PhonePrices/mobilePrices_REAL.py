# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 13:26:39 2019

@author: Vimal Bhat
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib qt
train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")
train_data.info()
train_data.describe()
X=train_data.loc[:,['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
       'touch_screen', 'wifi', ]]
y=train_data.loc[:,['price_range']]

from sklearn.feature_selection import SelectKBest,chi2
bestFeatures=SelectKBest(chi2,k=10)
bestFeatures.fit(X,y)
columnsDf=pd.DataFrame({"columns":X.columns.values,"pValues":bestFeatures.pvalues_,"Scores":bestFeatures.scores_})
selected=columnsDf.nlargest(10,"Scores")
X=train_data[selected["columns"]]
import seaborn as sns
corrmat=train_data.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g= sns.heatmap(train_data.corr(),annot=True,cmap='RdYlGn')

from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X,y)
X_test=test_data[selected["columns"]]
y_pred=classifier.predict(X_test)