# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import numpy as np
# 读取train.txt
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submit=pd.read_csv('sample_submit.csv')

y_train=train.pop('y')
# lasso_param={'alpha':np.arange(0.1,1,0.1)}
# lasso_clf=linear_model.Lasso()
# lasso_GS=GridSearchCV(lasso_clf,lasso_param,cv=5)
# lasso_GS.fit(train,y_train)
# print('lasso_best_estimator_',lasso_GS.best_estimator_)
# print('lasso_best_score_',lasso_GS.best_score_)
# print('lasso_best_params_',lasso_GS.best_params_)
lasso_clf=linear_model.Lasso(alpha=0.1)
lasso_clf.fit(train,y_train)
lasso_train=lasso_clf.predict(train)
print(r2_score(y_train,lasso_train))
# lasso_pred=lasso_clf.predict(test)
# # 若预测值是负数，则取0
# lasso_pred = map(lambda x: x if x >= 0 else 0, lasso_pred)
# submit['y']=list(lasso_pred)
# submit.to_csv('LassoPrediction.csv',index=False)

EN_clf=ElasticNet(alpha=0.1,l1_ratio=0.9)
# EN_param={'alpha':np.arange(0.1,1,0.1),
#           'l1_ratio':np.arange(0.1,1,0.1)
#           }
# EN_GS=GridSearchCV(EN_clf,EN_param,cv=5)
# EN_GS.fit(train,y_train)
# print('lasso_best_estimator_',EN_GS.best_estimator_)
# print('lasso_best_score_',EN_GS.best_score_)
# print('lasso_best_params_',EN_GS.best_params_)

EN_clf.fit(train,y_train)
EN_train=EN_clf.predict(train)
print(r2_score(y_train,EN_train))
EN_pred=EN_clf.predict(test)
# 若预测值是负数，则取0
EN_pred = map(lambda x: x if x >= 0 else 0, EN_pred)
submit['y']=list(EN_pred)
submit.to_csv('ENPrediction.csv',index=False)
