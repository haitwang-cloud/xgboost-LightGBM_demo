import numpy as np
import time
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV


data_np=np.array(pd.read_csv('../xgboost_demo/dataSetCon_uci.csv'))


X=preprocessing.scale(np.array([line[:-1] for line in data_np]))
y=np.array([line[-1] for line in data_np])

xgb_model=XGBClassifier(n_estimators=5000,nthread=4,
                        silent=False,objective='multi:softmax',
                        scale_pos_weight=1,max_depth=3,min_child_weight=2,
                        seed=1993,gamma=4.4,colsample_bytree=0.1,subsample=0.1,
                        learning_rate=0.1)

start_time=time.clock()
# param_grid_xgboost=dict()
# start_time=time.clock()
# grid_xgboost=GridSearchCV(xgb_model,param_grid_xgboost,cv=10,scoring='accuracy')
# grid_xgboost.fit(X,y)
# endtime=time.clock()
# print('score',grid_xgboost.grid_scores_)
# print('Xgboost_best_estimator_',grid_xgboost.best_estimator_)
# print('Xgboost_best_score_',grid_xgboost.best_score_)
# print('Xgboost_best_params_',grid_xgboost.best_params_)
# print("run_time",endtime-start_time)
score_all=0
kf=KFold(n_splits=10,shuffle=True)
for train,test in kf.split(X):
    X_train=X[train]
    X_test=X[test]
    y_train=y[train]
    y_test=y[test]
    xgb_model.fit(X_train,y_train)
    preds=xgb_model.predict(X_test)
    score=np.sum(preds==y_test)/len(y_test)
    print("score:",score)
    score_all=score_all+score
print("score_all",score_all/10)
endtime=time.clock()
print("run_time",endtime-start_time)

