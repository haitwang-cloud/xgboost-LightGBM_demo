import numpy as np
import time
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

data_np=np.array(pd.read_csv('./UCI_CAD.csv'))


X=np.array([line[:-1] for line in data_np])
y=np.array([line[-1] for line in data_np])

lgb=LGBMClassifier(objective='multiclass',boosting_type='goss',num_leaves=10,
                    max_depth=8,n_estimators=29,learning_rate=0.1,subsample_for_bin=800,n_jobs=4)
# # specify your configurations as a dict
# param_grid_xgboost={'min_child_samples':np.arange(10,100,10)}
# start_time=time.clock()
# grid_lgb=GridSearchCV(lgb,param_grid_xgboost,cv=5,scoring='accuracy')
# grid_lgb.fit(X,y)
# endtime=time.clock()
# print('score',grid_lgb.grid_scores_)
# print('Xgboost_best_estimator_',grid_lgb.best_estimator_)
# print('Xgboost_best_score_',grid_lgb.best_score_)
# print('Xgboost_best_params_',grid_lgb.best_params_)
# print("run_time",endtime-start_time)


start_time=time.clock()
score_all=0
kf=KFold(n_splits=5,shuffle=True)
for train,test in kf.split(X):
    print(len(train),len(test))
    X_train=X[train]
    X_test=X[test]
    y_train=y[train]
    y_test=y[test]
    lgb.fit(X_train,y_train)
    preds=lgb.predict(X_test)
    score=accuracy_score(y_test,preds)
    print("score:",score)
    score_all=score_all+score
print("score_all",score_all/5)
endtime=time.clock()
print("run_time",endtime-start_time)