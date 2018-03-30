import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.grid_search import GridSearchCV
import time

data_train=pd.read_csv('dataset.csv')
y_train=data_train.pop('follower')
X_train=data_train
print(X_train.shape,y_train.shape)

#构建XGboost模型
model_xgb=xgb.XGBRegressor()
param_xgb_list={
    "max_depth":[10],
    "min_child_weight":[3],
    "n_estimators":[200],
    "learning_rate":[0.05]

}
grid_xgb=GridSearchCV(model_xgb,param_grid=param_xgb_list,cv=4,verbose=10,n_jobs=-1,scoring='r2')
start_time=time.clock()
grid_xgb.fit(X_train,y_train)
endtime=time.clock()
print('Xgboost_best_estimator_',grid_xgb.best_estimator_)
print('Xgboost_best_score_',grid_xgb.best_score_)
print('Xgboost_best_params_',grid_xgb.best_params_)
print("run_time",endtime-start_time)

#构建LightGBM模型
model_lgb=lgb.LGBMRegressor()
param_lgb_list={
    "max_depth":[10],
    "n_estimators":[200],
    "learning_rate":[0.05],
    "num_leaves":[600],

}
grid_lgb=GridSearchCV(model_lgb,param_grid=param_lgb_list,cv=4,verbose=10,n_jobs=-1,scoring='r2')
start_time=time.clock()
grid_lgb.fit(X_train,y_train)
endtime=time.clock()
print('LightGBM_best_estimator_',grid_lgb.best_estimator_)
print('LightGBM_best_score_',grid_lgb.best_score_)
print('LightGBM_best_params_',grid_lgb.best_params_)
print("run_time",endtime-start_time)