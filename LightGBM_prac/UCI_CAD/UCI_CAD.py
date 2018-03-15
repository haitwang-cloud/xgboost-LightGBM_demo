import numpy as np
import time
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score

data_np=np.array(pd.read_csv('./UCI_CAD.csv'))


X=preprocessing.scale(np.array([line[:-1] for line in data_np]))
y=np.array([line[-1] for line in data_np])

# specify your configurations as a dict


lgb=LGBMClassifier()



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