import pandas as pd
import numpy as np
import random
data_pd=pd.read_csv('../xgboost_demo/UCI_CAD.csv',header='infer',encoding='utf-8')
data_np=np.array(data_pd)
random.shuffle(data_np)
dataSet=[]
for line in data_np:
    dataSet.append(list(line))
dataSetCon=[]
for vet in dataSet:
    vetCon=[]
    Age,Sex,CP,Trestbps,chol,fbs,restecg,Thalach,Exang,Oldpeak,Slope,Ca,Thal,Num=vet
    if Age<40:
        Age_state=1
    if 40<=Age<60:
        Age_state=2
    if 60<=Age<80:
        Age_state=3
    vetCon.append(Age_state)
    vetCon.append(Sex)
    vetCon.append(CP)
    if(Trestbps<=100):
        Trestbps_state=1
    if(100<Trestbps<=120):
        Trestbps_state=2
    if(120<Trestbps<=140):
        Trestbps_state=3
    if(140<Trestbps<=160):
        Trestbps_state=4
    if(160<Trestbps<=180):
        Trestbps_state=5
    if(180<Trestbps):
        Trestbps_state=6              
    vetCon.append(Trestbps_state)                       
    if chol<200:
        chol_state=1
    if 200<=chol<250:
        chol_state=2
    if 250<=chol<300:
        chol_state=3
    if 300<=chol:
        chol_state=4
    vetCon.append(chol_state)
    vetCon.append(fbs)
    vetCon.append(restecg)          
    if Thalach<=100:
        Thalach_state=1
    if 100<Thalach<120:
        Thalach_state=2
    if 120<=Thalach<140:
        Thalach_state=3
    if 140<=Thalach<160:
        Thalach_state=4
    if 160<=Thalach:
        Thalach_state=5    
    vetCon.append(Thalach_state)
    vetCon.append(Exang)
    if Oldpeak<=0:
        Oldpeak=1
    if 0<Oldpeak<1:
        Oldpeak=2
    if 1<Oldpeak<2:
        Oldpeak=3
    if 2<=Oldpeak:
        Oldpeak=4
    vetCon.append(Oldpeak)
    vetCon.append(Slope)
    vetCon.append(Ca)
    vetCon.append(Thal)
    vetCon.append(Num)                                                                          
    dataSetCon.append(vetCon)
dataSetCon=pd.DataFrame(dataSetCon)
dataSetCon.to_csv('../xgboost_demo/dataSetCon_uci.csv',header=False,index=False)
print(dataSetCon)

