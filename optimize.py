# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 12:12:15 2023

@author: 赵上
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score as r2
import matplotlib.pyplot as plt
#from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from statistics import mean
from statistics import variance
import time

####Data preprocessing

all_data = pd.read_csv("RHEAs.1000.HT.feature.add ite3.81data.12feature.csv")
all_data = all_data.drop(['samples'],axis=1)
y_all = all_data.iloc[:, 0]
feartures = ['D6', 'D14', 'D16', 'D18','D25','D36']

X_all = all_data[feartures]
print(y_all.shape, X_all.shape)

features_df = pd.read_csv("RHEAs_element property.csv")
features_df.iloc[:,1]
#Select elements and descriptors:V Ti Mo Nb Zr
features_df = features_df.iloc[[2,3,4,6,7],[0,3,7,8,9,13,18]]
features_df.shape 
#该函数计算特征
def fun(F, p):
     #F represent feature，P represent element
    feature_mean = (F[2]*p[0]+F[3]*p[1]+F[4]*p[2]+F[6]*p[3]+F[7]*p[4])*0.01 
    feature_delta = np.sqrt(0.01*p[0]*np.square(1-F[2]/feature_mean) +
                    0.01*p[1]*np.square(1-F[3]/feature_mean)+
                    0.01*p[2]*np.square(1-F[4]/feature_mean)+
                    0.01*p[3]*np.square(1-F[6]/feature_mean)+
                    0.01*p[4]*np.square(1-F[7]/feature_mean))
    return feature_mean, feature_delta

def schaffer4(p):
    #x1,x2,x3,x4,x5,x6 represent the elements,respectively
    x1 ,x2, x3 ,x4 ,x5 = p[0], p[1], p[2], p[3], p[4]
    D5, D6 = fun(features_df.iloc[:,1], p)
    D13, D14 = fun(features_df.iloc[:,2], p)
    D15, D16 = fun(features_df.iloc[:,3], p)
    D17, D18 = fun(features_df.iloc[:,4], p)
    D25, D26 = fun(features_df.iloc[:,5], p)
    D35, D36 = fun(features_df.iloc[:,6], p)
    
    feature = []
    feature.append(D6)
    feature.append(D14)
    feature.append(D16)
    feature.append(D18)
    feature.append(D25)
    feature.append(D36)    
    feature = pd.DataFrame(feature).T
    feature.columns = ['D6','D14','D16','D18','D25','D36']
    
    return feature

#####
def ucb_for2(p):
    
    aa = schaffer4(p)
    aa=np.array(aa)
    print('aa type is :', type(aa))
    predict_validation, predict_validation_std = model.predict(aa, return_std=True)
    print('predict_validation is :', predict_validation)
    print('predict_validation_std is :', predict_validation_std)
    ucb_validation = predict_validation + predict_validation_std
    for i in ucb_validation:
        a = -i
        print('a is :', a)
    return a


####
constraint_eq = [
    lambda x: 100 - x[0] - x[1] - x[2] - x[3] - x[4]
]

######Load the optimization function package
from sko.GA import GA
from sko.PSO import PSO
import random
import pandas as pd
from lolopy.learners import RandomForestRegressor
#from sklearn.ensemble import RandomForestRegressor

ga_best_rf_y_ucb_for =[]
ga_best_rf_x_ucb_for = []
ga_generation_best_rf_y_true = []
ga_generation_best_rf_y_i_need =[]

pso_best_y =[]
pso_best_x =[]
pso_generation_best_y =[]
#####optimize
    
ga_ucb_for_start_time = time.time()

np.random.seed(0)##After the initial population seed is changed, the optimization results are different from the initial population
######training
X_all_train = X_all
y_all_train = y_all
X_all_train = np.array(X_all_train)
y_all_train = np.array(y_all_train)
#RF model   
model = RandomForestRegressor(num_trees=1000)

model.fit(X_all_train,y_all_train,random_seed=2023)
predict_train = model.predict(X_all_train)
n=50
while n<=1000:
    ga_ucb_for = GA(func=ucb_for2, n_dim= 5, size_pop=n, max_iter=100, prob_mut=0.001, 
                    lb=[0, 0, 0, 1, 0], ub=[50, 50, 50, 50, 50],
                    constraint_eq=constraint_eq,
                    precision=[1e-0, 1e-0, 1e-0, 1e-0, 1e-0])
    best_x_ga_ucb_for, best_y_ga_ucb_for = ga_ucb_for.run()
    Y_history_3 = pd.DataFrame(ga_ucb_for.all_history_Y)

    ga_best_rf_y_ucb_for.append(best_y_ga_ucb_for)
    ga_best_rf_x_ucb_for.append(best_x_ga_ucb_for)
    ga_generation_best_rf_y_true.append(ga_ucb_for.generation_best_Y)
    ga_generation_best_rf_y_i_need.append(Y_history_3.min(axis=1).cummin())
    
    
    ######PSO
    pso = PSO(func=ucb_for2, n_dim=5, pop=n, max_iter=100, 
              # lb=[5, 5, 5, 5, 5], 
              lb=[0, 0, 0, 1, 0], ub=[50, 50, 50, 50, 50],
              w=0.8, c1=0.5, c2=0.5)
    pso.run() 
    #print('best_x_pso is ', pso.gbest_x, 'best_y_pso is', pso.gbest_y)
        
    
    data = pso.record_value['Y']
    data = np.array(data)
    data = np.reshape(data, (100, n)) #n表示粒子数
    Y_history_pso = pd.DataFrame(data)

        
    pso_best_y.append(pso.best_y)
    pso_best_x.append(pso.best_x)

    a=[]
    for j in range(len(pso.gbest_y_hist)):
        for k in list(pso.gbest_y_hist[j]):
            a.append(k)
        
    pso_generation_best_y.append(a)
    n = n+50
    
ga_ucb_for_end_time = time.time()
ga_ucb_for_run_time = ga_ucb_for_end_time - ga_ucb_for_start_time
print('ga_ucb_for_run_time is :', ga_ucb_for_run_time)
   


#####输出csv###
#####rf ucb######
ga_best_rf_x_ucb_for_ite1 = pd.DataFrame(ga_best_rf_x_ucb_for)
ga_best_rf_y_ucb_for_ite1 = pd.DataFrame(ga_best_rf_y_ucb_for)
ga_ucb_for_optimical_all_ite1 = pd.concat([ga_best_rf_x_ucb_for_ite1, ga_best_rf_y_ucb_for_ite1], 
                          axis=1, join='outer')
ga_ucb_for_optimical_all_ite1.columns = ['V','Ti','Mo','Nb','Zr','ucb']
ga_generation_best_rf_y_true_ite1 = pd.DataFrame(ga_generation_best_rf_y_true)
ga_generation_best_rf_y_true_ite1 = ga_generation_best_rf_y_true_ite1.T
ga_generation_best_rf_y_i_need_ite1 = pd.DataFrame(ga_generation_best_rf_y_i_need)
ga_generation_best_rf_y_i_need_ite1 = ga_generation_best_rf_y_i_need_ite1.T

pso_best_x_ite1 = pd.DataFrame(pso_best_x)
pso_best_y_ite1 = pd.DataFrame(pso_best_y)
#pso_best_rf_y_std_ite1 = pd.DataFrame(pso_best_rf_y_std)
pso_optimical_all = pd.concat([pso_best_x_ite1, pso_best_y_ite1], 
                          axis=1, join='outer')
pso_optimical_all.columns = ['V','Ti','Mo','Nb','Zr','ucb']

pso_generation_best_y_ite1 = pd.DataFrame(pso_generation_best_y)
pso_generation_best_y_ite1 = pso_generation_best_y_ite1.T


ga_ucb_for_optimical_all_ite1.to_csv('seed0_ga_ucb_for_optimical_all_ite1.csv')
ga_generation_best_rf_y_i_need_ite1.to_csv('ga_generation_best_rf_y_i_need_ite1.for.csv')

pso_optimical_all.to_csv('seed0_pso_optimical_all.csv')
pso_generation_best_y_ite1.to_csv('pso_generation_best_y_ite1.csv')


