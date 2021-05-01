import sys, os
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Tuple

def top2_accuray_lgb(predt: np.ndarray, data: lgb.Dataset, threshold: float=0.5,) -> Tuple[str, float, bool]:
    s_0=31
    s_1=int(len(predt)/s_0)
    
    predt = predt.reshape(s_0, s_1)
    y = data.get_label()
    p = predt.argsort(axis=0)[::-1,:]
    accuracy = ((y==p[0,:])|(y==p[1,:])).mean()

    # # eval_name, eval_result, is_higher_better
    return 'top2_accuray', float(accuracy), True

def softmax(predt):
    return np.exp(predt)/np.sum(np.exp(predt), axis=1).reshape(-1,1)

def my_logistic_obj(predt: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    w_dict = {4.0: 0.7590094259401983, 12.0: 0.8095400631514924, 13.0: 0.8225826394000231, 14.0: 0.8549528042689977, 30.0: 0.8549528042689977, 11.0: 0.8790811910945177, 0.0: 0.8797399286456303, 6.0: 0.8981458256286846, 19.0: 0.91289269651962, 8.0: 0.9193468859159446, 23.0: 0.9257599332783142, 9.0: 0.9466869195345441, 28.0: 0.9616129930949363, 5.0: 0.9636161932936556, 22.0: 0.9670600916375024, 27.0: 0.9896031798239419, 20.0: 0.998140820116188, 15.0: 1.0347287996927568, 10.0: 1.0440428983274967, 17.0: 1.0568262071050216, 2.0: 1.060630426576832, 24.0: 1.0737709894542427, 7.0: 1.083904634845265, 21.0: 1.0942847373031122, 16.0: 1.0961254610380533, 1.0: 1.098000827862746, 26.0: 1.1133105616315027, 3.0: 1.199290359855132, 25.0: 1.2044053703071544, 18.0: 1.227637388888927, 29.0: 1.2703169414985653}
    class_num=31
    data_size=int(len(predt)/class_num)
    y = data.get_label().astype(int)
    idx = np.arange(0, data_size).reshape(-1, 1)
    
    predt = predt.reshape(class_num, data_size)
    predt = np.transpose(predt)
    predt = softmax(predt)
    
    
    #grad
    dy_dx = predt.copy()
    dy_dx*=-predt[idx, y.reshape(-1, 1)]
    dy_dx[idx, y.reshape(-1, 1)]+=predt[idx, y.reshape(-1, 1)]
    df_dy = np.tile(-1/predt[idx, y.reshape(-1, 1)], (1, 31))
    grad=df_dy*dy_dx
    
    #hess
    d2y_dx2 = predt.copy()
    d2y_dx2=d2y_dx2*(2*d2y_dx2-1)
    d2y_dx2*=predt[idx, y.reshape(-1, 1)]
    d2y_dx2[idx, y.reshape(-1, 1)]-=predt[idx, y.reshape(-1, 1)]
    d2f_dxdy = (df_dy**2)*dy_dx
    hess = df_dy*d2y_dx2 + d2f_dxdy*dy_dx

    for key, value in w_dict.items():
        grad[y==key]*=value
        hess[y==key]*=value
    
    hess = np.transpose(hess).flatten()
    grad = np.transpose(grad).flatten()
    
    
    return grad, hess