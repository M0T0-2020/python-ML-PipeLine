import os, sys
import gc
import numpy as np
import pandas as pd
import time, random, math
import pickle

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
from sklearn import metrics

import lightgbm as lgb
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')

class LightGBM_wrapper:
    def __init__(self, features, target, cat_cols="auto"):
        self.param = {
            'boosting_type': 'gbdt',
            "objective":"rmse",
            'n_estimators': 1400, 'boost_from_average': False,'verbose': -1,'random_state':42,
    
            'max_bin': 82, 'subsample': 0.4507168737623794, 'subsample_freq': 0.6485534887326423,
            'learning_rate': 0.06282022587205358, 'num_leaves': 8, 'feature_fraction': 0.638399152042614,
            'bagging_freq': 1, 'min_child_samples': 37, 'lambda_l1': 0.007062503953162337, 'lambda_l2': 0.14272770413312064
        }
        
        self.features = features
        self.target = target
        self.cat_cols = cat_cols 

        self.models=[]
    
    def get_importance(self, importance_type='gain'):
        return np.mean([model.feature_importance(importance_type) for model in self.models], axis=0)

    def set_param(self, param):
        self.param = param

    def save_model(self, name=''):
        with open(f'{name}_lgb_wrapper.pickle', 'wb') as f:
            pickle.dump(self, f)
    
    def predict(self, test_df):
        preds = []
        for m in self.models:
            preds.append(m.predict(test_df[self.features]))
        test_df[f'{self.target}_lgb_predict'] = np.mean(preds, axis=0)
        return test_df

    
    def train(self, train_df):
        """
        Fold のカラム名は　基本Fold
        """        
        Folds = set(train_df["Fold"])
        valid = []
        
        for fold in Folds:
            train_data = lgb.Dataset(train_df.loc[train_df["Fold"]!=fold, self.features], train_df.loc[train_df["Fold"]!=fold, self.target])
            valid_df = train_df[train_df["Fold"]==fold]
            valid_data = lgb.Dataset(valid_df[self.features], valid_df[self.target])
            model = lgb.train(self.param,  train_data,  num_boost_round=99999,  valid_sets=[train_data, valid_data], 
                              #feval=lgb_f1_score,
                              verbose_eval=500, early_stopping_rounds=200, categorical_feature=self.cat_cols)
            
            
            valid_df[f'{self.target}_lgb_predict'] = model.predict(valid_df[self.features])
            valid.append(valid_df)
            self.models.append(model)
            
        oof_df = pd.concat(valid, axis=0).sort_index().reset_index(drop=True)
        return oof_df

class MultiClassLightGBM_wrapper:
    def __init__(self, features, target, num_class, cat_cols="auto"):
        self.param = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass', 'metric': 'multiclass', 'num_class':num_class,
            'n_estimators': 1400, 'boost_from_average': False,'verbose': -1,'random_state':42,
    
            'max_bin': 82, 'subsample': 0.4507168737623794, 'subsample_freq': 0.6485534887326423,
            'learning_rate': 0.06282022587205358, 'num_leaves': 8, 'feature_fraction': 0.638399152042614,
            'bagging_freq': 1, 'min_child_samples': 37, 'lambda_l1': 0.007062503953162337, 'lambda_l2': 0.14272770413312064
        }
        
        self.features = features
        self.target = target
        self.cat_cols = cat_cols 
        self.num_class = num_class

        self.models=[]

        
    def set_param(self, param):
        self.param = param

    def save_model(self, name=''):
        with open(f'{name}_malticlass_lgb_wrapper.pickle', 'wb') as f:
            pickle.dump(self, f)

    def predict(self, test_df):
        for i in range(self.num_class):
            test_df[f'{self.target}{i}_lgb_predict'] = 0
        
        for m in self.models:
            preds = m.predict(test_df[self.features])
            for i in range(self.num_class):
                test_df[f'{self.target}{i}_lgb_predict'] = preds[:,i]/len(self.models)
        return test_df
    
    def train(self, train_df):
        """
        Fold のカラム名は　基本Fold
        """        
            
        Folds = set(train_df["Fold"])
        valid = []
        
        for fold in Folds:
            train_data = lgb.Dataset(train_df.loc[train_df["Fold"]!=fold, self.features], train_df.loc[train_df["Fold"]!=fold, self.target])
            valid_df = train_df[train_df["Fold"]==fold]
            valid_data = lgb.Dataset(valid_df[self.features], valid_df[self.target])
            model = lgb.train(self.param,  train_data,  num_boost_round=3000,  valid_sets=[train_data, valid_data], 
                              
                              #feval=my__metircs,
                              #fobj=my_obj,

                              verbose_eval=500, early_stopping_rounds=200, categorical_feature=self.cat_cols)
            valid_preds = model.predict(valid_df[self.features])
            for i in range(self.num_class):
                valid_df[f'{self.target}{i}_lgb_predict'] = valid_preds[:,i]
                
            valid.append(valid_df)
            self.models.append(model)
            
        oof_df = pd.concat(valid, axis=0).sort_index().reset_index(drop=True)
        return oof_df