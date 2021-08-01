
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb

import sys, os

import pandas as pd
import numpy as np
import feather
import gc
import copy
import warnings
import pickle
from typing import Tuple

warnings.filterwarnings('ignore')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class XGBoost_wrapper:
    def __init__(self, features, target):
        self.param = {
            'objective':'reg:squarederror',

            #"objective":"reg:tweedie", "tweedie_variance_power":1.5,

            #'objective':'binary:logistic',
            
            #'objective': 'multi:softprob', 'num_class': 31, #'eval_metric': 'mlogloss', 
            #"tree_method":'gbdt', 
            "tree_method":'gpu_hist',

            'random_state':42, "n_jobs":-1,
            'learning_rate':0.03,
            'max_depth': 6, 'max_bin': 218, 'subsample': 0.5857575474392711, 
            'colsample_bynode': 0.8649826507212149, 
            'colsample_bytree': 0.9418682988772633, 'gamma': 5.977650814260331, 
            'min_child_weight': 2.2593440056509055, 'reg_alpha': 0.0011878124961710594,
            'reg_lambda': 0.000391443159919518}

        self.features = features
        self.target = target

        self.models = []
        
    def set_param(self, param):
        self.param = param
        
    def save_model(self, name):
        with open(f'{name}_xgb_wrapper.pickle', 'wb') as f:
            pickle.dump(self, f)
    
    def predict(self, test_df):
        test_data = xgb.DMatrix(test_df[self.features].fillna(-999), missing=-999)
        preds = []
        for m in self.models:
            preds.append(m.predict(test_data))
        test_df[f'{self.target}_xgb_predict'] = np.mean(preds, axis=0)
        return test_df

    def train(self, train_df):
        Folds = set(train_df["Fold"])
        valid = []

        for fold in Folds:
            trn_df = train_df[train_df["Fold"]!=fold]
            val_df = train_df[train_df["Fold"]==fold]
            train_data = xgb.DMatrix(trn_df[self.features].fillna(-999), label=trn_df[ self.target], missing=-999)
            valid_data = xgb.DMatrix(val_df[self.features].fillna(-999), label=val_df[self.target], missing=-999)

            model = xgb.train(
                self.param, train_data, num_boost_round=3000, 
                evals=[(train_data, 'train'), (valid_data, 'val')], 
                verbose_eval=200, early_stopping_rounds=100
                )

            val_df[f'{self.target}_xgb_predict'] = model.predict(valid_data)
            valid.append(val_df)
            self.models.append(model)
            gc.collect()        
        oof_df = pd.concat(valid, axis=0).sort_index().reset_index(drop=True)
        return oof_df

class MultiClassXGBoost_wrapper:
    def __init__(self, features, target, num_class):
        self.param = {
            'objective':'binary:logistic',
            
            #'objective': 'multi:softprob', 'num_class': 31, #'eval_metric': 'mlogloss', 
            #"tree_method":'gbdt', 
            "tree_method":'gpu_hist',

            'random_state':42, "n_jobs":-1,
            'learning_rate':0.03,
            'max_depth': 6, 'max_bin': 218, 'subsample': 0.5857575474392711, 
            'colsample_bynode': 0.8649826507212149, 
            'colsample_bytree': 0.9418682988772633, 'gamma': 5.977650814260331, 
            'min_child_weight': 2.2593440056509055, 'reg_alpha': 0.0011878124961710594,
            'reg_lambda': 0.000391443159919518
            }

        self.features = features
        self.target = target
        self.num_class = num_class

        self.models = []
        
    def set_param(self, param):
        self.param = param
        
    def save_model(self, name):
        with open(f'{name}_malticlass_xgb_wrapper.pickle', 'wb') as f:
            pickle.dump(self, f)
    
    def predict(self, test_df):
        for i in range(self.num_class):
            test_df[f'{self.target}{i}_xgb_predict'] = 0
        
        test_data = xgb.DMatrix(test_df[self.features].fillna(-999), missing=-999)
        for m in self.models:
            preds = m.predict(test_data)
            for i in range(self.num_class):
                test_df[f'{self.target}{i}_xgb_predict'] = preds[:,i]/len(self.models)
        return test_df

    def train(self, train_df):
        Folds = set(train_df["Fold"])
        valid = []

        for fold in Folds:
            trn_df = train_df[train_df["Fold"]!=fold]
            val_df = train_df[train_df["Fold"]==fold]
            train_data = xgb.DMatrix(trn_df[self.features].fillna(-999), label=trn_df[ self.target], missing=-999)
            valid_data = xgb.DMatrix(val_df[self.features].fillna(-999), label=val_df[self.target], missing=-999)

            model = xgb.train(
                self.param, train_data, num_boost_round=3000, 
                evals=[(train_data, 'train'), (valid_data, 'val')], 
                verbose_eval=200, early_stopping_rounds=100
                )
            
            preds = model.predict(valid_data)
            for i in range(self.num_class):
                val_df[f'{self.target}{i}_xgb_predict'] = preds[:,i]
            valid.append(val_df)
            self.models.append(model)
            gc.collect()        
        oof_df = pd.concat(valid, axis=0).sort_index().reset_index(drop=True)
        return oof_df
        