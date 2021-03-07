
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
    def __init__(self, config, features, target):
        self.save_path = config.param_save_path
        self.param = {
            'objective':'binary:logistic',
            #'objective': 'multi:softprob', 'num_class': 31, #'eval_metric': 'mlogloss', 
            #"tree_method":'gbdt', 
            'random_state':42, "n_jobs":-1,
            'learning_rate':0.03,
            'max_depth': 6, 'max_bin': 218, 'subsample': 0.5857575474392711, 
            'colsample_bynode': 0.8649826507212149, 
            'colsample_bytree': 0.9418682988772633, 'gamma': 5.977650814260331, 
            'min_child_weight': 2.2593440056509055, 'reg_alpha': 0.0011878124961710594,
            'reg_lambda': 0.000391443159919518}

        self.features = features
        self.target = target
        
    def set_param(self, param):
        self.param = param
        
    def save_model(self, save_name):
        with open(f'{self.save_path}/{save_name}', 'wb') as f:
            pickle.dump(self.models, f)
    
    def predict(self, test_df):
        test_data = xgb.DMatrix(test_df[self.features].fillna(-999), missing=-999)
        preds = []
        for m in self.models:
            preds.append(m.predict(test_data))
        test_df[f'{self.target}_xgb_predict'] = np.mean(preds, axis=0)
        return test_df

    def train(self, train_df):
        self.models = []
        Folds = set(train_df["Fold"])
        valid = []

        for fold in Folds:
            trn_df = train_df[train_df["Fold"]!=fold]
            valid_df = train_df[train_df["Fold"]==fold]
            train_data = xgb.DMatrix(trn_df[self.features].fillna(-999), label=trn_df[ self.target], missing=-999)
            valid_data = xgb.DMatrix(valid_df[self.features].fillna(-999), label=valid_df[self.target], missing=-999)
            model = xgb.train(self.param, train_data, num_boost_round=3000, evals=[(train_data, 'train'), (valid_data, 'val')], 
                              verbose_eval=200, early_stopping_rounds=100)
            valid_df[f'{self.target}_xgb_predict'] = model.predict(valid_data)
            valid.append(valid_df)
            self.models.append(model)
            gc.collect()        
        train_df = pd.concat(valid, axis=0).sort_values('id').reset_index(drop=True)
        return train_df

def train_xgb(xgb_model, train_df, xgb_param, name):
    target_col='state'
    for i, p in enumerate(xgb_param):
        xgb_model.set_param(p)
        train_df = xgb_model.train(train_df)
        train_df[f'{name}{i}'] = train_df[f'{target_col}_xgb_predict']
        xgb_model.save_model(f'{name}_param{i}_xgb.pkl')
        del xgb_model.models;gc.collect()        
    return train_df

def predict_xgb(config, xgb_model, test_df, xgb_param_path, name):
    target_col='state'
    path = config.param_save_path
    for i, model_path in enumerate(xgb_param_path):
        with open(f"{path}/{model_path}", 'rb') as f:
            models = pickle.load(f)
        xgb_model.models = models
        test_df = xgb_model.predict(test_df)
        test_df[f'{name}{i}'] = test_df[f'{target_col}_xgb_predict']
        del xgb_model.models;gc.collect()        
    return test_df