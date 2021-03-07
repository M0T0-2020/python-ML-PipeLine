import os, sys
import gc
import numpy as np
import pandas as pd
import time, random, math

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
from sklearn import metrics

import lightgbm as lgb
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True


class LightGBM_wrapper:
    def __init__(self, features, target, cat_cols="auto"):
        self.param = {
            'boosting_type': 'gbdt',
            "objective":"xentropy",
            'n_estimators': 1400, 'boost_from_average': False,'verbose': -1,'random_state':42,
    
            'max_bin': 82, 'subsample': 0.4507168737623794, 'subsample_freq': 0.6485534887326423,
            'learning_rate': 0.06282022587205358, 'num_leaves': 8, 'feature_fraction': 0.638399152042614,
            'bagging_freq': 1, 'min_child_samples': 37, 'lambda_l1': 0.007062503953162337, 'lambda_l2': 0.14272770413312064
        }
        
        self.features = features
        self.target = target
        self.cat_cols = cat_cols 
        
    def set_param(self, param):
        self.param = param
    
    def train_predict(self, train_df, _test_df):
        """
        Fold のカラム名は　基本Fold
        """        
        models=[]
        test_df = _test_df.sort_values('object_id').reset_index(drop=True).copy()
        test_df[f'{self.target}_lgb_predict']=0
        Folds = set(train_df["Fold"])
        valid = []
        
        for fold in Folds:
            train_data = lgb.Dataset(train_df.loc[train_df["Fold"]!=fold, self.features], train_df.loc[train_df["Fold"]!=fold, self.target])
            valid_df = train_df[train_df["Fold"]==fold]
            valid_data = lgb.Dataset(valid_df[self.features], valid_df[self.target])
            model = lgb.train(self.param,  train_data,  num_boost_round=3000,  valid_sets=[train_data, valid_data], 
                              #feval=lgb_f1_score,
                              verbose_eval=500, early_stopping_rounds=100, categorical_feature=self.cat_cols)
            
            
            valid_df[f'{self.target}_lgb_predict'] = model.predict(valid_df[self.features])
            test_df[f'{self.target}_lgb_predict'] += model.predict(test_df[self.features])/len(Folds)
            valid.append(valid_df)
            models.append(model)
            
        train_df = pd.concat(valid, axis=0).sort_values('object_id').reset_index(drop=True)
        test_df = test_df.sort_values('object_id').reset_index(drop=True)
        self.models = models
        return train_df, test_df
    
def run_cv(train_df, test_df, config):
    scores=[]
    oof_preds = []
    test_preds = []
    models = []
    param = config.param_lgb
    features = config.features_lgb
    cat_features = config.cat_features_lgb
    seeds = config.seeds_lgb
    target_col = 'likes'

    for random_state in seeds:
        param['random_state'] = random_state
        lightgbm_wrapper = LightGBM_wrapper(features, target_col, cat_features)
        lightgbm_wrapper.set_param(param)
        oof_df, test_df = lightgbm_wrapper.train_predict(train_df, test_df)

        oof_preds.append(oof_df.sort_values('object_id')[f'{target_col}_lgb_predict'].values)
        test_preds.append(test_df.sort_values('object_id')[f'{target_col}_lgb_predict'].values)

        score = metrics.mean_squared_error(oof_df[target_col], np.round(oof_df[f'{target_col}_lgb_predict']))
        scores.append(score**0.5)
        print(f'random_state {int(random_state)}   score {score:.5f}')
        models.append(lightgbm_wrapper.models)

    print(f"mean score {np.mean(scores):.5f}")
    oof_preds = np.mean(oof_preds, axis=0)
    test_preds = np.mean(test_preds, axis=0)   
    return oof_preds, test_preds, scores