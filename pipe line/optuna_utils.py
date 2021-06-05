import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn import metrics

import lightgbm as lgb

import optuna

import warnings
warnings.filterwarnings('ignore')

class Optuna_LightGBM:
    """
    optimize_optuna = Optimize_by_Optuna(
        data=train_df, features=feature, target_colname= 'jobflag'
        )
    study = optuna.create_study(direction='minimize')#maximize
    study.optimize(optimize_optuna.objective, n_trials=40)
    print(study.best_value)
    print(study.best_params)
    """

    def __init__(self, df, features, cat_cols, target):
        self.df = df
        self.features = features
        self.cat_cols = cat_cols
        self.target = target
        
    
    def make_score(self, y, preds):
        s =  metrics.mean_squared_error(y, preds)

        return s

    def objective(self, trial):
                        
        PARAMS = {#'boosting_type': 'gbdt',
            
            'boosting_type': 'gbdt',
            
            #'objective': 'multiclass','metric': 'multiclass', 'num_class':4,"class_weight":"balanced",
            #'metric': 'rmse',  'metric': 'tweedie',
            #'objective': 'tweedie',
            #'tweedie_variance_power': trial.suggest_uniform('tweedie_variance_power', 1.01, 1.8),

            'n_estimators': 1400,
            'boost_from_average': False,'verbose': -1,'random_state':2020,

            'max_bin': trial.suggest_int('max_bin', 50, 300),
            'subsample': trial.suggest_uniform('subsample', 0.4, 0.9),
            'subsample_freq': trial.suggest_uniform('subsample_freq', 0.4, 0.9),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.03, 0.5),
            'num_leaves': trial.suggest_int('num_leaves', 4, 2*5),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 0.0001, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 0.0001, 10.0),
        }
        
        score = 0
        n = 3
        for i in range(n):
            k = StratifiedKFold(n_splits=5, random_state=i, shuffle=True)

            for trn, val in k.split(self.df, self.df[self.target]):
                train_df = self.df.iloc[trn,:]
                val_df = self.df.iloc[val,:]
                train_set= lgb.Dataset(train_df[self.features],  train_df[self.target])
                val_set = lgb.Dataset(val_df[self.features],  val_df[self.target])   
                
                self.model = lgb.train(
                    params=PARAMS, 
                    train_set=train_set, valid_sets=[train_set, val_set],
                    num_boost_round=99999, categorical_feature=self.cat_cols,
                    early_stopping_rounds=200, verbose_eval=200
                    )
                    
                preds = self.model.predict(val_df[self.features])
                y = val_df[self.target]
                s = self.make_score(y, preds)
                score+=s/(n*k.n_splits)
        return score

class Optuna_XGBoost:
    """
    optimize_optuna = Optimize_by_Optuna(
        data=train_df, features=feature, target_colname= 'jobflag'
        )
    study = optuna.create_study(direction='minimize')#maximize
    study.optimize(optimize_optuna.objective, n_trials=40)
    print(study.best_value)
    print(study.best_params)
    """

    def __init__(self, df, features, target):
        self.df = df
        self.features = features
        self.target = target
        
    
    def make_score(self, y, preds):
        s_1 = metrics.mean_squared_error(y, preds)

        return s_1

    def objective(self, trial):
        PARAM = {
            'objective':'binary:logistic',
            #'objective':'reg:squarederror',
            #"objective":"reg:tweedie", "tweedie_variance_power":1.5,
            #'objective': 'multi:softprob', 'num_class': 31, #'eval_metric': 'mlogloss', 
            #"tree_method":'gbdt', 
            "tree_method":'gpu_hist',

            'random_state':42, "n_jobs":-1,
            
            'learning_rate':trial.suggest_loguniform('learning_rate', 0.03, 0.5),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'max_bin': trial.suggest_int('max_bin', 50, 300),
            'subsample': trial.suggest_uniform('subsample', 0.4, 0.9),
            'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.4, 0.9),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 0.9),
            'gamma': trial.suggest_uniform('gamma', 0, 100),
            'min_child_weight': trial.suggest_uniform('min_child_weight', 0, 100),,
            'lambda': trial.suggest_loguniform('lambda', 0.0001, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 0.0001, 10.0),
        }
        
        score = 0
        k = StratifiedKFold(n_splits=5)
        n = 3
        for i in range(n):
            k = StratifiedKFold(n_splits=5, random_state=i, shuffle=True)

            for trn, val in k.split(self.df, self.df[self.target]):
                trn_df = self.df.iloc[trn,:]
                val_df = self.df.iloc[val,:]
                train_data = xgb.DMatrix(trn_df[self.features].fillna(-999), label=trn_df[ self.target], missing=-999)
                valid_data = xgb.DMatrix(val_df[self.features].fillna(-999), label=val_df[self.target], missing=-999)

                model = xgb.train(
                    PARAM, train_data, num_boost_round=99999, 
                    evals=[(train_data, 'train'), (valid_data, 'val')], 
                    verbose_eval=200, early_stopping_rounds=100
                    )
                    
                preds = self.model.predict(valid_data)
                y = val_df[self.target]
                s = self.make_score(y, preds)
                score+=s/(n*k.n_splits)
            
        return score
