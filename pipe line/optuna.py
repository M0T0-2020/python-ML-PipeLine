import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn import metrics

import lightgbm as lgb

import optuna

import warnings
warnings.filterwarnings('ignore')

class Optimize_by_Optuna:
    """
    optimize_optuna = Optimize_by_Optuna(
        data=train_df, features=feature, target_colname= 'jobflag'
        )
    study = optuna.create_study(direction='minimize')#maximize
    study.optimize(optimize_optuna.objective, n_trials=40)
    print(study.best_value)
    print(study.best_params)
    """

    def __init__(self, data, features, target_colname):
        self.data = data
        self.features = features
        self.target = target_colname
        
    
    def make_score(self, y, preds):
        s_1=1 - metrics.mean_squared_error(y, preds)

        return s_1

    def objective(self, trial):
                        
        PARAMS = {#'boosting_type': 'gbdt',
            
            'boosting_type': 'gbdt',
            
            #'objective': 'multiclass','metric': 'multiclass', 'num_class':4,
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
        k = StratifiedKFold(n_splits=5)
        
        for trn, val in k.split(self.data, self.data[self.target]):
            train_df = self.data.iloc[trn,:]
            val_df = self.data.iloc[val,:]
            train_set= lgb.Dataset(train_df[self.features],  train_df[self.target])
            val_set = lgb.Dataset(val_df[self.features],  val_df[self.target])   
            
            self.model = lgb.train(
                train_set=train_set, valid_sets=[train_set, val_set], params=PARAMS, num_boost_round=3000, 
                early_stopping_rounds=200, verbose_eval=500
                )
                
            preds = self.model.predict(val_df[self.features])
            preds = np.round(preds)
            y = val_df[self.target]
            s = self.make_score(y, preds)
            score+=s/5
            
        return score
