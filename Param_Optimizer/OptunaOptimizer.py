import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
import pickle 

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn import metrics

import lightgbm as lgb

import optuna

from optuna_utils import *

import warnings
warnings.filterwarnings('ignore')

def _get_model_wrapper(model, features, target, cat_cols, base_param):
    name = model.__name__
    if name=='lightgbm':
        return optuna_lightgbm(model, features, target, cat_cols, base_param)
    if name=='xgboost':
        return optuna_xgboost(model, features, target, base_param)
    if name=='GradientBoostingRegressor':
        return None
    if name=='RandomForestRegressor':
        return None
        
class OptunaOptimizer:
    """
    optimize_optuna = Optimize_by_Optuna(
        data=train_df, features=feature, target_colname= 'jobflag'
        )
    study = optuna.create_study(direction='minimize')#maximize
    study.optimize(optimize_optuna.objective, n_trials=40)
    print(study.best_value)
    print(study.best_params)
    """

    def __init__(self, df, model, features, target, base_param=None, criterion=None, cat_cols=None, num_iter=1, kfold=StratifiedKFold, n_splits=5):
        self.df = df
        self.model_name = model.__name__
        self.model_wrapper = _get_model_wrapper(model, features, target, cat_cols, base_param)

        self.features = features
        self.criterion =  self.make_score if criterion is None else criterion
        self.target = target
        self.base_param = base_param
        self.cat_cols = cat_cols
        self.num_iter = num_iter
        self.kfold = kfold
        self.n_splits = n_splits

    def make_score(self, y, preds):
        s =  self.criterion(y, preds)
        return s

    def objective(self, trial):
        score = 0
        params = self.model_wrapper.prepare_parameters(trial)

        for i in range(self.num_iter):
            k = self.kfold(n_splits=self.n_splits, random_state=i, shuffle=True)
            for trn, val in k.split(self.df, self.df[self.target]):
                train_df = self.df.iloc[trn,:]
                val_df = self.df.iloc[val,:]
                preds = self.model_wrapper.main(params, train_df, val_df)
                y = val_df[self.target]
                s = self.make_score(y, preds)
                score+=s/(self.num_iter*self.n_splits)
        return score

    def main(self, n_trials, direction='minimize'):
        self.study = optuna.create_study(direction=direction)#maximize
        self.study.optimize(self.objective, n_trials=n_trials)
        # save study
        num_trials = len(self.study.get_trials())
        with open(f"{self.model_name}_{num_trials}.pickle", 'wb') as f:
            pickle.dump(self.study, f)