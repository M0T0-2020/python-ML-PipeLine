import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn import metrics

import lightgbm as lgb
import xgboost as xgb
import catboost
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.svm import SVR

import optuna

import warnings
warnings.filterwarnings('ignore')

class optuna_lightgbm:
    def __init__(self, model, features, target, cat_cols, base_param):
        BASE_PARAM = {
            'boosting_type': 'gbdt',
            "objective":"rmse",
            #'objective': 'multiclass','metric': 'multiclass', 'num_class':4,"class_weight":"balanced",
            #'metric': 'rmse',  'metric': 'tweedie',
            #'objective': 'tweedie',
            #'tweedie_variance_power': trial.suggest_uniform('tweedie_variance_power', 1.01, 1.8),
            'n_estimators': 10000, 'boost_from_average': False,'verbose': -1,'random_state':2020,}
        self.model = model
        self.features = features
        self.cat_cols = cat_cols
        self.target = target
        self.base_param = base_param if base_param is not None else BASE_PARAM
        
    def prepare_dateset(self, train_df, val_df):
        train_set= self.model.Dataset(train_df[self.features],  train_df[self.target])
        val_set = self.model.Dataset(val_df[self.features],  val_df[self.target])
        return train_set, val_set
    
    def prepare_parameters(self, trial):
        params = {
            'max_bin': trial.suggest_int('max_bin', 50, 300),
            'subsample': trial.suggest_uniform('subsample', 0.4, 0.9),
            'subsample_freq': trial.suggest_uniform('subsample_freq', 0.4, 0.9),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.03, 0.5),
            'num_leaves': trial.suggest_int('num_leaves', 4, 2*5),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            #'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 0.0001, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 0.0001, 10.0),
        }
        params.update(self.base_param)
        return params
    
    def train_model(self, params, train_set, val_set):
        model = self.model.train(params=params, train_set=train_set, valid_sets=[train_set, val_set],
                  num_boost_round=99999, categorical_feature=self.cat_cols, 
                  early_stopping_rounds=200, verbose_eval=200
                 )
        return model

    def predict(self, model, val_df):
        preds = model.predict(val_df[self.features])
        return preds

    def main(self, params, train_df, val_df):
        train_set, val_set = self.prepare_dateset(train_df, val_df)
        model = self.train_model(params, train_set, val_set)
        preds = self.predict(model, val_df)
        return preds

class optuna_xgboost:
    def __init__(self, model, features, target, base_param):
        BASE_PARAM = {
            #'objective':'binary:logistic',
            'objective':'reg:squarederror',
            #"objective":"reg:tweedie", "tweedie_variance_power":1.5,
            #'objective': 'multi:softprob', 'num_class': 31, #'eval_metric': 'mlogloss', 
            #"tree_method":'gbdt', 
            "tree_method":'gpu_hist',
            'random_state':42, "n_jobs":-1,}

        self.model = model
        self.features = features
        self.target = target
        self.base_param = base_param if base_param is not None else BASE_PARAM
        
    def prepare_dateset(self, train_df, val_df):
        train_set = xgb.DMatrix(train_df[self.features].fillna(-999), label=train_df[ self.target], missing=-999)
        val_set = xgb.DMatrix(val_df[self.features].fillna(-999), label=val_df[self.target], missing=-999)
        return train_set, val_set
    
    def prepare_parameters(self, trial):
        params = {
            'learning_rate':trial.suggest_loguniform('learning_rate', 0.03, 0.5),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'max_bin': trial.suggest_int('max_bin', 50, 300),
            'subsample': trial.suggest_uniform('subsample', 0.4, 0.9),
            'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.4, 0.9),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 0.9),
            'gamma': trial.suggest_uniform('gamma', 0, 100),
            'min_child_weight': trial.suggest_uniform('min_child_weight', 0, 100),
            'lambda': trial.suggest_loguniform('lambda', 0.0001, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 0.0001, 10.0),
        }
        params.update(self.base_param)
        return params
    
    def train_model(self, params, train_set, val_set):
        model = self.model.train(
            params, train_set, num_boost_round=99999, 
            evals=[(train_set, 'train'), (val_set, 'val')], 
            verbose_eval=200, early_stopping_rounds=100
            )
        return model

    def predict(self, model, val_df):
        preds = model.predict(val_df[self.features])
        return preds

    def main(self, params, train_df, val_df):
        train_set, val_set = self.prepare_dateset(train_df, val_df)
        model = self.train_model(params, train_set, val_set)
        preds = self.predict(model, val_df)
        return preds

class optuna_gradientboostingregressor:
    def __init__(self, model, features, target, base_param):
        BASE_PARAM = {
            #'objective':'binary:logistic',
            'objective':'reg:squarederror',
            #"objective":"reg:tweedie", "tweedie_variance_power":1.5,
            #'objective': 'multi:softprob', 'num_class': 31, #'eval_metric': 'mlogloss', 
            #"tree_method":'gbdt', 
            "tree_method":'gpu_hist',
            'random_state':42, "n_jobs":-1,}

        self.model = model
        self.features = features
        self.target = target
        self.base_param = base_param if base_param is not None else BASE_PARAM
        
    def prepare_dateset(self, train_df, val_df):
        train_set = xgb.DMatrix(train_df[self.features].fillna(-999), label=train_df[ self.target], missing=-999)
        val_set = xgb.DMatrix(val_df[self.features].fillna(-999), label=val_df[self.target], missing=-999)
        return train_set, val_set
    
    def prepare_parameters(self, trial):
        params = {
            'learning_rate':trial.suggest_loguniform('learning_rate', 0.03, 0.5),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'max_bin': trial.suggest_int('max_bin', 50, 300),
            'subsample': trial.suggest_uniform('subsample', 0.4, 0.9),
            'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.4, 0.9),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 0.9),
            'gamma': trial.suggest_uniform('gamma', 0, 100),
            'min_child_weight': trial.suggest_uniform('min_child_weight', 0, 100),
            'lambda': trial.suggest_loguniform('lambda', 0.0001, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 0.0001, 10.0),
        }
        params.update(self.base_param)
        return params
    
    def train_model(self, params, train_set, val_set):
        model = self.model.train(
            params, train_set, num_boost_round=99999, 
            evals=[(train_set, 'train'), (val_set, 'val')], 
            verbose_eval=200, early_stopping_rounds=100
            )
        return model

    def predict(self, model, val_df):
        preds = model.predict(val_df[self.features])
        return preds

    def main(self, params, train_df, val_df):
        train_set, val_set = self.prepare_dateset(train_df, val_df)
        model = self.train_model(params, train_set, val_set)
        preds = self.predict(model, val_df)
        return preds

class optuna_randomforesst:
    def __init__(self, model, features, target, base_param):
        BASE_PARAM = {
            #'objective':'binary:logistic',
            'objective':'reg:squarederror',
            #"objective":"reg:tweedie", "tweedie_variance_power":1.5,
            #'objective': 'multi:softprob', 'num_class': 31, #'eval_metric': 'mlogloss', 
            #"tree_method":'gbdt', 
            "tree_method":'gpu_hist',
            'random_state':42, "n_jobs":-1,}

        self.model = model
        self.features = features
        self.target = target
        self.base_param = base_param if base_param is not None else BASE_PARAM
        
    def prepare_dateset(self, train_df, val_df):
        train_set = xgb.DMatrix(train_df[self.features].fillna(-999), label=train_df[ self.target], missing=-999)
        val_set = xgb.DMatrix(val_df[self.features].fillna(-999), label=val_df[self.target], missing=-999)
        return train_set, val_set
    
    def prepare_parameters(self, trial):
        params = {
            'learning_rate':trial.suggest_loguniform('learning_rate', 0.03, 0.5),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'max_bin': trial.suggest_int('max_bin', 50, 300),
            'subsample': trial.suggest_uniform('subsample', 0.4, 0.9),
            'colsample_bynode': trial.suggest_uniform('colsample_bynode', 0.4, 0.9),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 0.9),
            'gamma': trial.suggest_uniform('gamma', 0, 100),
            'min_child_weight': trial.suggest_uniform('min_child_weight', 0, 100),
            'lambda': trial.suggest_loguniform('lambda', 0.0001, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 0.0001, 10.0),
        }
        params.update(self.base_param)
        return params
    
    def train_model(self, params, train_set, val_set):
        model = self.model.train(
            params, train_set, num_boost_round=99999, 
            evals=[(train_set, 'train'), (val_set, 'val')], 
            verbose_eval=200, early_stopping_rounds=100
            )
        return model

    def predict(self, model, val_df):
        preds = model.predict(val_df[self.features])
        return preds

    def main(self, params, train_df, val_df):
        train_set, val_set = self.prepare_dateset(train_df, val_df)
        model = self.train_model(params, train_set, val_set)
        preds = self.predict(model, val_df)
        return preds