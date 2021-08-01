import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
from sklearn import metrics

import lightgbm as lgb

import optuna

import warnings
warnings.filterwarnings('ignore')

class Null_Importance:
    def __init__(self, df, features, target, cat_cols, PARAMS=None, N=30, n_splits=5, feature_length='all'):
        """
        df : DataFrame - contain "Fold" columns
        features: list - training features
        target: str - target of task
        cat_cols: list - use fof categorical_feature of lightgbm
        PARAMS : dict - lightgbm parameter
        N : int - number of getting null importance
        n_splits : int - n_splits of cross validation
        feature_length: int or str - number of features to get importance default 'all'
        """
        self.df = df.reset_index(drop=True)
        self.features = features
        self.target = target
        self.cat_cols = cat_cols

        if PARAMS is not None:
            self.PARAMS = PARAMS  
        else:
            self.PARAMS = {
                'boosting_type': 'gbdt',
                "objective":"rmse",
                'n_estimators': 1400, 'boost_from_average': False,'verbose': -1,'random_state':42,
            
                'max_bin': 82, 'subsample': 0.4507168737623794, 'subsample_freq': 0.6485534887326423,
                'learning_rate': 0.06282022587205358, 'num_leaves': 8, 'feature_fraction': 0.638399152042614,
                'bagging_freq': 1, 'min_child_samples': 37, 'lambda_l1': 0.007062503953162337, 'lambda_l2': 0.14272770413312064
                }

        self.N = N
        self.n_splits = n_splits
        self.feature_length = feature_length

    def make_true_importance_df(self, features):
        importance_df=pd.DataFrame()
        importance_df['col'] = features
        print("""
        
        Train True Importance
        
        """ )
        
        importance=[]
        fold_cols = [col for col in self.df.columns if "Fold" in col]
        for fold_col in fold_cols:
            folds = sorted(set(self.df[fold_col]))
            for fold in folds:
                trn_df = self.df.loc[self.df[fold_col]!=fold,:]
                val_df = self.df.loc[self.df[fold_col]==fold,:]
        
                train_set = lgb.Dataset(trn_df[features], trn_df[self.target])
                val_set = lgb.Dataset(val_df[features], val_df[self.target])
                
                model = lgb.train(params=self.PARAMS, train_set=train_set, valid_sets=[train_set, val_set],
                                categorical_feature=self.cat_cols,
                                num_boost_round=999999, early_stopping_rounds=200, verbose_eval=500)
                #preds = model.predict(val_X)
            
                importance.append(model.feature_importance('gain'))
        importance_df['true_importance'] = np.mean(importance, axis=0)
        importance_df.sort_values("true_importance", ascending=False, inplace=True)
        importance_df.reset_index(inplace=True)
        return importance_df

    def make_null_importance_df(self, features):
        self.null_importance=pd.DataFrame()
        self.null_importance['col'] = features
        train_y = self.df[self.target]
        for i in range(self.N):
            k = KFold(n_splits=self.n_splits, random_state=i, shuffle=True)
            tmp_null_importance=[]
            _train_y = train_y.sample(frac=1).reset_index(drop=True)
            print(f"""
            Train Null Importance   {i+1}
            """ )
            for trn, val in k.split(self.df, _train_y):
                trn_df, val_df = self.df.loc[trn,:], self.df.loc[val,:]
                trn_y, val_y = _train_y.iloc[trn], _train_y.iloc[val]
                train_set = lgb.Dataset(trn_df[features], trn_y)
                val_set = lgb.Dataset(val_df[features], val_y)
                model = lgb.train(params=self.PARAMS,
                train_set=train_set, valid_sets=[train_set, val_set], categorical_feature=self.cat_cols,
                num_boost_round=999999, early_stopping_rounds=200, verbose_eval=500)
                tmp_null_importance.append(model.feature_importance('gain'))
            self.null_importance[f'null_importance_{i+1}'] = np.mean(tmp_null_importance, axis=0)

    def calu_importance(self):
        tmp_df = pd.merge(
            self.true_importance_df, self.null_importance, on='col'
            )
        null_importance_col = [col for col in tmp_df.columns if 'null' in col]
        self.null_importance_df=pd.DataFrame()
        for idx, row in tmp_df.iterrows():
            acc_v = 1e-10+row['true_importance']
            null_v = 1+np.percentile(row[null_importance_col], 75)
            self.null_importance_df[row['col']] = [np.log(acc_v/null_v)]
        self.null_importance_df = self.null_importance_df.T

    def get_true_importance(self):
            return self.true_importance_df
    
    def get_null_importance(self):
         return self.null_importance

    def get_null_importance_df(self):
            return self.null_importance_df


    def save_info_csv(self, name):
        s = ["true_importance_df", "null_importance", "null_importance_df"]
        for i, df in enumerate([self.true_importance_df, self.null_importance, self.null_importance_df]):
            df.to_csv(f'{name}_{s[i]}.csv', index=False)

    def main(self):
        features = self.features
        importance_df = self.make_true_importance_df(features)
        if self.feature_length != 'all':
            features = list(importance_df["col"][:self.feature_length])
            importance_df = self.make_true_importance_df(features)
        self.true_importance_df = importance_df.copy()
        
        print("""
        
        Train Null Importance
        
        """ )
        features = list(self.true_importance_df["col"])
        self.make_null_importance_df(self.features)
        
        print("""
        
        Calulate null_null_importance
        
        """ )
        self.calu_importance()
        self.null_importance_df = self.null_importance_df.reset_index()
        self.null_importance_df.columns = ['col', 'score']
        return self.null_importance_df