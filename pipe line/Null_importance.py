import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn import metrics

import lightgbm as lgb

import optuna

import warnings
warnings.filterwarnings('ignore')

class Null_Importance:
    def __init__(self, df, features, target, cat_cols, PARAMS, N=30, n_splits=5):
        self.df = df.reset_index(drop=True)
        self.features = features
        self.target = target
        self.cat_cols = cat_cols
        self.PARAMS = PARAMS
        self.N = N
        self.n_splits = n_splits

    def make_null_importance_df(self):
        features = self.features
        k = StratifiedKFold(n_splits=self.n_splits)
        self.null_importance=pd.DataFrame()
        self.null_importance['col'] = features
        train_y = self.df[self.target]
        for i in range(self.N):
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
        importance_df=pd.DataFrame()
        features = self.features
        importance_df['col'] = features
        print("""
        
        Train True Importance
        
        """ )
        
        k = StratifiedKFold(n_splits=self.n_splits)
        importance=[]

        for trn, val in k.split(self.df, self.df[self.target]):
            trn_df, val_df = self.df.loc[trn,:], self.df.loc[val,:]
            
            train_set = lgb.Dataset(trn_df[features], trn_df[self.target])
            val_set = lgb.Dataset(val_df[features], val_df[self.target])
            
            model = lgb.train(params=self.PARAMS, train_set=train_set, valid_sets=[train_set, val_set],
                            categorical_feature=self.cat_cols,
                            num_boost_round=999999, early_stopping_rounds=200, verbose_eval=500)
            #preds = model.predict(val_X)
            
            importance.append(model.feature_importance('gain'))
        importance_df['true_importance'] = np.mean(importance, axis=0)
        self.true_importance_df = importance_df.copy()
        print("""
        
        Train Null Importance
        
        """ )
        self.make_null_importance_df()
        
        print("""
        
        Calulate null_null_importance
        
        """ )
        self.calu_importance()
        self.null_importance_df = self.null_importance_df.reset_index()
        self.null_importance_df.columns = ['col', 'score']
        return self.null_importance_df