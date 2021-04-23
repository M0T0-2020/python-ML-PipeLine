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
    def __init__(self, train_X, train_y, PARAMS):
        self.train_X = train_X
        self.train_y = train_y
        self.PARAMS = PARAMS

    def make_null_importance_df(self):
        k = StratifiedKFold(n_splits=5)
        null_importance=pd.DataFrame()
        null_importance['col'] = self.train_X.columns.tolist()
        try:
            for i in range(50):
                tmp_null_importance=[]
                _train_y = self.train_y.sample(frac=1).reset_index(drop=True)
                print(f"""
                
                Train Null Importance   {i+1}
                
                """ )
                for trn, val in k.split(self.train_X, self.train_y):
                    trn_X, val_X = self.train_X.iloc[trn,:], self.train_X.iloc[val,:]
                    trn_y, val_y = _train_y.iloc[trn], _train_y.iloc[val]
                    train_set = lgb.Dataset(trn_X, trn_y)
                    val_set = lgb.Dataset(val_X, val_y)

                    model = lgb.train(params=self.PARAMS,
                                      train_set=train_set, 
                                      valid_sets=[train_set, val_set],
                                      categorical_feature=self.cat_cols,
                                    num_boost_round=3000, early_stopping_rounds=200, verbose_eval=500)
                    
                    preds = model.predict(val_X)
                    tmp_null_importance.append(model.feature_importance('gain'))
                null_importance[f'null_importance_{i+1}'] = np.mean(tmp_null_importance, axis=0)
            return null_importance
        except:
            return null_importance

    def calu_importance(self, importance_df, null_importance_df):
        importance_df = pd.merge(
            importance_df, null_importance_df, on='col'
            )
        null_importance_col = [col for col in importance_df.columns if 'null' in col]
        null_importance=pd.DataFrame()
        for idx, row in importance_df.iterrows():
            acc_v = 1e-10+row['true_importance']
            null_v = 1+np.percentile(row[null_importance_col], 75)
            null_importance[row['col']] = [np.log(acc_v/null_v)]
        null_importance = null_importance.T
        return null_importance

    def all_flow(self, cat_cols):
        score=[]
        importance=[]
        self.cat_cols = cat_cols

        importance_df=pd.DataFrame()
        importance_df['col'] = self.train_X.columns
        print("""
        
        Train True Importance
        
        """ )
        
        k = StratifiedKFold(n_splits=5)

        for trn, val in k.split(self.train_X, self.train_y):
            trn_X, val_X = self.train_X.iloc[trn,:], self.train_X.iloc[val,:]
            trn_y, val_y = self.train_y.iloc[trn], self.train_y.iloc[val]
            train_set = lgb.Dataset(trn_X, trn_y)
            val_set = lgb.Dataset(val_X, val_y)
            
            model = lgb.train(params=self.PARAMS, train_set=train_set, valid_sets=[train_set, val_set],
                            categorical_feature=self.cat_cols,
                            num_boost_round=3000, early_stopping_rounds=200, verbose_eval=500)
            #preds = model.predict(val_X)
            
            importance.append(model.feature_importance('gain'))
        importance_df['true_importance'] = np.mean(importance, axis=0)
        
        print("""
        
        Train Null Importance
        
        """ )
        try:
            null_importance_df = self.make_null_importance_df()
        except:
            pass
        print("""
        
        Calulate null_null_importance
        
        """ )
        null_importance = self.calu_importance(importance_df, null_importance_df)
        null_importance = null_importance.reset_index()
        null_importance.columns = ['col', 'score']
        return null_importance