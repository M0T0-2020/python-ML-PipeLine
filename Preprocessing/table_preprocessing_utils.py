import pandas as pd
import numpy as np
import os, sys
from tqdm import tqdm
import gc, time, random, math
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from joblib import Parallel, delayed

import umap
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

from warnings import filterwarnings
filterwarnings('ignore')

class BaseTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self

class OneHotEncoder(BaseTransformer):
    """
    one_hot_enc = OneHotEncoder(to_onehot_cols)
    one_hot_enc.fit(df)
    df = one_hot_enc.transform(df)
    """
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X):
        self.unique = {}
        for col in self.cols:
            self.unique[col] = list(set(X[col]))
        return self

    def _get_column_names(self):
        col_names = []
        for col in self.cols:
            for cat in self.unique[col]:
                col_names.append(f"{col}_{cat}")
        return col_names

    def transform(self, X):
        for col in self.cols:
            series = X[col]
            onehot = pd.get_dummies(series).reindex(columns=self.unique).fillna(0).astype(int)
            onehot.columns = [f"{col}_{cat}" for cat in onehot.columns]
            X = pd.concat([X, onehot], axis=1)
        return X

class CountEncoder(BaseTransformer):
    def __init__(self, cols):
        self.cols = cols

    def _get_column_names(self):
        col_names = []
        for col in self.cols:
            col_names.append(f"{col}_countenc")
        return col_names

    def fit(self, X):
        self.counts = {}
        for col in self.cols:
            self.counts[col] = X[col].value_counts()
        return self
    
    def transform(self, X):
        df = pd.DataFrame()
        for col in self.cols:
            df[f"{col}_countenc"] = X[col].map(self.counts[col]).fillna(0)
        return df
    
class KNNFeatureExtractor(BaseTransformer):
    def __init__(self, N_CLASSES, n_neighbors=5):
        self.knn = KNeighborsClassifier(n_neighbors + 1)
        self.N_CLASSES = N_CLASSES
        self.is_train_data = True
    
    def set_is_train_data(self, is_train_data):
        self.is_train_data = is_train_data
        
    def fit(self, X, y):
        """
        X : feature (note: not assume null value.)
        y : target
        """
        self.knn.fit(X, y)
        self.y = y if isinstance(y, np.ndarray) else np.array(y)
        return self
        
    def _get_column_names(self):
        cols = []
        score_columns = [f"knn_score_class{c}" for c in range(self.N_CLASSES)]
        cols+=score_columns
        cols.append("max_knn_scores")
        cols.append("sum_knn_scores")
        cols += [f"sub_max_knn_scores_{col}" for col in score_columns]
        for i, col1 in enumerate(score_columns):
            for j, col2 in enumerate(score_columns[i+1:], i+1):
                cols.append(f"sub_{col1}_{col2}")
        return cols

    def transform(self, X):
        distances, indexes = self.knn.kneighbors(X)
        distances = distances[:, 1:] if self.is_train_data else distances[:, :-1]
        indexes = indexes[:, 1:] if self.is_train_data else indexes[:, :-1]
        labels = self.y[indexes]
        score_columns = [f"knn_score_class{c}" for c in range(self.N_CLASSES)]
        df_knn = pd.DataFrame(
            [np.bincount(labels_, distances_, self.N_CLASSES) for labels_, distances_ in zip(labels, 1.0 / distances)],
            columns=score_columns
        )
        df_knn["max_knn_scores"] = df_knn.max(1)
        df_knn["sum_knn_scores"] = df_knn.sum(1)
        for col in score_columns:
            df_knn[f"sub_max_knn_scores_{col}"] = df_knn["max_knn_scores"] - df_knn[col]
        for i, col1 in enumerate(score_columns):
            for j, col2 in enumerate(score_columns[i+1:], i+1):
                df_knn[f"sub_{col1}_{col2}"] = df_knn[col1] - df_knn[col2]
        return df_knn
        
class UMapFeatureExtractor(BaseTransformer):
    def __init__(self, cols, n_components, random_state=42):
        self.cols = cols
        self.n_components = n_components
        self.reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        
    def fit(self, X):
        self.reducer.fit(X[self.cols])
        
    def _get_column_names(self):
        return [f"umap_feature_{i}" for i in range(self.n_components)]
    
    def transform(self, X):
        df = self.reducer.transform(X[self.cols])
        df = pd.DataFrame(df, columns=[f"umap_feature_{i}" for i in range(self.n_components)])
        return df
        
class KMeansFeatureExtractor(BaseTransformer):
    def __init__(self, n_clusters, random_state=42):
        self.n_clusters = n_clusters
        self.k = KMeans(n_clusters=n_clusters, random_state=random_state)
        
    def fit(self, X):
        self.k.fit(X)
        
    def _get_column_names(self):
        return [f"kmeans_feature_{i}" for i in range(self.n_clusters)]
    
    def transform(self, X):
        df = self.k.predict(X)
        df = pd.DataFrame(df, columns=[f"kmeans_feature"])
        return df
    
class StandardScalerFeatureExtractor(BaseTransformer):
    def __init__(self, cols):
        self.cols = cols
        self.std_dscalers = {}

    def _get_column_names(self):
        return [f"standardscaled_{col}" for col in self.cols]

    def fit(self, df):
        for col in self.cols:
            std_dscaler = StandardScaler()
            std_dscaler.fit(df[[col]])
            self.std_dscalers[col] = std_dscaler
        
    def transform(self, df):
        for col, std_dscaler in self.std_dscalers.items():
            df[f"standardscaled_{col}"] = std_dscaler.transform(df[[col]])[:, 0]
        return df

class KBinsDiscreteFeatureExtractor(BaseTransformer):
    """
    'uniform', 'quantile', 'kmeans'
    """
    def __init__(self, cols, n_bins=10, encode="ordinal", strategy="uniform"):
        self.cols = cols
        self.n_bins = n_bins if type(n_bins)!=int else [n_bins for _ in range(len(cols))]
        self.kbin = KBinsDiscretizer(n_bins=self.n_bins, encode=encode, strategy=strategy)
        self.encode=encode
        self.strategy=strategy
    
    def _get_column_names(self):
        return [f"k{bins}bins{self.strategy}category_{self.cols[i]}" for i, bins in zip(range(len(self.cols)), self.n_bins)]
    
    def fit(self, X):
        self.kbin.fit(X[self.cols].fillna(0))
        
    def transform(self, X):
        df = pd.DataFrame()
        value = self.kbin.transform(X[self.cols].fillna(0))
        for i, bins in zip(range(len(self.cols)), self.n_bins):
            df[f"k{bins}bins{self.strategy}category_{self.cols[i]}"] = value[:,i]
        return df
    
class GroupFeatureExtractor(BaseTransformer):
    # 参考: https://signate.jp/competitions/449/discussions/lgbm-baseline-lb06240
    EX_TRANS_METHODS = ["deviation", "zscore"]
    
    def __init__(self, group_key, group_values, agg_methods):
        """
        group_key : list of columns names for grouping
        group_values : list of column names for aggregate
        agg_methods : list (aggregate names)

        note: you must this instance for each group_key.
        """
        self.group_key = group_key
        self.group_values = group_values

        self.ex_trans_methods = [m for m in agg_methods if m in self.EX_TRANS_METHODS]
        self.agg_methods = [m for m in agg_methods if m not in self.ex_trans_methods]
        self.df_agg = None

    def fit(self, df, y=None):
        """
        df : DataFrame contains group_key and group_values columns
        """
        df_train = df[self.group_key+self.group_values]
        self.dfs = {}
        if self.ex_trans_methods:
            tmp_df = df_train.groupby(self.group_key)[self.group_values].agg(['mean', 'std'])
            tmp_df.columns = [f"{col}_{agg}" for col, agg in tmp_df.columns]
            self.ex_trans_methods_col_name = list(tmp_df.columns)
            self.dfs["ex_trans_methods"] = tmp_df
            
        if self.agg_methods:
            tmp_df = df_train.groupby(self.group_key)[self.group_values].agg(self.agg_methods)
            tmp_df.columns = [f"{col}_{agg}" for col, agg in tmp_df.columns]
            self.agg_methods_col_name = list(tmp_df.columns)
            self.dfs["agg_methods"] = tmp_df
            
    def transform(self, df):
        """
        df : DataFrame contains group_key and group_values columns
        """
        df_eval = df[self.group_key+self.group_values]
        new_dfs = []
        if self.ex_trans_methods:
            new_cols = []
            ex_trans_methods_df_features = pd.merge(df_eval[self.group_key+self.group_values], self.dfs["ex_trans_methods"] , on=self.group_key, how="left")
            if "deviation" in self.ex_trans_methods:
                for col in self.group_values:
                    ex_trans_methods_df_features[self._get_column_names(col, "deviation")
                                                ] = ex_trans_methods_df_features[col] - ex_trans_methods_df_features[f"{col}_mean"]
                    new_cols.append(self._get_column_names(col, "deviation"))
            if "zscore" in self.ex_trans_methods:
                for col in self.group_values:
                    ex_trans_methods_df_features[self._get_column_names(col, "zscore")
                                                ] = (ex_trans_methods_df_features[col] - ex_trans_methods_df_features[f"{col}_mean"])/(ex_trans_methods_df_features[f"{col}_std"]+ 1e-8)
                    new_cols.append(self._get_column_names(col, "zscore"))
            new_dfs.append(ex_trans_methods_df_features[new_cols])
            
        if self.agg_methods:
            new_cols = []
            agg_methods_df_features = pd.merge(df_eval[self.group_key+self.group_values], self.dfs["agg_methods"], on=self.group_key, how="left")
            for col in self.group_values:
                for agg in self.agg_methods:
                    agg_methods_df_features[self._get_column_names(col, agg)] = agg_methods_df_features[col]/agg_methods_df_features[f"{col}_{agg}" ]
                    new_cols.append(self._get_column_names(col, agg))
            new_dfs.append(agg_methods_df_features[new_cols])
        df_features = pd.concat(new_dfs, axis=1)
        return df_features

    def _get_column_names(self, col, method):
        n = ''
        for s in self.group_key:
            n+=s
        return f"agg_{method}_{col}_grpby_{n}"
