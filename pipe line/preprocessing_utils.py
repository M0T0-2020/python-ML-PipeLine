import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
import umap
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from warnings import filterwarnings
filterwarnings('ignore')


class CountEncoder:
    def fit(self, series):
        self.counts = series.groupby(series).count()
        return self
    
    def transform(self, series):
        return series.map(self.counts).fillna(0)
    
    def fit_transform(self, series):
        return self.fit(series).transform(series)
    
class KNNFeatureExtractor:
        def __init__(self, n_neighbors=5):
            self.knn = KNeighborsClassifier(n_neighbors + 1)

        def fit(self, X, y):
            self.knn.fit(X, y)
            self.y = y if isinstance(y, np.ndarray) else np.array(y)
            return self

        def transform(self, X, is_train_data):
            distances, indexes = self.knn.kneighbors(X)
            distances = distances[:, 1:] if is_train_data else distances[:, :-1]
            indexes = indexes[:, 1:] if is_train_data else indexes[:, :-1]
            labels = self.y[indexes]
            score_columns = [f"knn_score_class{c:02d}" for c in range(N_CLASSES)]
            df_knn = pd.DataFrame(
                [np.bincount(labels_, distances_, N_CLASSES) for labels_, distances_ in zip(labels, 1.0 / distances)],
                columns=score_columns
            )
            df_knn["max_knn_scores"] = df_knn.max(1)
            for col in score_columns:
                df_knn[f"sub_max_knn_scores_{col}"] = df_knn["max_knn_scores"] - df_knn[col]
            for i, col1 in enumerate(score_columns):
                for j, col2 in enumerate(score_columns[i+1:], i+1):
                    if {i, j} & {8, 10}:
                        df_knn[f"sub_{col1}_{col2}"] = df_knn[col1] - df_knn[col2]
            df_knn["sum_knn_scores"] = df_knn.sum(1)

            return df_knn
        
class UMapFeatureExtractor:
    def __init__(self, n_components):
        self.n_components = n_components
        self.reducer = umap.UMAP(n_components=n_components, random_state=42)
        
    def fit(self, X):
        self.reducer.fit(X)
        
    def _get_column_names(self):
        return [f"umap_feature_{i}" for i in range(self.n_components)]
    
    def transform(self, X):
        df = self.reducer.transform(X)
        df = pd.DataFrame(df, columns=[f"umap_feature_{i}" for i in range(self.n_components)])
        return df
        
class KMeansFeatureExtractor:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.k = KMeans(n_clusters=n_clusters, random_state=42)
        
    def fit(self, X):
        self.k.fit(X)
        
    def _get_column_names(self):
        return [f"kmeans_feature_{i}" for i in range(self.n_clusters)]
    
    def transform(self, X):
        df = self.k.predict(X)
        df = pd.DataFrame(df, columns=[f"kmeans_feature"])
        return df
    
class StandardScalerFeatureExtractor:
    def __init__(self, cols):
        self.cols = cols
        self.std_dscalers = {}
        
    def fit(self, df):
        for col in self.cols:
            std_dscaler = StandardScaler()
            std_dscaler.fit(df[[col]])
            self.std_dscalers[col] = std_dscaler
        
    def transform(self, df):
        for col, std_dscaler in self.std_dscalers.items():
            df[f"standardscaled_{col}"] = std_dscaler.transform(df[[col]])[:, 0]
        return df
    
class KBinsDiscreteFeatureExtractor:
    def __init__(self, cols, n_bins=10, encode="ordinal", strategy="uniform"):
        self.cols = cols
        self.kbin = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
        
    def fit(self, df):
        self.kbin.fit(df[self.cols].fillna(0))
        
    def transform(self, df):
        value = self.kbin.transform(df[self.cols].fillna(0))
        for i in range(value.shape[-1]):
            df[f"kbinscategory_{self.cols[i]}"] = value[:,i]
        return df
    
class GroupFeatureExtractor:  # 参考: https://signate.jp/competitions/449/discussions/lgbm-baseline-lb06240
    EX_TRANS_METHODS = ["deviation", "zscore"]
    
    def __init__(self, group_key, group_values, agg_methods):
        self.group_key = group_key
        self.group_values = group_values

        self.ex_trans_methods = [m for m in agg_methods if m in self.EX_TRANS_METHODS]
        self.agg_methods = [m for m in agg_methods if m not in self.ex_trans_methods]
        self.df_agg = None

    def fit(self, df_train, y=None):
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
            
    def transform(self, df_eval):
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

    def fit_transform(self, df_train, y=None):
        self.fit(df_train, y=y)
        return self.transform(df_train)  

