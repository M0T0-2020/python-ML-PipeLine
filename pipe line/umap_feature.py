import umap
from sklearn.cluster import KMeans

def make_umap_feature(df, cols, name):
    new_features=[]
    reducer = umap.UMAP(n_components=3, random_state=42)
    reduced = reducer.fit_transform(df[cols].fillna(0).values)
    for i in range(3):
        df[f"{name}_{i}"] = reduced[:,i]
        new_features.append(f"{name}_{i}")
    k = KMeans(n_clusters=10, random_state=42)
    df[f"{name}_kmeans_class"] = k.fit_predict(reduced)
    new_features.append(f"{name}_kmeans_class")
    return df, new_features