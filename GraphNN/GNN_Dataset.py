import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from GraphNNConfig import GraphNNConfig


class GNN_Dataset:
    def __init__(self, config:GraphNNConfig) -> None:
        self.node_features_cols = config.node_features_cols
        self.edge_feature_cols = config.edge_feature_cols

    def build_nodes(self, sample: pd.DataFrame):
        x = sample[self.node_features_cols].values.astype(np.float32) # [num_nodes, num_node_features]
        return torch.tensor(x, dtype=torch.float)

    def build_edges(self, sample: pd.DataFrame):
        num_nodes = len(sample)
        edge_feature = sample[self.edge_feature_cols].values
        edge_idx, edge_attr = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i!=j:
                    edge_idx.append([i,j])
                    edge_data = np.hstack((edge_feature[i,:] - edge_feature[j,:], edge_feature[i,:] / edge_feature[j,:])).tolist()
                    #edge_data = (edge_feature[i,:] - edge_feature[j,:]).tolist()
                    edge_attr.append(edge_data)
        edge_attr = torch.FloatTensor(edge_attr)  # [num_edges, num_edge_features]
        return torch.tensor(edge_idx, dtype=torch.long).t().contiguous(), edge_attr

    def build_labels(self, sample: pd.DataFrame):
        y = sample.class_id.values
        return torch.tensor(y, dtype=torch.long)

    def build_data_list(self, input_df: pd.DataFrame, istrain=True):
        data_list = []
        bcid_index = []
        for id in tqdm(input_df['id'].unique()):
            sample = input_df.query("id == @id")
            x = self.build_nodes(sample)
            edge_index, edge_attr = self.build_edges(sample)
            if istrain:
                y =self. build_labels(sample)
            else:
                y = 0
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)
            bcid_index.append(id)
        return data_list, bcid_index

"""
train_data_list = build_data_list(tr_df)
valid_data_list = build_data_list(val_df)
test_data_list = build_data_list(test_df, istrain=False)
train_loader = DataLoader(train_data_list, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_data_list, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data_list, batch_size=64, shuffle=False)
"""