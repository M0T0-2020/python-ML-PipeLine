import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, confusion_matrix, consensus_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as g_nn


class GCN(nn.Module):
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 node_hidden_channels: int,
                 edge_hidden_channels: int,
                 GATv2Conv_hidden_size: int,
                 num_heads:int,
                 num_classes: int):
        super(GCN, self).__init__()

        self.node_encoder = nn.Linear(num_node_features, node_hidden_channels)
        self.edge_encoder = nn.Linear(num_edge_features, edge_hidden_channels)
        
        # NNConv's nn out dim is in_channels*node_hidden_channels
        self.conv1 = g_nn.NNConv(in_channels=node_hidden_channels,
                            out_channels=node_hidden_channels,
                            nn=nn.Linear(edge_hidden_channels, node_hidden_channels * node_hidden_channels))

        self.gatv2conv = g_nn.GATv2Conv(in_channels=node_hidden_channels,
                            out_channels=GATv2Conv_hidden_size, heads=num_heads)
        gatv2conv_out_dim = GATv2Conv_hidden_size*num_heads

        self.conv2 = g_nn.NNConv(in_channels=gatv2conv_out_dim,
                            out_channels=node_hidden_channels,
                            nn=nn.Linear(edge_hidden_channels, gatv2conv_out_dim*node_hidden_channels))
        self.linear = nn.Linear(node_hidden_channels, num_classes)

    def forward(self, data):
        x = data.x #(num_batch*num_sample_graph, node_feature)
        edge_index = data.edge_index #(2, num_batch*num_sample_graph)
        edge_attr = data.edge_attr  #(num_batch*num_sample_graph, node_edge_features)

        x = self.node_encoder(x) # [num_batch*num_sample_graph, node_hidden_channels]
        edge_attr = self.edge_encoder(edge_attr) # [num_batch*num_sample_graph, edge_hidden_channels]

        x = self.conv1(x, edge_index, edge_attr) # [num_batch*num_sample_graph, node_hidden_channels]
        x = F.relu(x)
        
        x = self.gatv2conv(x, edge_index) # [num_batch*num_sample_graph, gatv2conv_out_dim]
        x = F.relu(x)
        
        x = self.conv2(x, edge_index, edge_attr) # [num_batch*num_sample_graph, node_hidden_channels]
        x = F.relu(x)
        
        x = self.linear(x) # [num_nodes, num_classes]

        return x
    
    def get_hidden_state(self, data):
        x = data.x #(num_batch*num_sample_graph, node_feature)
        edge_index = data.edge_index #(2, num_batch*num_sample_graph)
        edge_attr = data.edge_attr  #(num_batch*num_sample_graph, node_edge_features)

        x = self.node_encoder(x) # [num_batch*num_sample_graph, node_hidden_channels]
        edge_attr = self.edge_encoder(edge_attr) # [num_batch*num_sample_graph, edge_hidden_channels]

        x = self.conv1(x, edge_index, edge_attr) # [num_batch*num_sample_graph, node_hidden_channels]
        x = F.relu(x)
        
        x = self.gatv2conv(x, edge_index) # [num_batch*num_sample_graph, gatv2conv_out_dim]
        x = F.relu(x)
        
        hidden = self.conv2(x, edge_index, edge_attr) # [num_batch*num_sample_graph, node_hidden_channels]
        x = F.relu(hidden)
        
        x = self.linear(x) # [num_nodes, num_classes]
        return x, hidden