import numpy as np
import pandas as pd
import os, sys
import random
import torch
from torch.optim import optimizer

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class GraphNNConfig:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    seed_everything(seed)
    epochs = 20
    batch_size = 64
    lr = 1e-3
    weight_decay = 1e-9
    n_folds = 5

    #model save
    model_save_dir = './'
    model_save_epoch_term = 10
    model_save_epoch_start = 40

    #model param
    num_classes = 10
    edge_hidden_channels = 64
    node_hidden_channels = 64
    GATv2Conv_hidden_size = 128
    num_heads = 1

    # optimizer scheduler
    sam = True
    optimizer_name = 'Adam'
    factor = 0.4

    # features
    node_features_cols = ['col_name']

    edge_feature_cols = ['col_name']
    
    #edge_feature_cols2 = [ 'x_center', 'y_center', 'area',]
    #angle_feature_cols = ['x_center', 'y_center']

    node_features_size = len(node_features_cols)
    edge_feature_size = len(edge_feature_cols)# + len(edge_feature_cols2) + 1