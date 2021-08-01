import numpy as np
import pandas as pd
import os, sys
import random
import torch

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class MLPConfig:
    def __init__(self, train_loader, in_size=10, out_size=1, hidden_layers=[512]):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_layers = hidden_layers
        self.initializer_range = 1

        self.epoch_num = 10

        self.swa = False
        self.sam = True

        self.train_loader_length = len(train_loader)

        #"RAdam", "LAMB", "Adam", "AdamW"
        self.optimizer='Adam'
        #"cosine", "step", "cosine_warmup"
        self.scheduler = "linear"
        self.num_training_steps = self.train_loader_length*self.epoch_num
        self.num_warmup_steps = self.train_loader_length*2
        self.swa_lr=0.025

        self.criterion = torch.nn.MSELoss()