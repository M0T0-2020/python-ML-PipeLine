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

class Config:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 42
        self.seed_everything()
        self.epoch_num = 20
        self.batch_size = 64
        self.lr = 1e-3
        self.weight_decay = 1e-9
        self.n_folds = 5
        self.hidden_size = 1024
        self.vectorizer_type = "CountVectorizer"
        #self.vectorizer_type = "TfidfVectorizer"

        self.param_lgb = {
            'boosting_type': 'gbdt',
            "objective":"rmse",
            "metrics":"rmse",
            'n_estimators': 1400, 'boost_from_average': False,'verbose': -1,'random_state':2020,
        
            'max_bin': 82, 'subsample': 0.4507168737623794, 'subsample_freq': 0.6485534887326423,
            'learning_rate': 0.06282022587205358, 'num_leaves': 8, 'feature_fraction': 0.638399152042614,
            'bagging_freq': 1, 'min_child_samples': 37, 'lambda_l1': 0.007062503953162337, 'lambda_l2': 0.14272770413312064
            }
    def seed_everything(self):
        seed_everything(self.seed)

    def set_attribute(self, name, value):
        setattr(self, name, value)

    def show_attribute(self):
        atts = {}

        for a in dir(self):
            if a[:2]=='__' and a[-2:]=='__':
                pass
            else:
                atts[a] = getattr(self, a)
        return atts