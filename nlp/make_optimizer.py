import os, sys
import numpy as np 
import pandas as pd 
import random, math, gc, pickle

from LAMB import Lamb
from sam import SAM

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import transformers

def get_optimizer_params(model):
    # differential learning rate and weight decay
    learning_rate = 5e-5
    no_decay = ['bias', 'gamma', 'beta']
    group1=['layer.0.','layer.1.','layer.2.','layer.3.']
    group2=['layer.4.','layer.5.','layer.6.','layer.7.']    
    group3=['layer.8.','layer.9.','layer.10.','layer.11.']
    group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
    optimizer_parameters = [
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.01},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.01, 'lr': learning_rate/2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.01, 'lr': learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.01, 'lr': learning_rate*2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.0},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.0, 'lr': learning_rate/2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.0, 'lr': learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.0, 'lr': learning_rate*2.6},
        {'params': [p for n, p in model.named_parameters() if "roberta" not in n], 'lr':1e-3, "momentum" : 0.99},
    ]
    return optimizer_parameters

def make_optimizer(model, optimizer_name="AdamW", sam=False):
    optimizer_grouped_parameters = get_optimizer_params(model)
    kwargs = {
            'lr':5e-5,
            'weight_decay':0.01,
            # 'betas': (0.9, 0.98),
            # 'eps': 1e-06
    }
    if sam:
        if optimizer_name == "LAMB":
            optimizer = Lamb(optimizer_grouped_parameters, **kwargs)
            return optimizer
        elif optimizer_name == "Adam":
            from torch.optim import Adam
            optimizer = Adam(optimizer_grouped_parameters, **kwargs)
            return optimizer
        elif optimizer_name == "AdamW":
            optimizer = transformers.AdamW(optimizer_grouped_parameters, **kwargs)
            return optimizer
        else:
            raise Exception('Unknown optimizer: {}'.format(optimizer_name))
    else:
        if optimizer_name == "LAMB":
            base_optimizer = Lamb
            optimizer = SAM(optimizer_grouped_parameters, base_optimizer, rho=0.05, **kwargs)
            return optimizer
        elif optimizer_name == "Adam":
            from torch.optim import Adam
            base_optimizer = Adam
            optimizer = SAM(optimizer_grouped_parameters, base_optimizer, rho=0.05, **kwargs)
            return optimizer
        elif optimizer_name == "AdamW":
            from transformers import AdamW
            base_optimizer = AdamW
            optimizer = SAM(optimizer_grouped_parameters, base_optimizer, rho=0.05, **kwargs)
            return optimizer
        else:
            raise Exception('Unknown optimizer: {}'.format(optimizer_name))
