import math
import torch
from torch import optim
from torch.optim import lr_scheduler
from transformers import (
    get_cosine_schedule_with_warmup, 
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

def make_scheduler(optimizer, decay_name='linear', t_max=None, warmup_steps=None):
    if decay_name == 'step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
    elif decay_name == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max
        )
    elif decay_name == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_max
        )
    elif decay_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=t_max
        )
    else:
        raise Exception('Unknown lr scheduler: {}'.format(decay_name))    
    return scheduler    
