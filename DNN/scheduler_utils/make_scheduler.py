import math
import torch
from torch import optim
from torch.optim import lr_scheduler
from transformers import (
    get_cosine_schedule_with_warmup, 
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

def make_scheduler(optimizer, decay_name='linear', num_training_steps=None, num_warmup_steps=None):
    if decay_name == 'step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
    elif decay_name == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps
        )
    elif decay_name == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif decay_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
    else:
        raise Exception('Unknown lr scheduler: {}'.format(decay_name))    
    return scheduler  