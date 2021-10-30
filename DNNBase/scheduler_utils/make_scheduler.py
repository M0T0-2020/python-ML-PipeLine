import math
import torch
from torch import optim
from torch.optim import lr_scheduler
from transformers import (
    get_cosine_schedule_with_warmup, 
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

def make_scheduler(optimizer:torch.optim.Optimizer, scheduler_name:str='linear', num_training_steps:int=None, num_warmup_steps:int=None):
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            
        )

    if scheduler_name == 'step':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
    elif scheduler_name == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps
        )
    elif scheduler_name == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
    else:
        raise Exception('Unknown lr scheduler: {}'.format(scheduler_name))    
    return scheduler  