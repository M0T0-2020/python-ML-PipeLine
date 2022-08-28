import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from transformers import AdamW

from .utils import *
from .BackBoneType import BackBoneType
from .loss import DINOLoss
from .models import get_student_teacher

def train_one_batch(input, student, teacher, dino_loss, optimizer,
                epoch, clip_grad:Optional[float], freeze_last_layer:Optional[int]):
    teacher_output = teacher(input[:2])  # only the 2 global views pass through the teacher
    student_output = student(input)
    loss = dino_loss(student_output, teacher_output, epoch)
    # student update
    optimizer.zero_grad()
    param_norms = None
    loss.backward()
    if clip_grad is not None:
        param_norms = utils.clip_gradients(student, clip_grad)
    cancel_gradients_last_layer(epoch, student, freeze_last_layer)
    optimizer.step()
    return loss.item()

def ema_update_for_teacher(student, teacher, m):
    """
    Parameters
    m:int momentum parameter
    """
    with torch.no_grad():  
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

def train_one_epoch(student, teacher, dino_loss, data_loader,
                    optimizer, momentum_schedule, epoch, clip_grad, freeze_last_layer,):
    avg_loss = 0
    for it, (images, _) in tqdm(enumerate(data_loader)):
        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        loss = train_one_batch(
            input, student, teacher, dino_loss, optimizer,
            epoch, clip_grad, freeze_last_layer)
        if not math.isfinite(loss):
            print(f"Loss is {loss}, stopping training", force=True)
            sys.exit(1)
        avg_loss += loss/len(data_loader)

        # EMA update for the teacher
        ema_update_for_teacher(student=student, teacher=teacher, m=momentum_schedule[it])
    return avg_loss

class Trainer:
    def __init__(
        self,
        model_type:BackBoneType,
        dl,
        device,
        out_dim,
        pretrained,
        use_bn,
        norm_last_layer,
        nlayers,
        hidden_dim,
        bottleneck_dim,
        local_crops_number,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        momentum_teacher,
        lr,
        weight_decay,
        weight_decay_end,
        nepochs,
        clip_grad,
        freeze_last_layer) -> None:
        self.epochs = nepochs
        self.device = device
        self.student, self.teacher = get_student_teacher(backbone_type=model_type, out_dim=out_dim,
                                                            pretrained=pretrained, use_bn=use_bn, norm_last_layer=norm_last_layer,
                                                            nlayers=nlayers, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)
        self.student, self.teacher = self.student.to(device), self.teacher.to(device)
        self.dataloader = dl
        self.dino_loss = DINOLoss(
            out_dim=out_dim,
            ncrops=local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number   because defualt num_crop is 8,  this number is 10 
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp = teacher_temp, 
            warmup_teacher_temp_epochs = warmup_teacher_temp_epochs,
            nepochs = nepochs,
        ).to(device)
        self.optimizer = AdamW(self.student.parameters(), lr=lr)
        self.wd_schedule, self.momentum_schedule = get_schedulers(weight_decay=weight_decay, weight_decay_end=weight_decay_end,
                                                                    epochs=nepochs, momentum_teacher=momentum_teacher, data_loader=self.dataloader)

        self.clip_grad = clip_grad
        self.freeze_last_layer = freeze_last_layer
    
    def train(self):
        for epoch in self.epochs:
            st = time.time()
            avg_loss = train_one_epoch(self.student, self.teacher, self.dino_loss, self.dataloader, self.optimizer, self.momentum_schedule, epoch, self.clip_grad, self.freeze_last_layer,)
            _t = time.time() - st
            print(f"EPOCH: {epoch+1} LOSS: {avg_loss:.4f} TIME: {_t:.3f}")