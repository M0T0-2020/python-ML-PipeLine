import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from glob import glob

from collections import OrderedDict
import math, scipy
import time, gc

import cv2
import PIL

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from torch.optim.optimizer import Optimizer, required
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch import Tensor



class ModelTrain():
    def __init__(self, model, trn, val, criterion, optimizer, sheduler, batch_size, test_batch_size, num_epochs, agg=None, score=None):
        self.model = model
        self.trn = trn
        self.val = val
        self.criterion = criterion
        self.optimizer = optimizer
        self.sheduler = sheduler
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_epochs = num_epochs
        self.do_agg = False if agg is None else True
        self.do_score = False if score is None else True
        self.agg = agg
        self.score = score 

    def train_model(self):
        
        trn_loss = []
        val_loss = []

        trn_score = []
        val_score = []
        
        train_loader = torch.utils.data.DataLoader(self.trn, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.val, batch_size=self.test_batch_size, shuffle=False)
        
        for epoch in range(self.num_epochs):
            # change model to be train_mode 
            self.model.train()
            
            trn_tmp_loss = 0.
            trn_tmp_score = 0
            
            #avg_val_loss = 0.
            for idx, (inputs,labels) in tqdm(enumerate(train_loader),total=len(train_loader)):
                inputs = inputs.to(device).unsqueeze(1).float()
                labels = labels.to(device)
                if self.do_agg:
                    labels = self.agg(labels)
                
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                del inputs
                gc.collect()
                
                loss = self.criterion(outputs,labels)
                
                trn_tmp_loss += float(loss.item())/len(train_loader)
                if self.do_score:
                    trn_score= self.score(outputs.argmax(1), labels.argmax(1))
                    trn_tmp_score+=trn_score/len(train_loader)
                
                loss.backward()
                self.optimizer.step()
                self.sheduler.step()
                del loss, labels, outputs
                gc.collect()
            
            trn_loss += [trn_tmp_loss]
            if self.do_score:
                trn_score+=[trn_tmp_score]
            
            val_tmp_loss = 0.
            val_tmp_score = 0
            model.eval()
            
            for idx, (inputs,labels) in tqdm(enumerate(test_loader),total=len(test_loader)):
                #print('test')
                
                inputs = inputs.to(device).unsqueeze(1).float()
                labels = labels.to(device)

                outputs = self.model(inputs)
                del inputs
                
                gc.collect()
                
                loss = criterion(outputs,labels)
                
                val_tmp_loss += float(loss.item())/len(test_loader)

                if self.do_score:
                    val_score = self.score(outputs.argmax(1), labels.argmax(1))
                    val_tmp_score+=val_score/len(test_loader)
                
                del loss, labels, outputs
                gc.collect()
                
            val_loss += [val_tmp_loss]
            if self.do_score:
                val_score+=[val_tmp_score]
                print(f"""
                train {epoch+1}
                loss {trn_tmp_loss:.4}   score {trn_tmp_score:.4}
                
                test {epoch+1}
                loss {val_tmp_loss:.4}   score {val_tmp_score:.4}
                """)
            else:
                print(f"""
                train {epoch+1}
                loss {trn_tmp_loss:.4}   loss {val_tmp_loss:.4}
                """)
                
        df = pd.DataFrame()
        df['trn_loss'] = trn_loss
        df['val_loss'] = val_loss

        if self.do_score:
            df['trn_score'] = trn_score
            df['val_score'] = val_score

        return model, df