import numpy as np

import torch
from torch import nn
from torch import optim
from torch import cuda
from torch.optim.swa_utils import AveragedModel, SWALR
from optimizer_utils.make_optimizer import make_optimizer

def count_nn_parameters(net):
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    return params

class ModelTrainer:
    def __init__(self, config):
        self.device =  'cuda' if cuda.is_available() else 'cpu'
        
        self.model = nn.Linear(20,2)
        #self.criterion = CustomLoss()
        #self.criterion = CustomLoss_2(4)
        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=1e-4, weight_decay=0)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=10)
    
    def load_params(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train(self, trn_dataloader):
        self.model.train()
        avg_loss=0
        all_preds = []
        all_labels = []
        for data in trn_dataloader:
            self.optimizer.zero_grad()
            x = data['input'].to(self.device)

            label = data['label'].squeeze(1)
            
            #if len(np.unique(label))!=self.output_size:
             #   continue
            
            label = label.to(self.device)
            x = self.model(x)   
            loss = self.criterion(x, label)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            avg_loss += loss.item()/len(trn_dataloader)
            all_preds+=x.detach().cpu().tolist()
            all_labels+=label.cpu().tolist()
            
        return avg_loss, np.array(all_preds), np.array(all_labels)

    def eval(self, val_dataloader):
        self.model.eval()
        avg_loss=0
        for data in val_dataloader:
            x = data['input'].to(self.device)
            label = data['label'].to(self.device).squeeze(1)
            x = self.model(x)
            loss = self.criterion(x, label)
            avg_loss += loss.item()/len(val_dataloader)
        return avg_loss, x.detach().cpu().numpy()
    
    def predict(self, data_loader):
        self.model.eval()
        preds = []
        for data in data_loader:
            x = data['input'].to(self.device)
            preds+=self.model(x).detach().cpu().tolist()
        return np.array(preds)

class Model_Trainer_With_SWA:
    def __init__(self):
        self.device =  'cuda' if cuda.is_available() else 'cpu'
        
        self.model = nn.Linear(20,2)
        self.swa_model = AveragedModel(self.model)

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=1e-4, weight_decay=0)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=10)
        self.swa_start = 5
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=0.025)
        self.epoch_num = 0
        
        #self.criterion = CustomLoss()
        #self.criterion = CustomLoss_2(4)
        self.criterion = nn.MSELoss()

    def load_params(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train(self, trn_dataloader):
        self.model.train()
        avg_loss=0
        all_preds = []
        all_labels = []
        for data in trn_dataloader:
            self.optimizer.zero_grad()
            x = self.model(data['input'].to(self.device))   
            loss = self.criterion(x.squeeze(1), data['label'].to(self.device))
            loss.backward()
            self.optimizer.step()
            if self.epoch_num<self.swa_start:
                self.scheduler.step()
            avg_loss += loss.item()/len(trn_dataloader)
            all_preds+=x.detach().cpu().tolist()
            all_labels+=label.cpu().tolist()
        self.epoch_num+=1
        return avg_loss, np.array(all_preds), np.array(all_labels)

    def eval(self, val_dataloader):
        self.model.eval()
        avg_loss=0
        for data in val_dataloader:
            x = data['input'].to(self.device)
            label = data['label'].to(self.device).squeeze(1)
            x = self.model(x)
            loss = self.criterion(x, label)
            avg_loss += loss.item()/len(val_dataloader)
        return avg_loss, x.detach().cpu().numpy()
    
    def predict(self, data_loader):
        self.model.eval()
        preds = []
        for data in data_loader:
            x = data['input'].to(self.device)
            preds+=self.model(x).detach().cpu().tolist()
        return np.array(preds)