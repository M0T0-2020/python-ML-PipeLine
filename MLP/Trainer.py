import numpy as np

import torch
from torch import nn
from torch import optim
from torch import cuda
from torch.optim.swa_utils import AveragedModel, SWALR
from DNN.optimizer_utils import make_optimizer
from deep_mlp import MLP
from DNN.scheduler_utils import make_scheduler

def count_nn_parameters(net):
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    return params

class Model_Trainer_With_SWA:
    def __init__(self, config):
        self.config = config
        self.device =  'cuda' if cuda.is_available() else 'cpu'
        
        self.model = MLP(config)
        self.swa_model = AveragedModel(self.model)

        self.optimizer = make_optimizer(self.model, optimizer_name=self.config.optimizer, sam=self.config.sam)
        self.scheduler = make_scheduler(self.optimizer, decay_name=self.config.scheduler,
                                        num_training_steps=self.config.num_training_steps,
                                        num_warmup_steps=self.config.num_warmup_steps)
        self.swa_start = self.config.swa_start
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.config.swa_lr)
        self.epoch_num = 0
        self.criterion = self.config.criterion

    def load_params(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train_batch(self, batch_data):
        self.optimizer.zero_grad()
        p = self.model(batch_data['input'].to(self.device))   
        loss = self.criterion(p.squeeze(1), batch_data['label'].to(self.device))
        loss.backward()
        self.optimizer.step()
        if self.epoch_num<self.swa_start:
            self.scheduler.step()
        p = p.squeeze(1).detach().cpu().tolist()
        return p, loss.item()

    def train(self, trn_dataloader):
        self.model.train()
        avg_loss=0
        all_preds = []
        all_labels = []
        for batch_data in trn_dataloader:
            p, loss = self.train_batch(batch_data)
            all_preds+=p
            all_labels+=data['label'].cpu().tolist()
            avg_loss += loss/len(trn_dataloader)
        self.epoch_num+=1
        return avg_loss

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