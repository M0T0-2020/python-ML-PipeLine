import numpy as np
import pickle
from tqdm import tqdm
from sklearn import metrics

import torch
from torch import nn
from torch import optim
from torch import cuda
from torch.optim.swa_utils import AveragedModel, SWALR

from DNNConfig import DNNConfig
from optimizer_utils.make_optimizer import make_optimizer
from scheduler_utils.make_scheduler import make_scheduler

def get_attr(_class, name, except_values) :
    try: return getattr(_class, name)
    except:  return 100

def count_nn_parameters(net):
    params = 0
    for p in net.parameters():
        if p.requires_grad:
            params += p.numel()
    return params

tmp_model = nn.Linear(20,2)

class ModelTrainer:
    def __init__(self, config:DNNConfig):
        self.config = config
        self.epochs = config.epoch_num
        self.device =  config.device
        
        self.model = tmp_model
        #self.criterion = CustomLoss()

        self.criterion = nn.MSELoss()

        optimizer_kwargs = {'lr':config.lr, 'weight_decay':config.weight_decay}
        self.sam = config.issam
        self.optimizer = make_optimizer(self.model, optimizer_kwargs, optimizer_name=config.optimizer_name, sam=config.issam)
        self.scheduler_name = config.scheduler_name
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=config.T_max)

        self.isswa = config.getattr('isswa', False)
        self.swa_start = config.getattr('swa_start', 0)

        if config.isswa:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=0.025)

        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
        #                                                      mode=config.mode, factor=config.factor)

        self.loss_log = {'train_loss':[],'train_score':[], 'valid_loss':[], 'valid_score':[]}

    def load_params(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def save_loss_log(self, save_name:str):
        with open(save_name, 'wb') as f:
            pickle.dump(self.loss_log, f)

    def loss_fn(self, y, preds):
        criterion = nn.MSELoss()
        loss = criterion(y, preds)
        return loss

    def valid_fn(self, y, preds):
        score = metrics.f1_score(y, preds, average='macro')
        return score

    def reshape_targets(self, y):
        return y.squeeze(1)

    def train(self, trn_dataloader, epoch):
        self.model.to(self.device)
        self.model.train()
        preds, targets, losses = [], [], []
        with tqdm(total=len(trn_dataloader), unit="batch") as pbar:
            pbar.set_description(f"[train] Epoch {epoch+1}/{self.epochs}")
            for data in trn_dataloader:
                x = data['input']
                y = self.reshape_targets(data['label'])
                output = self.model(x.to(self.device))
                if not self.sam:
                    loss = self.loss_fn(output, y.to(self.device))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                else:
                    loss = self.loss_fn(output, y.to(self.device))
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)
                    # second forward-backward pass, make sure to do a full forward pass
                    self.loss_fn(self.model(x.to(self.device)), y.to(self.device)).backward()  
                    self.optimizer.second_step(zero_grad=True)
                
                if self.scheduler_name!='ReduceLROnPlateau':
                    if self.isswa and self.swa_start<=epoch: # if swa phase do nothing
                        pass
                    else:
                        self.scheduler.step()

                losses.append(loss.item())
                preds += output.detach().cpu().tolist() #torch.argmax(output, dim=1).detach().cpu().tolist()
                targets += data.y.detach().cpu().tolist()
                batch_score = self.valid_fn(np.array(targets), np.array(preds))
                pbar.set_postfix(loss=np.mean(losses), score=batch_score)
                pbar.update(1)
        
        if self.scheduler_name=='ReduceLROnPlateau':
            if self.isswa and self.swa_start<=epoch:# if swa phase, update parameters of swa_model 
                self.swa_model.to(self.device)
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                self.swa_model.to('cpu')
            else:
                self.scheduler.step(batch_score)

        self.loss_log['train_loss'].append(np.mean(losses))
        self.loss_log['train_score'].append(batch_score)
        self.model.to('cpu')
        self.model.eval()

    def eval(self, val_dataloader, epoch):
        self.model.to(self.device)
        self.model.eval()
        preds, targets, losses = [], [], []
        with tqdm(total=len(val_dataloader), unit="batch") as pbar:
            pbar.set_description(f"[eval]  Epoch {epoch+1}/{self.epochs}")
            with torch.no_grad():
                for data in val_dataloader:
                    x = data['input']
                    y = self.reshape_targets(data['label'])
                    output = self.model(x.to(self.device)).cpu()
                    target = y.cpu()
                    losses.append(self.loss_fn(target, output))
                    preds += output.tolist()#torch.argmax(output, dim=1).detach().cpu().tolist()
                    targets += target.tolist()
                    batch_score = self.valid_fn(np.array(targets), np.array(preds))
                    pbar.set_postfix(loss=np.mean(losses), score=batch_score)
                    pbar.update(1)
        self.loss_log['valid_loss'].append(np.mean(losses))
        self.loss_log['valid_score'].append(batch_score)
        self.model.to('cpu')
    
    def inference_swa(self, train_loader, data_loader):
        self.swa_model.to(self.device)
        self.swa_model.eval()
        torch.optim.swa_utils.update_bn(train_loader, self.swa_model)
        preds = []
        with tqdm(total=len(data_loader), unit="batch") as pbar:
            pbar.set_description(f"[inference]")
            with torch.no_grad():
                for data in data_loader:
                    x = data['input']
                    output = self.swa_model(x.to(self.device))
                    preds.append(output.cpu().numpy())
                    #hidden_state.append(hidden.cpu().numpy())
                    pbar.update(1)
        return np.vstack(preds)

    def inference(self, data_loader):
        self.model.to(self.device)
        self.model.eval()
        preds, hidden_state = [], []
        with tqdm(total=len(data_loader), unit="batch") as pbar:
            pbar.set_description(f"[inference]")
            with torch.no_grad():
                for data in data_loader:
                    x = data['input']
                    output = self.model(x.to(self.device))
                    preds.append(output.cpu().numpy())
                    #hidden_state.append(hidden.cpu().numpy())
                    pbar.update(1)
        return np.vstack(preds)#, np.vstack(hidden_state)