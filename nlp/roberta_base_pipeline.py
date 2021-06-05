import os, sys
import numpy as np 
import pandas as pd 
import random, math, gc, pickle
       
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import transformers

from make_optimizer import make_optimizer
from make_scheduler import make_scheduler

def convert_examples_to_features(data, tokenizer, max_len, is_test=False):
    data = data.replace('\n', '')
    inputs = tokenizer.encode_plus(
            data,
            # [CLS],[SEP]を入れるか
            add_special_tokens = True, 
            # paddingとtrancation(切り出し)を使って、単語数をそろえる
            max_length = 314, 
            # ブランク箇所に[PAD]を入れる
            padding = "max_length", 
            # 切り出し機能。例えばmax_length10とかにすると、最初の10文字だけにしてくれる機能。入れないと怒られたので、入れておく
            truncation = True, 
            return_tensors = 'pt',
            return_token_type_ids = True
        )
    
    return inputs

class DatasetRoberta(Dataset):
    def __init__(self, texts, target, max_len, param_path="roberta-base", is_test=False):
        self.texts = list(texts)
        self.targets = list(target)
        self.param_path = param_path
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(self.param_path)
        self.is_test = is_test
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        excerpt = self.texts[idx]
        features = convert_examples_to_features(
            excerpt, self.tokenizer, 
            self.max_len, self.is_test
        )
        if not self.is_test:
            label = self.targets[idx]
            features["label"] = torch.tensor(label, dtype=torch.double)
        return features
        
def make_robertaConfig(param_path, epochs, train_loader, num_labels=1):
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    config = transformers.AutoConfig.from_pretrained(param_path)
    config.update({'num_labels':num_labels, 'param_path':param_path, 'epochs':epochs, 'train_loader_length':len(train_loader)})
    return config

class RobertaModel(nn.Module):
    def __init__(self, config,  multisample_dropout=False, output_hidden_states=False):
        super(RobertaModel, self).__init__()
        self.config = config
        self.roberta = transformers.RobertaModel.from_pretrained(config.param_path, output_hidden_states=output_hidden_states)
        
        #self.roberta.load_state_dict(torch.load(f'../input/commonlit-roberta-base-i/model{fold}.bin'))
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        if multisample_dropout:
            self.dropouts = nn.ModuleList([
                nn.Dropout(0.5) for _ in range(5)
            ])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(0.3)])
        self.regressor = nn.Linear(config.hidden_size, 1)
        self._init_weights(self.layer_norm)
        self._init_weights(self.regressor)
 
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
 
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.roberta( input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,)
        sequence_output = outputs[1]
        sequence_output = self.layer_norm(sequence_output)
 
        # multi-sample dropout
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.regressor(dropout(sequence_output))
            else:
                logits += self.regressor(dropout(sequence_output))
        
        logits /= len(self.dropouts)
 
        # calculate loss
        loss = None
        if labels is not None:
            loss_fn = torch.nn.MSELoss()
            #　size -> (batch_size, )
            logits = logits.view(-1).to(labels.dtype)
            loss = torch.sqrt(loss_fn(logits, labels.view(-1)))
        
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

class RobertaTrainer:
    def __init__(self, config, log_interval=1, evaluate_interval=1):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = RobertaModel(config, multisample_dropout=True)

        max_train_steps = config.epochs * config.train_loader_length
        warmup_proportion = 0
        if warmup_proportion != 0:
            warmup_steps = math.ceil((max_train_steps * 2) / 100)
        else:
            warmup_steps = 0

        self.optimizer = make_optimizer(self.model, optimizer_name='AdamW')
        self.scheduler = make_scheduler(self.optimizer, decay_name='cosine_warmup', t_max=max_train_steps, warmup_steps=warmup_steps)    
        self.scalar = torch.cuda.amp.GradScaler() # GPUでの高速化。
        self.log_interval = log_interval
        self.evaluate_interval = evaluate_interval
        self.evaluator = Evaluator(self.model, self.scalar)
        self.best_val_loss = np.inf

    def train(self, train_loader, valid_loader, epoch, fold):
        count = 0
        losses = []
        self.model.to(self.device)
        self.model.train()

        
        for batch_idx, batch_data in enumerate(train_loader):
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=batch_data['input_ids'].to(self.device),
                    attention_mask=batch_data['attention_mask'].to(self.device),
                    token_type_ids=batch_data['token_type_ids'].to(self.device),
                    labels=batch_data['label'].to(self.device),
                    )

            loss, _ = outputs[:2]
            count += batch_data['label'].size(0)
            losses.append(loss.item())
            
            self.optimizer.zero_grad()
            self.scalar.scale(loss).backward()
            self.scalar.step(self.optimizer)
            self.scalar.update()

            self.scheduler.step()

            if batch_idx % self.log_interval == 0:
                _s = str(len(str(len(train_loader.sampler))))
                ret = [
                    ('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count, len(train_loader.sampler), 100 * count / len(train_loader.sampler)),
                    'train_loss: {: >4.5f}'.format(np.mean(losses)),
                ]
                print(', '.join(ret))
            
            if batch_idx % self.evaluate_interval == 0:
                loss = self.evaluator.evaluate(
                    valid_loader, 
                    epoch
                )
                if loss < self.best_val_loss:
                    print("{} epoch, best epoch was updated! valid_loss: {: >4.5f}".format(epoch, loss))
                    self.best_val_loss = loss
                    torch.save(self.model.state_dict(), f"model{fold}.bin")

        self.model.to('cpu')
        return loss

class Evaluator:
    def __init__(self, model, scalar):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.scalar = scalar

    def save(self, result):
        return None

    def load(self):
        return None

    def evaluate(self, data_loader, epoch):
        losses = []
        self.model.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=batch_data['input_ids'].to(self.device),
                        attention_mask=batch_data['attention_mask'].to(self.device),
                        token_type_ids=batch_data['token_type_ids'].to(self.device),
                        labels=batch_data['label'].to(self.device),
                        )
                loss, _ = outputs[:2]
                losses.append(loss.item())

        print('----Validation Results Summary----')
        print('Epoch: [{}] valid_loss: {: >4.5f}'.format(epoch, np.mean(losses)))
        self.model.to('cpu')
        return np.mean(losses)