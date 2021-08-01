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

def make_robertaConfig(path, cudnn_benchmark=False, num_labels=1, seed=42):
    import numpy as np 
    import math, random, torch
    
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    
    config = transformers.AutoConfig.from_pretrained(path)
    config.update({'seed':seed, 'model_path':path, "num_labels":num_labels,
                   #"optimizer_name":'AdamW',
                   "optimizer_name":'LAMB',
                   "sam":False,
                   'decay_name':'cosine_warmup'})
    return config

def convert_examples_to_features(sequence, tokenizer, max_len, is_test=False):
    sequence = sequence.replace('\n', '')
    inputs = tokenizer.encode_plus(
            sequence,
            # [CLS],[SEP]を入れるか
            add_special_tokens = True, 
            # paddingとtrancation(切り出し)を使って、単語数をそろえる
            max_length = max_len, 
            # ブランク箇所に[PAD]を入れる
            padding = "max_length", 
            # 切り出し機能。例えばmax_length10とかにすると、最初の10文字だけにしてくれる機能。入れないと怒られたので、入れておく
            truncation = True, 
            # output dtype is TensorLong
            return_tensors = 'pt',
            
            #Do it when using two sequences as inputs.
            #return_token_type_ids = True
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
        for key, value in features.items():
            # size: (1, max_len) ->(max_len,)
            features[key] = value.squeeze(0)

        if not self.is_test:
            label = self.targets[idx]
            features["labels"] = torch.tensor(label, dtype=torch.double)
        return features

class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim, num_targets):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim

        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.W(features))

        score = self.V(att)

        attention_weights = torch.softmax(score, dim=1)

        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector

class RobertaModel(nn.Module):
    def __init__(self, param_path, config, output_hidden_states=True):
        super(RobertaModel, self).__init__()
        self.config = config
        self.output_hidden_states = output_hidden_states
        self.roberta = transformers.RobertaModel.from_pretrained(param_path, output_hidden_states=output_hidden_states)
        
        self.head = AttentionHead(config.hidden_size, config.hidden_size, 1)
        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.hidden_size//4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size//4, config.num_labels),
        )
 
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
 
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.roberta( input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,)
        
        if self.output_hidden_states:
            sequence_output = torch.cat([
                outputs["hidden_states"][-4][:, 0].reshape((-1, 1, 768)),
                outputs["hidden_states"][-3][:, 0].reshape((-1, 1, 768)),
                outputs["hidden_states"][-2][:, 0].reshape((-1, 1, 768)),
                outputs["pooler_output"].reshape((-1, 1, 768)),
                outputs["last_hidden_state"] # size (batch, seq_length, 768)
            ], 1)  
            sequence_output = self.head(sequence_output)
            logits = self.regressor(sequence_output)
            del sequence_output; gc.collect()
        else:
            sequence_output = outputs["last_hidden_state"]
            sequence_output = self.head(sequence_output)
            logits = self.regressor(sequence_output)
            del sequence_output; gc.collect()

        return (logits, outputs["pooler_output"])

class RobertaTrainer:
    def __init__(self, param_path, config, log_interval=1, evaluate_interval=1):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = RobertaModel(param_path, config)

        max_train_steps = config.epochs * config.train_loader_length
        warmup_proportion = 0
        if warmup_proportion != 0:
            warmup_steps = math.ceil((max_train_steps * 2) / 100)
        else:
            warmup_steps = 0
        
        self.sam = config.sam
        self.optimizer = make_optimizer(self.model, optimizer_name=config.optimizer_name)
        self.scheduler = make_scheduler(self.optimizer, decay_name=config.decay_name, t_max=max_train_steps, warmup_steps=warmup_steps)    
        self.scalar = torch.cuda.amp.GradScaler() # GPUでの高速化。
        self.log_interval = log_interval
        self.evaluate_interval = evaluate_interval
        self.best_val_loss = np.inf
        self.trn_loss_log = []
        self.val_loss_log = []
    
    def load_param(self, path):
        self.model.load_state_dict(
            torch.load(path)
            )
    def save_param(self, name):
        torch.save(self.model.state_dict(), name)

    def eval(self, valid_loader):
        preds=[]
        labels=[]
        losses = []
        self.model.to(self.device)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(valid_loader):
                with torch.cuda.amp.autocast():
                    logits, _output = self.model(
                        input_ids=batch_data['input_ids'].to(self.device),
                        attention_mask=batch_data['attention_mask'].to(self.device),
                        #token_type_ids=batch_data['token_type_ids'].to(self.device),
                        )
                preds+=logits.detach().view(-1).cpu().tolist()
                labels+=batch_data['labels'].view(-1).cpu().tolist()
        
        preds = torch.FloatTensor(preds)
        labels = torch.FloatTensor(labels)
        loss_fn = torch.nn.MSELoss()
        loss = torch.sqrt(loss_fn(preds, labels))
        self.val_loss_log.append(loss)
        self.model.to('cpu')
        self.model.eval()
        return loss

    def train(self, batch_data, losses, count):
        self.model.to(self.device)
        self.model.train()

        loss_fn = torch.nn.MSELoss()

        if self.sam:
            logits, _output = self.model(
                    input_ids=batch_data['input_ids'].to(self.device),
                    attention_mask=batch_data['attention_mask'].to(self.device),
                    #token_type_ids=batch_data['token_type_ids'].to(self.device),
                    )
            # calculate loss
            #　size -> (batch_size, )
            logits = logits.view(-1)
            loss = torch.sqrt(
                loss_fn(batch_data['labels'].view(-1).to(self.device), logits)
            )
            count += batch_data['labels'].size(0)
            losses.append(loss.item())
            
            # first forward-backward pass
            self.optimizer.zero_grad()
            loss.backward()
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            logits, _output = self.model(
                    input_ids=batch_data['input_ids'].to(self.device),
                    attention_mask=batch_data['attention_mask'].to(self.device),
                    #token_type_ids=batch_data['token_type_ids'].to(self.device),
                    )
            logits = logits.view(-1)
            torch.sqrt(
                loss_fn(batch_data['labels'].view(-1).to(self.device), logits)
            ).backward()  
            self.optimizer.second_step(zero_grad=True)

            self.scheduler.step()
        
        else:
            with torch.cuda.amp.autocast():
                logits, _output = self.model(
                    input_ids=batch_data['input_ids'].to(self.device),
                    attention_mask=batch_data['attention_mask'].to(self.device),
                    #token_type_ids=batch_data['token_type_ids'].to(self.device),
                    )
                # calculate loss
                #　size -> (batch_size, )
                logits = logits.view(-1)
                loss = torch.sqrt(
                    loss_fn(logits, batch_data['labels'].view(-1).to(self.device))
                    )
                count += batch_data['labels'].size(0)
                losses.append(loss.item())
                
                self.optimizer.zero_grad()
                self.scalar.scale(loss).backward()
                self.scalar.step(self.optimizer)
                self.scalar.update()
                self.scheduler.step()
        self.model.to('cpu')
        self.model.eval()
        return losses, count

    def predict(self, test_loader):
        preds=[]
        self.model.to(self.device)

        for batch_idx, batch_data in enumerate(test_loader):
            with torch.no_grad():
                logits, _output = self.model(
                    input_ids=batch_data['input_ids'].to(self.device),
                    attention_mask=batch_data['attention_mask'].to(self.device),
                    #token_type_ids=batch_data['token_type_ids'].to(self.device),
                    )
                preds+=logits.detach().cpu()
        self.model.to('cpu')
        return preds
        
    def run(self, train_loader, valid_loader, epoch, fold):
        count = 0
        losses = []
        for batch_idx, batch_data in enumerate(train_loader):
            losses, count = self.train(batch_data, losses, count)
            if batch_idx % self.log_interval == 0:
                _s = str(len(str(len(train_loader.sampler))))
                ret = [
                    ('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count, len(train_loader.sampler), 100 * count / len(train_loader.sampler)),
                    'train_loss: {: >4.5f}'.format(np.mean(losses)),
                ]
                print(', '.join(ret))
                self.trn_loss_log.append(np.mean(losses))

            if batch_idx % self.evaluate_interval == 0:
                loss = self.eval(valid_loader)
                self.val_loss_log.append(loss)
                print('Epoch: [{}] valid_loss: {: >4.5f}'.format(epoch, loss))

                if loss < self.best_val_loss:
                    print("{} epoch, best epoch was updated! valid_loss: {: >4.5f}".format(epoch, loss))
                    self.best_val_loss = loss
                    torch.save(self.model.state_dict(), f"model{fold}.bin")