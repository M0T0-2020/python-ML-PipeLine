import torch
from torch import nn
        
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.in_size = config.in_size
        self.out_size = config.out_size
        self.hidden_layers = config.hidden_layers
        
        self.layers = nn.ModuleList()
        for i, hidden_size in enumerate(self.hidden_layers):
            if i==0:
                lin = nn.Linear(self.in_size, hidden_size)
                self._init_weights(lin)
                self.layers.append(lin)
                self.layers.append(nn.ReLU())
            else:
                lin = nn.Linear(self.hidden_layers[i-1], hidden_size)
                self._init_weights(lin)
                self.layers.append(lin)
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(0.1))
        lin = nn.Linear(self.hidden_layers[-1], self.out_size)
        self._init_weights(lin)
        self.layers.append(lin)
        
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
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x