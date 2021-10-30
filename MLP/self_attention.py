import copy
import torch
from torch import nn

class AttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, layer_norm_eps=1e-5):
        super(AttentionEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead,dropout=dropout) # (seq_length, batch_num, d_model)=> same_size
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)# (seq_length, batch_num, d_model)=>(_, _, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)# (seq_length, batch_num, dim_feedforward)=>(_, _, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)# (seq_length, batch_num, d_model)=>same_size
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)# (seq_length, batch_num, d_model)=>same_size
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self,x):
        x2, attention = self.self_attn(x, x, x, attn_mask=None, key_padding_mask=None)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.relu(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x, attention

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
    def forward(self, x):
        output = x
        for i, mod in enumerate(self.layers):
            output, atn = mod(output)
            if i==0:
                attention = atn
        return output, attention

def attention_forward(x, layer):
    N =  x.size(0)
    inp_dim = x.size(1)
    h_dim = x.size(2)
    w_dim = x.size(3)
    att_inp = torch.flatten(x,2).permute(2,0,1)#（バッチサイズ, 次元数, W, H）=>（バッチサイズ, 次元数,系列長）=>（系列長, バッチサイズ, 次元数）
    att_inp, attention = layer(att_inp)
    att_inp = att_inp.permute(1,2,0)
    att_inp = att_inp.reshape(N, inp_dim, h_dim, w_dim)
    return att_inp, attention

"""
self.transformer_encoder_1 = Encoder(
    AttentionEncoderLayer(d_model=input_dim, nhead=input_dim), num_layers=3
    )
attention_forward(x, self.transformer_encoder_1) #(batch, d_model, W, H)=>(batch, d_model, W, H)
x = torch.flatten(x, 1) #(batch, d_model, W, H) => (batch, d_model * W * H)
"""