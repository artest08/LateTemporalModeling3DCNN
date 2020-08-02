import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512, freq=64):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(freq) / d_model)).exp()
        if d_model%2==1:
             div_term2 = div_term[:-1]
        else:
             div_term2 = div_term
             
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term2)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


    
class LearnedPositionalEmbedding2(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float().to(device='cuda')
        pe.require_grad = True
        pe = pe.unsqueeze(0)
        self.pe=nn.Parameter(pe)
        torch.nn.init.normal_(self.pe,std=0.02)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class LearnedPositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float().to(device='cuda')
        pe.require_grad = True
        pe = pe.unsqueeze(0)
        self.pe=nn.Parameter(pe)
        torch.nn.init.normal_(self.pe,std = d_model ** -0.5)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class LearnedPositionalEmbedding3(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float().to(device='cuda')
        self.a_2 = nn.Parameter(torch.ones_like(pe)).to(device='cuda')
        self.b_2 = nn.Parameter(torch.zeros_like(pe)).to(device='cuda')
        pe.require_grad = True
        pe = pe.unsqueeze(0)
        self.pe=nn.Parameter(pe)
        torch.nn.init.normal_(self.pe, std = d_model ** -0.5)

    def forward(self, x):
        return self.a_2 * self.pe[:, :x.size(1)] + self.b_2 
    