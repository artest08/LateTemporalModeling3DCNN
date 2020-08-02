import torch.nn as nn
import torch
from .single import Attention, Attention2


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
    
class MultiHeadedAttention2(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers1 = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.linear_layers2 = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear1 = nn.Linear(d_model, d_model)
        self.output_linear2 = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.attention2 =  Attention2()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_rgb, input_flow, mask=None):
        batch_size = input_rgb.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query_rgb, key_rgb, value_rgb = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers1, (input_rgb, input_rgb, input_rgb))]
        
        query_flow, key_flow, value_flow = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers2, (input_flow, input_flow, input_flow))]

        # 2) Apply attention on all the projected vectors in batch.
        _ , attn_rgb = self.attention( query_rgb, key_rgb, value_rgb, mask=mask)
        _ , attn_flow = self.attention( query_flow, key_flow, value_flow, mask=mask)

        attn = torch.max(attn_rgb, attn_flow)
        
        output_rgb , _ = self.attention2(value_rgb, attn, self.dropout)
        output_flow , _ = self.attention2(value_flow, attn, self.dropout)
        
        # 3) "Concat" using a view and apply a final linear.
        output_rgb = output_rgb.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        output_flow = output_flow.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear1(output_rgb), self.output_linear2(output_flow)
    
    

