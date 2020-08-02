import torch.nn as nn

import torch

from .transformer import TransformerBlock, TransformerBlock2
from .embedding import BERTEmbedding, BERTEmbedding2, BERTEmbedding3, BERTEmbedding4



class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, input_dim, max_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, mask_prob=0.8):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len=max_len
        self.input_dim=input_dim
        self.mask_prob=mask_prob
        
        
        clsToken = torch.zeros(1,1,self.input_dim).float().cuda()
        clsToken.require_grad = True
        self.clsToken= nn.Parameter(clsToken)
        torch.nn.init.normal_(self.clsToken, std = hidden ** -0.5)
        self.a_2 = nn.Parameter(torch.ones_like(self.clsToken))
        self.b_2 = nn.Parameter(torch.zeros_like(self.clsToken))
        
        

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding4(input_dim=input_dim, max_len=max_len+1)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                #nn.init.xavier_normal_(module.weight)
                nn.init.normal_(module.weight, mean=0, std = 0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
         
        # nn.init.xavier_normal_(self.transformer_blocks[0].feed_forward.w_2.weight, gain = 1/(0.425) ** 0.5)
        # nn.init.xavier_normal_(self.transformer_blocks[0].feed_forward.w_1.weight, gain = 1)

     
    def forward(self, input_vectors):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        batch_size=input_vectors.shape[0]
        sample=None
        if self.training:
            bernolliMatrix=torch.cat((torch.tensor([1]).float().cuda(), (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)), 0).unsqueeze(0).repeat([batch_size,1])
            self.bernolliDistributor=torch.distributions.Bernoulli(bernolliMatrix)
            sample=self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask=torch.ones(batch_size,1,self.max_len+1,self.max_len+1).cuda()

        # embedding the indexed sequence to sequence of vectors
        clstoken_scales = self.clsToken * self.a_2 + self.b_2
        x = torch.cat((clstoken_scales.repeat(batch_size,1,1),input_vectors),1)
        x = self.embedding(x)
        
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        return x, sample  

class BERT2(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, input_dim, max_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, mask_prob=0.8):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len=max_len
        self.input_dim=input_dim
        self.mask_prob=mask_prob
        
        
        clsToken = torch.zeros(1,1,self.input_dim).float().cuda()
        clsToken.require_grad = True
        self.clsToken= nn.Parameter(clsToken)
        torch.nn.init.normal_(self.clsToken, std = hidden ** -0.5)
        
        

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding3(input_dim=input_dim, max_len=max_len+1)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])
  
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                #nn.init.normal_(module.weight, mean=0, std=0.02)
                nn.init.uniform_(module.weight, -0.06, 0.06)
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            if isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
     
    def forward(self, input_vectors):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        batch_size=input_vectors.shape[0]
        sample=None
        if self.training:
            bernolliMatrix=torch.cat((torch.tensor([1]).float().cuda(), (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)), 0).unsqueeze(0).repeat([batch_size,1])
            self.bernolliDistributor=torch.distributions.Bernoulli(bernolliMatrix)
            sample=self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask=torch.ones(batch_size,1,self.max_len+1,self.max_len+1).cuda()

        # embedding the indexed sequence to sequence of vectors
        x = torch.cat((self.clsToken.repeat(batch_size,1,1),input_vectors),1)
        x = self.embedding(x)
        
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        return x, sample  
        
class BERT3(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, input_dim, max_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, mask_prob=0.8):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len=max_len
        self.input_dim=input_dim
        self.mask_prob=mask_prob
        
        
        clsToken = torch.zeros(1,1,self.input_dim).float().cuda()
        clsToken.require_grad = True
        self.clsToken= nn.Parameter(clsToken)
        torch.nn.init.normal_(clsToken,std=0.02)
        

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden 

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding2(input_dim=input_dim, max_len=max_len+1)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

    
    
    def forward(self, input_vectors):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        batch_size=input_vectors.shape[0]
        sample=None
        if self.training:
            bernolliMatrix=torch.cat((torch.tensor([1]).float().cuda(), (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)), 0).unsqueeze(0).repeat([batch_size,1])
            self.bernolliDistributor=torch.distributions.Bernoulli(bernolliMatrix)
            sample=self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask=torch.ones(batch_size,1,self.max_len+1,self.max_len+1).cuda()

        # embedding the indexed sequence to sequence of vectors
        x = torch.cat((self.clsToken.repeat(batch_size,1,1),input_vectors),1)
        x = self.embedding(x)
        
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        return x, sample  

class BERT4(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, input_dim, max_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, mask_prob=0.8):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len=max_len
        self.input_dim=input_dim
        self.mask_prob=mask_prob
        
        
        clsToken = torch.zeros(1,1,self.input_dim).float().cuda()
        clsToken.require_grad = True
        self.clsToken= nn.Parameter(clsToken)
        torch.nn.init.normal_(self.clsToken,std=0.02)
        
        

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding2(input_dim=input_dim, max_len=max_len+1)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])
  
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            if isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
     
    def forward(self, input_vectors):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        batch_size=input_vectors.shape[0]
        sample=None
        if self.training:
            bernolliMatrix=torch.cat((torch.tensor([1]).float().cuda(), (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)), 0).unsqueeze(0).repeat([batch_size,1])
            self.bernolliDistributor=torch.distributions.Bernoulli(bernolliMatrix)
            sample=self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask=torch.ones(batch_size,1,self.max_len+1,self.max_len+1).cuda()

        # embedding the indexed sequence to sequence of vectors
        x = torch.cat((self.clsToken.repeat(batch_size,1,1),input_vectors),1)
        x = self.embedding(x)
        
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        return x, sample  
    


class BERT5(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, input_dim, max_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, mask_prob=0.8):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len=max_len
        self.input_dim=input_dim
        self.mask_prob=mask_prob
        
        
        clsToken = torch.zeros(1,1,self.input_dim).float().cuda()
        clsToken.require_grad = True
        self.clsToken= nn.Parameter(clsToken)
        torch.nn.init.normal_(clsToken,std=0.02)
        

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding2(input_dim=input_dim, max_len=max_len+1)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

    
    
    def forward(self, input_vectors):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        batch_size=input_vectors.shape[0]
        sample=None
        if self.training:
            bernolliMatrix=torch.cat((torch.tensor([1]).float().cuda(), (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)), 0).unsqueeze(0).repeat([batch_size,1])
            self.bernolliDistributor=torch.distributions.Bernoulli(bernolliMatrix)
            sample=self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask=torch.ones(batch_size,1,self.max_len+1,self.max_len+1).cuda()

        # embedding the indexed sequence to sequence of vectors
        x = torch.cat((self.clsToken.repeat(batch_size,1,1),input_vectors),1)
        x = self.embedding(x)
        
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        return x, sample    
    
    
class BERT6(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, input_dim, max_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, mask_prob=0.8):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len=max_len
        self.input_dim=input_dim
        self.mask_prob=mask_prob
        
        
        clsToken = torch.zeros(1,1,self.input_dim).float().cuda()
        clsToken.require_grad = True
        self.clsToken= nn.Parameter(clsToken)
        torch.nn.init.normal_(self.clsToken,std=0.02)
        
        

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(input_dim=input_dim, max_len=max_len+1)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])
  
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            if isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
     
    def forward(self, input_vectors):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        batch_size=input_vectors.shape[0]
        sample=None
        if self.training:
            bernolliMatrix=torch.cat((torch.tensor([1]).float().cuda(), (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)), 0).unsqueeze(0).repeat([batch_size,1])
            self.bernolliDistributor=torch.distributions.Bernoulli(bernolliMatrix)
            sample=self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask=torch.ones(batch_size,1,self.max_len+1,self.max_len+1).cuda()

        # embedding the indexed sequence to sequence of vectors
        x = torch.cat((self.clsToken.repeat(batch_size,1,1),input_vectors),1)
        x = self.embedding(x)
        
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        return x, sample  
    
class BERT7(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, input_dim, max_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, mask_prob=0.8):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len=max_len
        self.input_dim=input_dim
        self.mask_prob=mask_prob
        
        
        clsToken = torch.zeros(1,1,self.input_dim).float().cuda()
        clsToken.require_grad = True
        torch.nn.init.normal_(clsToken,std=0.02)
        self.clsToken= nn.Parameter(clsToken)
        
        
        maskToken = torch.zeros(1,1,self.input_dim).float().cuda()
        maskToken.require_grad = True
        torch.nn.init.normal_(maskToken,std=0.02)
        self.maskToken= nn.Parameter(maskToken)
        

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding2(input_dim=input_dim, max_len=max_len+1)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

    
    
    def forward(self, input_vectors):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        batch_size=input_vectors.shape[0]
        sample=None
        mask = None


        mask=torch.ones(batch_size,1,self.max_len+1,self.max_len+1).cuda()
        # embedding the indexed sequence to sequence of vectors
        x = torch.cat((self.clsToken.repeat(batch_size,1,1),input_vectors),1)
        
        
        if self.training:
            bernolliMatrix=torch.cat((torch.tensor([1]).float().cuda(), (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)), 0).unsqueeze(0).repeat([batch_size,1])
            self.bernolliDistributor=torch.distributions.Bernoulli(bernolliMatrix)
            sample=self.bernolliDistributor.sample()
            x[sample == 0] = self.maskToken
           
        x = self.embedding(x)
            
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        return x, sample  
    


    

class BERT5_BOTH(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, input_dim, max_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, mask_prob=0.8):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len=max_len
        self.input_dim=input_dim
        self.mask_prob=mask_prob
        
        
        clsToken_rgb = torch.zeros(1,1,self.input_dim).float().cuda()
        clsToken_rgb.require_grad = True
        torch.nn.init.normal_(clsToken_rgb,std=0.02)
        self.clsToken_rgb= nn.Parameter(clsToken_rgb)
        
        clsToken_flow = torch.zeros(1,1,self.input_dim).float().cuda()
        clsToken_flow.require_grad = True
        torch.nn.init.normal_(clsToken_flow,std=0.02)
        self.clsToken_flow= nn.Parameter(clsToken_flow)
        

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding1 = BERTEmbedding2(input_dim=input_dim, max_len=max_len+1)
        self.embedding2 = BERTEmbedding2(input_dim=input_dim, max_len=max_len+1)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock2(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

    
    
    def forward(self, input_vectors_rgb, input_vectors_flow):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        batch_size=input_vectors_rgb.shape[0]
        sample=None
        if self.training:
            bernolliMatrix=torch.cat((torch.tensor([1]).float().cuda(), (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)), 0).unsqueeze(0).repeat([batch_size,1])
            self.bernolliDistributor=torch.distributions.Bernoulli(bernolliMatrix)
            sample=self.bernolliDistributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask=torch.ones(batch_size,1,self.max_len+1,self.max_len+1).cuda()

        # embedding the indexed sequence to sequence of vectors
        input_vectors_rgb = torch.cat((self.clsToken_rgb.repeat(batch_size,1,1),input_vectors_rgb),1)
        input_vectors_rgb = self.embedding1(input_vectors_rgb)
        
        input_vectors_flow = torch.cat((self.clsToken_flow.repeat(batch_size,1,1),input_vectors_flow),1)
        input_vectors_flow = self.embedding1(input_vectors_flow)
        
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x_rgb, x_flow = transformer.forward(input_vectors_rgb, input_vectors_flow, mask)
        
        return x_rgb, x_flow, sample          

