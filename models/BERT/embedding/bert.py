import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding, LearnedPositionalEmbedding2, LearnedPositionalEmbedding, LearnedPositionalEmbedding3
from .segment import SegmentEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, input_dim, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.position = PositionalEmbedding(d_model=input_dim,max_len=max_len, freq=64)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.position(sequence)+sequence
        return self.dropout(x)
    
class BERTEmbedding3(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, input_dim, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.learnedPosition = LearnedPositionalEmbedding(d_model=input_dim,max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.learnedPosition(sequence)+sequence
        return self.dropout(x)
    
class BERTEmbedding2(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, input_dim, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.learnedPosition = LearnedPositionalEmbedding2(d_model=input_dim,max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.learnedPosition(sequence)+sequence
        return self.dropout(x)
    
    
class BERTEmbedding4(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, input_dim, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.learnedPosition = LearnedPositionalEmbedding3(d_model=input_dim,max_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        x = self.learnedPosition(sequence)+sequence
        return self.dropout(x)
    
