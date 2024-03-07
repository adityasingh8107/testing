#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
import math


# In[5]:


class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# In[10]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


# In[11]:


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float=1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) 
        self.bias = nn.Parameter(torch.zeros(features)) 

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# In[12]:


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# In[13]:


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        batch_size, seq_len, d_model = x.size()
        q = self.w_q(x).view(batch_size, seq_len, self.h, self.d_k)
        k = self.w_k(x).view(batch_size, seq_len, self.h, self.d_k)
        v = self.w_v(x).view(batch_size, seq_len, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores.masked_fill_(mask == 0, -1e9)  # Masking
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.w_o(attn_output)
        return attn_output


# In[14]:


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x+ self.dropout(sublayer(self.norm(x)))


# In[17]:


class EncoderBlock (nn.Module):

    def __init__ (self, self_attention_block: MultiHeadSelfAttention, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward (self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self,feed_forward_block)
        return x


# In[18]:


class Encoder(nn.Module):
    def __init__ (self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norma = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        


# In[20]:


class DecoderBlock(nn.Module):

    def __init__ (self, self_attentoion_block: MultiHeadSelfAttention, cross_attention_block: MultiHeadSelfAttention, feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self,feed_forward_block = feedforward_block
        self.residual_connections = nn.Module([ResidualConnection(dropout) for _ in range(3)])

    def forward(Self, x, encoder_output, src_mask, tgt_mask):
        x = self.resudual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.resudual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

        


# In[21]:


class Decoder(nn.Module):

    def __init__ (self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
        


# In[22]:


class ProjectionLayer(nn.Module):

    def __init__ (self, d_model: int, vocab_size: int) ->None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim =-1)


# In[23]:


class Transformer (nn.Module):
    
    def __init__ (self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer)-> None:
        super().__init()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode (self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_mask(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


# In[24]:


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seg_len: int, d_model: int = 512, N:int =6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos - PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock (d_model, h, dropout)
        feed_forward_block = FeedForwardBlock (d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_block = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock (d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock (d_model, d_ff, dropout)
        encoder_block = EncoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer


# In[25]:


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, num_layers: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MultiHeadSelfAttention(d_model, h, dropout),
                FeedForward(d_model, d_ff, dropout)
            ]) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([LayerNormalization(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for self_attention, feed_forward, layer_norm in zip(*self.layers, self.layer_norms):
            attn_output = self_attention(x, mask)
            x = x + self.dropout(layer_norm(attn_output))
            ff_output = feed_forward(x)
            x = x + self.dropout(layer_norm(ff_output))
        return x

class Classifier(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        return x


# In[ ]:




