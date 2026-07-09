import torch
import torch.nn as nn
from MultiHeadAttentionFunctionImplementation import MultiHeadAttentionFunctionImplementation

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # paper uses d_ff = 2048 for d_model = 512
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        #feedforward uses "two linear transformations with a relu in between"
    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        #REMEMBER: h is number of heads, d_model is size of embedding vectors
        super().__init__()
        self.self_attn = MultiHeadAttentionFunctionImplementation(d_model, h)
        self.ff = PositionwiseFeedForward(d_model, d_ff)

        # two separate LayerNorms: one after attention, one after FF
        # (each sublayer gets its own norm + residual, per the paper's "Add & Norm")
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # --- sublayer 1: self-attention ---
        # Q, K, V are all x here since it's *self*-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))  #residual connection (as done in citation 11 in paper) and then norm
        # --- sublayer 2: feedforward ---
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out)) #the dropout here and above was added as part of training in section 5

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, h, d_ff, num_layers, dropout=0.1):
        super().__init__()
        # ModuleList, not a plain python list so PyTorch tracks the params of each layer
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers) #stack all the encoder layers
        ])

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model) — already embedded + positionally encoded before this point
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
