from encoder import PositionwiseFeedForward
from MultiHeadAttentionFunctionImplementation import MultiHeadAttentionFunctionImplementation
import torch.nn as nn
import torch

def make_causal_mask(seq_len, device):
    # lower-triangular matrix: 1 where j <= i (allowed), 0 where j > i (blocked)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask  # (seq_len, seq_len)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout=0.1):
        #REMEMBER: h is number of heads, d_model is size of embedding vectors
        super().__init__()
        self.self_attn = MultiHeadAttentionFunctionImplementation(d_model, h)
        self.mask_self_attn = MultiHeadAttentionFunctionImplementation(d_model, h)
        self.ff = PositionwiseFeedForward(d_model, d_ff)

        # two separate LayerNorms: one after attention, one after FF
        # (each sublayer gets its own norm + residual, per the paper's "Add & Norm")
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, mask=None):
        #need the output of the encoder for the second sublayer

        # --- sublayer 1: masked-self-attention ---
        attn_out, _ = self.mask_self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))  #residual connection (as done in citation 11 in paper) and then norm
        # --- sublayer 2: normal-self-attention ---
        enc_attn_out, _ = self.self_attn(x, enc_out, enc_out)
        x = self.norm2(x + self.dropout2(enc_attn_out))
        # --- sublayer 3: feedforward ---
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout3(ff_out)) #the dropout here and above was added as part of training in section 5

        return x


class Decoder(nn.Module):
    def __init__(self, d_model, h, d_ff, num_layers, dropout=0.1):
        super().__init__()
        # ModuleList, not a plain python list, so PyTorch tracks the params of each layer
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, h, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_out):
        # x: (batch, tgt_len, d_model) -- already embedded + positionally encoded before this point
        seq_len = x.size(1)
        causal_mask = make_causal_mask(seq_len, x.device)

        for layer in self.layers:
            x = layer(x, enc_out, causal_mask)
        return self.norm(x)