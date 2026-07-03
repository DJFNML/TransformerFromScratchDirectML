import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
from ScaledDotProductAttention import ScaledDotProductAttention

class MultiHeadAttentionFunctionImplementation(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        #the initial paper says that instead of using a single attention function, a linear projection works better
        #these projections work in parallel, and allegedly produce a better result for a similar computational cost
        #this is basically how we're splitting up into different heads; h is the number of heads, and d_k = d_v, as in
        #the original paper.
        assert d_model % h == 0, f"Cannot split into heads unevenly"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h  # dimension per head

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # the final "Linear" after concat

    def forward(self, Q, K, V, mask = None):
        #docstring same as in other modules
        """
        Q: (batch, seq_q, d)
        K: (batch, seq_k, d)
        V: (batch, seq_k, d_v)
        """
        device = torch_directml.device()

        batch_size = q.size(0)

        # take the original values in the full dimensions
        Q = self.W_q(q)  # (batch, seq_len, d_model)
        K = self.W_k(k)
        V = self.W_v(v)

        Q = Q.to(device)
        K = K.to(device)
        V = V.to(device)

        # split into heads: (batch, seq_len, d_model) --> (batch, seq_len, h, d_k)
        # then swap around seq_len and h with transpose so the attention function works properly (batch, h, seq_len, d_k)
        # the attention function works properly because based off of what we defined, we need seq_len and d_k to be
        #the last two parameters
        #also just use d_k here because it's the same as d_v and I'm lazy
        Q = Q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # apply the scaled dot-product attention (with the relevant mask(
        attn_output, attn_weights = ScaledDotProductAttention(Q, K, V, mask)

        # concat heads back into full dimension: (batch, h, seq_len, d_k) --> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # final linear
        output = self.W_o(attn_output)
        return output, attn_weights



