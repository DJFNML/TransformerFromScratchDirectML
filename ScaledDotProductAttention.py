import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
#We create our attention mechanism as per the instructions in the paper, to implement the given attention function
class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale  # if None, we compute it dynamically

    def forward(self, Q, K, V, mask = None):
        #docstring gives info on query, key, value matrix
        """
        Q: (batch, seq_q, d)
        K: (batch, seq_k, d)
        V: (batch, seq_k, d_v)
        """
        #Setup: put everything on the GPU using DirectML (I think? Not sure if it should go here or in main, and pass the whole thing to the GPU then. Will see later)
        device = torch_directml.device()

        Q = Q.to(device)
        K = K.to(device)
        V = V.to(device)
        # Matmul: QK^T (Maths grads will know: computes a bunch of dot products)
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_q, seq_k)

        #Scale (look at the original paper, and see how we scale by the root of the dimensions)
        if self.scale is None:
            scale = Q.size(-1) ** 0.5
        else:
            scale = self.scale

        scores = scores / scale

        #Applies the given mask and sets the appropriate values to -infty (so the softmax takes those probabilities to 0)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax (over keys dimension, classic way to return the probability distribution)
        attn = F.softmax(scores, dim=-1)

        # Matmul with V (multiply in the value matrix)
        out = torch.matmul(attn, V)

        return out, attn
