import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
#We create our attention mechanism as per the instructions in the paper, to implement the given attention function
class EncoderLayer(nn.Module):
    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale  # if None, we compute it dynamically

    def forward(self, Q):
        #docstring gives info on query matrix (encoder input)
        """
        Q: (batch, seq_q, d)
        """
