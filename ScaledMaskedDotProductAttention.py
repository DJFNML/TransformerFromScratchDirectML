import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
#See ScaledDotProductAttention.py (largely the same, but adding the mask sublayer for masked multi-head attention. This is used in the decoder (whilst the other version is used in the encoder).
#NOTE: TOTALLY DEFUNCT. IMPLEMENTED THE MASK WRONG, BUT THEN FOUND A WAY TO IMPLEMENT A MASK IN THE ORIGINAL CLASS AS A PARAMETER
#(PROBABLY SHOULD'VE THOUGHT OF THAT BEFORE). ANYWAY, I WAS A BIT SLEEP DEPRIVED WHEN DOING THIS AND TOTALLY FORGOT THAT IT'S
#NOT THE UPPER HALF TRIANGLE THAT GETS REMOVED, JUST AN UPPER TRIANGLE SO THIS DOESN'T REALLY WORK ANYWAY
class ScaledMaskedDotProductAttention(nn.Module):
    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale

    def forward(self, Q, K, V):

        """
        Q: (batch, seq_q, d)
        K: (batch, seq_k, d)
        V: (batch, seq_k, d_v)
        """

        device = torch_directml.device()

        Q = Q.to(device)
        K = K.to(device)
        V = V.to(device)
        scores = torch.matmul(Q, K.transpose(-2, -1))


        if self.scale is None:
            scale = Q.size(-1) ** 0.5
        else:
            scale = self.scale

        scores = scores / scale

        #This is the mask, added for the reason given in the paper (so that the predictions for position i are only affected by the known outputs at positions less than i).
        #I believe this means that the upper right triangle half of the matrix is set to - infty (so the softmax makes these probabilities 0)?
        #Couldn't find any sort of in-built masking in Pytorch but doesn't matter because I've built my own
        #Don't know much about DirectML since this is my first time using it, but I figure it'd probably be smart to put the mask on the GPU with Q
        seq_len = Q.size(1)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device = Q.device), diagonal=1)
        scores = scores + mask

        attn = F.softmax(scores, dim=-1)


        out = torch.matmul(attn, V)

        return out, attn