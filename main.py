import torch
import torch_directml
import torch.nn as nn
import math
from encoder import Encoder
from decoder import Decoder

class Embeddings(nn.Module):
    """Paper section 3.4: embedding layers share weights with pre-softmax linear,
    and are multiplied by sqrt(d_model)."""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Paper section 3.5, fixed sinusoidal encoding (eq 1 & 2)."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # not a parameter, but moves with .to(device)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Generator(nn.Module):
    """Final linear + softmax (paper section 3.4)."""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, h=8,
                 d_ff=2048, N=6, dropout=0.1, max_len=5000):
        super().__init__()
        self.src_embed = Embeddings(src_vocab_size, d_model)
        self.tgt_embed = Embeddings(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)

        self.encoder = Encoder(d_model, h, d_ff, N, dropout)
        self.decoder = Decoder(d_model, h, d_ff, N, dropout)

        self.generator = Generator(d_model, tgt_vocab_size)

        self._init_params()

    def _init_params(self):
        # Something called Xavier? Need to look into that
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src):
        x = self.pos_enc(self.src_embed(src))
        return self.encoder(x)

    def decode(self, tgt, enc_out):
        x = self.pos_enc(self.tgt_embed(tgt))
        return self.decoder(x, enc_out)

    def forward(self, src, tgt):
        enc_out = self.encode(src)
        dec_out = self.decode(tgt, enc_out)
        return self.generator(dec_out)

if __name__ == "__main__":
    device = torch_directml.device()

    src_vocab, tgt_vocab = 1000, 1000
    model = Transformer(src_vocab, tgt_vocab).to(device)

    src = torch.randint(0, src_vocab, (2, 10)).to(device)
    tgt = torch.randint(0, tgt_vocab, (2, 10)).to(device)

    out = model(src, tgt)
    print(out.shape)