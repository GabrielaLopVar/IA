import torch
import torch.nn as nn
from torch.nn import functional as F

# Cargamos la configuración para mantener consistencia
try:
    import config as cfg
    Config = cfg.Config
except ImportError:
    class Config:
        N_EMBED = 128
        N_HEAD = 4
        N_LAYER = 4
        DROPOUT = 0.1
        MAX_SEQUENCE_LEN = 64

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(Config.N_EMBED, head_size, bias=False)
        self.query = nn.Linear(Config.N_EMBED, head_size, bias=False)
        self.value = nn.Linear(Config.N_EMBED, head_size, bias=False)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        
        # Cálculo de afinidad (Atención)
        # 
        wei = q @ k.transpose(-2, -1) * C**-0.5
        
        # MÁSCARA DINÁMICA: Se ajusta al tamaño T actual para evitar el RuntimeError
        tril = torch.tril(torch.ones(T, T, device=x.device))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(Config.N_EMBED, Config.N_EMBED)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(Config.DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class CachitoGPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, Config.N_EMBED)
        self.position_embedding_table = nn.Embedding(max_seq_len, Config.N_EMBED)
        self.blocks = nn.Sequential(*[Block(Config.N_EMBED, Config.N_HEAD) for _ in range(Config.N_LAYER)])
        self.ln_f = nn.LayerNorm(Config.N_EMBED)
        self.lm_head = nn.Linear(Config.N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) 
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss