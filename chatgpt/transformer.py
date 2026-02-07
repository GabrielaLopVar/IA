import torch
import torch.nn as nn
from .attention import MultiHeadAttention  # Importamos tu clase
from .network import FeedForward           # Importamos tu clase

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Capa de Normalización 1
        self.ln1 = nn.LayerNorm(d_model)
        
        # Tu mecanismo de atención
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        
        # Capa de Normalización 2
        self.ln2 = nn.LayerNorm(d_model)
        
        # Tu red de paso hacia adelante
        self.feed_forward = FeedForward(d_model=d_model, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Atención con Conexión Residual
        # "x + ..." es lo que permite que el gradiente fluya bien
        attn_out = self.attention(self.ln1(x), mask=mask)
        x = x + self.dropout(attn_out)
        
        # 2. FeedForward con Conexión Residual
        ff_out = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_out)
        
        return x