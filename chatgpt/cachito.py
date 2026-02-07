import torch
import torch.nn as nn
import math
from typing import Optional

class SequenceEmbedding(nn.Module):
   
    def __init__(self, vocab_dim: int, embedding_dim: int, max_sequence_len: int = 512) -> None:
        
        super().__init__()
        self.embedding_dim: int = embedding_dim
        
        self.lexical_embedding: nn.Embedding = nn.Embedding(vocab_dim, embedding_dim)
        
        self.positional_embedding: nn.Embedding = nn.Embedding(max_sequence_len, embedding_dim)

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
       
        batch_size, seq_len = token_indices.size()
        
        position_indices: torch.Tensor = torch.arange(
            seq_len, 
            device=token_indices.device
        ).expand(batch_size, seq_len)
        
        embeddings = self.lexical_embedding(token_indices) * math.sqrt(self.embedding_dim)
        embeddings = embeddings + self.positional_embedding(position_indices)
        
        return embeddings

    def get_vocab_embeddings(self) -> torch.Tensor:
       
        return self.lexical_embedding.weight

    def get_positional_embeddings(self) -> torch.Tensor:
       
        return self.positional_embedding.weight