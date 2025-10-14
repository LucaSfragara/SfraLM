from torch import nn
import torch
from typing import Optional

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = d_model ** 0.5
        self.dropout = nn.Dropout(dropout)

        # init similar to GPT/LLaMA
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            with torch.no_grad():
                self.embed.weight[padding_idx].zero_()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens) * self.scale   # (B,T,D)
        return self.dropout(x)
