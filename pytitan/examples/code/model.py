from pytitan.model.mac import MemoryAsContext
import torch.nn as nn
import torch
from pytitan.examples.finance.dataset import FinanceDataset

class CodeModel(nn.Module):
    def __init__(
        self, 
        ctx_length: int,
        embed_dim: int,
        vocab_size: int,
        padding_value: int
    ):
        super(CodeModel, self).__init__()
        # from [B, N, 1] to [B, N, E]
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_value)
        
        self.non_lin = nn.SELU()
        
        self.transformer = MemoryAsContext(
            dim_in=embed_dim,
            long_term_memory_dim=embed_dim,
            persistent_memory_dim=16,
            long_term_update_chunk_size=ctx_length // 20
        )
                
        self.transformer2 = MemoryAsContext(
            dim_in=embed_dim,
            long_term_memory_dim=embed_dim,
            persistent_memory_dim=16,
            long_term_update_chunk_size=ctx_length // 20
        )
        
        self.transformer3 = MemoryAsContext(
            dim_in=embed_dim,
            long_term_memory_dim=embed_dim,
            persistent_memory_dim=16,
            long_term_update_chunk_size=ctx_length // 20
        )
        
        self.proj = nn.Linear(
            embed_dim, 
            vocab_size
        )
        

    def forward(self, x):
        x = self.embed(x)
        x2 = self.transformer(x)
        x3 = self.transformer2(self.non_lin(x2 + x))
        x4 = self.transformer3(self.non_lin(x3 + x2))
        x4 = self.proj(x4) # [B, N, V]
        x4 = x4[:, -1, :] # [B, V]
        return x4
