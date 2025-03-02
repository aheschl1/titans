import torch.nn as nn
import torch
from pytitan.model.neural_memory import NeuralMemory

class MemoryAsContext(nn.Module):
    def __init__(
        self,
        dim_in: int,
        short_term_memory_heads: int = 8,
        long_term_memory_lr: float = 1e-3,
        long_term_memory_dim: int = 64,
        long_term_memory_weight: nn.Module=None,
        persistent_memory_dim: int = 64,
        persistent_memory_weight: nn.Module=None,
    ):
        super(MemoryAsContext, self).__init__()
        self.persistent_memory = persistent_memory_weight or self.initialize_persistent_memory(persistent_memory_dim)
        embed_dim = dim_in + long_term_memory_dim + persistent_memory_dim
        self.long_term_memory = long_term_memory_weight or NeuralMemory(embed_dim, long_term_memory_dim, long_term_memory_lr)
        self.short_term_memory = nn.MultiheadAttention(dim_in, short_term_memory_heads, batch_first=True)
            
    def initialize_persistent_memory(self, persistent_memory_dim: int):
        persistent_memory = nn.Parameter(torch.empty(1, persistent_memory_dim, requires_grad=True))
        nn.init.xavier_normal_(persistent_memory)
        return persistent_memory
            
    def forward(self, x):
        """
        Given an input x, compute the forward pass of the model.

        Sample from the long term memory, then short term memory, then update the long term memory, and use the output to gate the short term memory.
        """
        lt_mem = self.long_term_memory(x)
        st_mem_token = torch.cat([x, lt_mem, self.persistent_memory], dim=1)
        # apply attention on st_mem_token
        st_mem = self.short_term_memory(st_mem_token, st_mem_token, st_mem_token)
        # update long term memory
        self.long_term_memory.condition(st_mem)
        # sample from the long term memory
        y = self.long_term_memory(st_mem)
        return st_mem * y
    
if __name__ == "__main__":
    x = torch.randn(12, 16, device="cuda") # tokens 1 x 10
    model = MemoryAsContext(dim_in=16)
    model = model.to("cuda")
    
    y = model(x)
    print(y.shape)