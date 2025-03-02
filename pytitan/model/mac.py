import torch.nn as nn
import torch
from pytitan.model.neural_memory import NeuralMemory
from torchviz import make_dot

class MemoryAsContext(nn.Module):
    """
    Memory as context model. This model uses a long term memory, short term memory, and persistent memory to condition the output.
    """
    def __init__(
        self,
        dim_in: int,
        short_term_memory_heads: int = 8,
        long_term_update_chunk_size: int = 4,
        long_term_memory_lr: float = 1e-3,
        long_term_memory_dim: int = 16,
        long_term_memory_weight: nn.Module=None,
        persistent_memory_dim: int = 16,
        persistent_memory_weight: nn.Module=None,
    ):
        """ Memory as a context
        
        Args:
            dim_in: int - the input dimension
            short_term_memory_heads: int - the number of heads for the short term memory
            long_term_update_chunk_size: int - the chunk size for updating the long term memory
            long_term_memory_lr: float - the learning rate for the long term memory
            long_term_memory_dim: int - the dimension of the long term memory
            long_term_memory_weight: nn.Module - the weight for the long term memory. if None, it will be initialized
            persistent_memory_dim: int - the dimension of the persistent memory
            persistent_memory_weight: nn.Module - the weight for the persistent memory. if None, it will be initialized
        """
        super(MemoryAsContext, self).__init__()
        # is embdedding dim info
        self.persistent_memory = persistent_memory_weight or self.initialize_persistent_memory(persistent_memory_dim)
        self.long_term_memory = long_term_memory_weight or NeuralMemory(
            dim_in, 
            long_term_memory_dim, 
            update_chunk_size=long_term_update_chunk_size, 
            lr=long_term_memory_lr
        )
        
        embed_dim = dim_in + long_term_memory_dim + persistent_memory_dim
        self.short_term_memory = nn.MultiheadAttention(embed_dim, short_term_memory_heads, batch_first=True)
        self.short_term_projection = nn.Linear(embed_dim, dim_in)
            
    def initialize_persistent_memory(self, persistent_memory_dim: int):
        persistent_memory = nn.Parameter(torch.empty(1, 1, persistent_memory_dim, requires_grad=True))
        nn.init.xavier_normal_(persistent_memory)
        return persistent_memory
            
    def forward(self, x):
        """
        Given an input x, compute the forward pass of the model.

        Sample from the long term memory, then short term memory, then update the long term memory, and use the output to gate the short term memory.
        """
        b = x.shape[0]
        n = x.shape[1]
                
        lt_mem = self.long_term_memory(x)
        st_mem_token = torch.cat([self.persistent_memory.expand(b, n, -1), lt_mem, x], dim=2)
        # apply attention on st_mem_token
        # shape is [B, N, E]
        st_mem, _ = self.short_term_memory(st_mem_token, st_mem_token, st_mem_token)
        # update long term memory
        st_mem = self.short_term_projection(st_mem)
        self.long_term_memory.condition(st_mem)
        # sample from the long term memory
        y = self.long_term_memory(st_mem, query=False)
        return st_mem * y
    
if __name__ == "__main__":
    x = torch.randn(2, 100, 16, device="cuda") # tokens 1 x 10
    model = MemoryAsContext(dim_in=16)
    model = model.to("cuda")
    
    y = model(x)
    print(y.shape)
    # visualize the model
    # dot = make_dot(y, params=dict(model.named_parameters()))
    # dot.render("memory_as_context")
