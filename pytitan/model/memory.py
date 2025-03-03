from abc import ABC, abstractmethod
from logging import warning
import torch
import torch.nn as nn
from typing import List, Tuple


class MemoryModule(ABC, nn.Module):
    def __init__(self, dim_in, dim_out, lr):
        """
        Memory module for the neural memory.
        """
        super(MemoryModule, self).__init__()
        self.lr = torch.tensor(lr)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self._construct_layers()
        
    @abstractmethod
    def construct_layers(self) -> List[Tuple[str, torch.Tensor]]:
        """
        Create the weights for the model. This is an abstract method that should be implemented by the subclass.
        
        Returns:
        - List of tuples of the form (name, weight) where name is the name of the weight and weight is the tensor.
        """
        ...
    
    @abstractmethod
    def forward(self, x: torch.Tensor)->torch.Tensor:
        ...
        
    def _construct_layers(self):
        """
        Create the buffers for the model. Also create the surprise buffers. One for each weight.
        """
        weights = self.construct_layers()
        for name, weight in weights:
            self.register_buffer(name, weight)
            self.register_buffer(MemoryModule._get_corresponding_surprise_name(name), torch.zeros_like(weight, requires_grad=False))
            
    def _update_memory(self, grads, eta, alpha):
        """
        Optimizer for the neural memory.
        
        Updates based on the gradient of the surprise metric.
        
        M_t = (1-alpha)*M_{t-1} + s_t
        s_t = eta*s_{t-1} + lr*grad
        
        Args:
        - grads: the gradient of the surprise metric
        - eta: data dependent momentum
        - alpha: decay factor
        """
        if len(grads) != len(self.get_named_weights()):
            raise ValueError(f"Number of gradients {len(grads)} does not match number of weights {len(self.get_named_weights())}")
        
        for grad, (name, weight) in zip(grads, self.get_named_weights()):
            sname = MemoryModule._get_corresponding_surprise_name(name)
            # get the past surprise for this weight
            past_surprise = self.get_buffer(sname)
            if grad is None:
                warning(f"Gradient for weight is None. Skipping update.")
                continue
            surpise = eta*past_surprise + self.lr*grad # surprise_t = eta*surprise_{t-1} + lr*grad
            self.register_buffer(sname, surpise.detach())  # Detach from computation graph
            self.register_buffer(name, (1-alpha)*weight + surpise)
    
    def update(self, loss, eta, alpha):
        """
        Update the memory with the gradients
        
        Args:
        - loss: the loss function to optimize
        - eta: data dependent momentum
        - alpha: decay factor
        """
        grads = torch.autograd.grad(loss, self.get_weights(), create_graph=True)
        self._update_memory(grads, eta, alpha)

    def zero_grad(self, set_to_none=True):
        """
        Clears gradients for the memory module while ensuring surprise buffers
        do not retain computation history.
        """
        for name, buffer in self.named_buffers():
            if "surprise" in name:
                continue
            nb = buffer.detach()
            nb.requires_grad_(True)
            self.register_buffer(name, nb)  # Detach from computation graph
        return super().zero_grad(set_to_none)

    
    @staticmethod
    def _get_corresponding_surprise_name(name):
        """
        Get the name of the surprise buffer corresponding to a weight name.
        """
        return f"{name}_surprise_tpast"
        
    def get_named_weights(self) -> List[Tuple[str, torch.Tensor]]:
        return [(name, weight) for name, weight in self.named_buffers() if "surprise" not in name]
    
    def get_weights(self) -> List[torch.Tensor]:
        return [weight for _, weight in self.get_named_weights()]
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
        
class LinearMemory(MemoryModule):
    """
    A linear memory module, which is a simple matrix multiplication. No bias. No activation.
    """
    def construct_layers(self) -> List[Tuple[str, torch.Tensor]]:
        w = torch.empty(self.dim_out, self.dim_in, requires_grad=True, dtype=torch.float32)
        nn.init.xavier_normal_(w)
        return [("model", w)]
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        Compute the forward pass of the model.
        
        Args:
        - x: input tensor of shape (batch_size, dim_in)
        
        Returns:
        - y: output tensor of shape (batch_size, dim_out)
        """
        return (self.model@x.permute(0, 2,1)).permute(0, 2, 1)