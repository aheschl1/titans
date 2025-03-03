import torch
import torch.nn as nn
from pytitan.model.memory import LinearMemory

class NeuralMemory(nn.Module):
    def __init__(self, 
                dim_in: int, 
                dim_out: int,
                update_chunk_size: int,
                lr: float=1e-3,
                initial_eta: float=0.7,
                initial_alpha: float=0.9
            ):
        """ Neural Memory model
        
        Args:
            dim_in: int - the input dimension
            dim_out: int - the output dimension
            update_chunk_size: int - the chunk size for updating the memory
            lr: float - the learning rate for the memory
        """
        super(NeuralMemory, self).__init__()
        self.update_chunk_size = update_chunk_size
        self.memory = LinearMemory(dim_in, dim_out, lr)
        self.key = nn.Linear(dim_in, dim_in, bias=True)
        self.value = nn.Linear(dim_in, dim_in, bias=True)
        self.query = nn.Linear(dim_in, dim_in, bias=True)
        # parameters for the memory
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        self.eta = nn.Parameter(torch.tensor(initial_eta))
        
        self.surprise_metric = nn.L1Loss(reduction='mean') # reduce over batch and sequence length
           
    def condition(self, x) -> torch.Tensor:
        """
        Condition the model on the input x
        
        Returns:
        - surprise: the surprise metric
        """
        # prepare the grad. inner loop only updates the model
        # chunk by chunk
        chunks = torch.split(x, self.update_chunk_size, dim=1)
        s_t_total = 0
        for x in chunks:
            k = self.key(x)
            v = self.value(x)
            s_t = self.surprise_metric(self.memory(k), v) # L1Loss
            s_t_total += s_t.detach()
            # Compute gradients w.r.t. model params
            self.memory.update(s_t, eta=self.eta, alpha=self.alpha)
        return s_t_total 

    def forward(self, x, query=True) -> torch.Tensor:
        """
        Internal forward to operate on chunks of the input x
        """
        if query:
            x = self.query(x)
        return self.memory(x)
    
    def zero_grad(self, set_to_none = True):
        self.memory.zero_grad(set_to_none)
        return super().zero_grad(set_to_none)
     

    
if __name__ == "__main__":
    x = torch.randn(2, 13, 10, device="cuda") # tokens 1 x 10
    model = NeuralMemory(dim_in=10, dim_out=10, update_chunk_size=4)
    model = model.to("cuda")
    
    model.condition(x)
    
    y = model(x) 
    loss = nn.L1Loss()(y, x)

    loss.backward() # d(MQx - x)/dw1   
    print(model.key.weight.grad)
    print(model.value.weight.grad)
    print(model.query.weight.grad)
    print(model.alpha.grad)
    print(model.eta.grad)
    