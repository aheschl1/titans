import torch
import torch.nn as nn
from torchviz import make_dot
from pytitan.model.memory import LinearMemory

class NeuralMemory(nn.Module):
    def __init__(self, 
                dim_in: int, 
                dim_out: int, 
                lr: float=1e-3
            ):
        super(NeuralMemory, self).__init__()
        self.memory = LinearMemory(dim_in, dim_out, lr)
        self.key = nn.Linear(dim_in, dim_in, bias=True)
        self.value = nn.Linear(dim_in, dim_in, bias=True)
        self.query = nn.Linear(dim_in, dim_in, bias=True)
        
        self.surprise_metric = nn.L1Loss(reduction='sum')
           
    def condition(self, x) -> torch.Tensor:
        """
        Condition the model on the input x
        
        Returns:
        - surprise: the surprise metric
        """
        # prepare the grad. inner loop only updates the model
        k = self.key(x)
        v = self.value(x)
        
        s_t = self.surprise_metric(self.memory(k), v) # L1Loss
        # Compute gradients w.r.t. model params
        grads = torch.autograd.grad(s_t, self.memory.get_weights(), create_graph=True, retain_graph=True)
        self.memory.update(grads, eta=torch.tensor(0.9), alpha=torch.tensor(0.1))
        return s_t

    def forward(self, x):
        return self.memory(self.query(x))
    
if __name__ == "__main__":
    x = torch.randn(12, 10, device="cuda") # tokens 1 x 10
    model = NeuralMemory(dim_in=10, dim_out=10)
    model = model.to("cuda")
    
    model.condition(x)
    
    y = model(x) 
    loss = nn.L1Loss()(y, x)

    loss.backward() # d(MQx - x)/dw1   
    print(model.key.weight.grad)
    print(model.value.weight.grad)
    print(model.query.weight.grad)
    