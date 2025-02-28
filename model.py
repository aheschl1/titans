from logging import warning
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchviz import make_dot


class BufferedState(nn.Module):
    def __init__(self):
        super(BufferedState, self).__init__()
        
    def register(self, key, value):
        self.register_buffer(key, value)
        
    def get_value(self, key):
        return getattr(self, key, None)

class NeuralMemory(nn.Module):
    def __init__(self, 
                dim_in: int, 
                dim_out: int, 
                lr: float=1e-3
            ):
        super(NeuralMemory, self).__init__()
        self.register_buffer("model", torch.empty(dim_out, dim_in, requires_grad=True, dtype=torch.float32))
        nn.init.xavier_normal_(self.model)
                   
        self.key = nn.Linear(dim_in, dim_in, bias=False)
        self.value = nn.Linear(dim_in, dim_in, bias=False)
        self.query = nn.Linear(dim_in, dim_in, bias=False)
        
        self.st_past_state = BufferedState()      
        self.lr = torch.tensor(lr)  
        self.surprise_metric = nn.L1Loss(reduction='sum')
    
    def _update_memory(self, grad, eta, alpha):
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
        past_surprise = self.st_past_state.get_value("weight")
        # check state for this params and initialize if not present
        if past_surprise is None:
            past_surprise = torch.zeros_like(self.model)
            self.st_past_state.register("weight", past_surprise)
            
        if grad is None:
            warning(f"Gradient for weight is None. Skipping update.")
        
        surpise = eta*past_surprise + self.lr*grad # surprise_t = eta*surprise_{t-1} + lr*grad
        
        self.st_past_state.register_buffer("weight", surpise)
        # now weights with decay. (1-alpha)*param + surprise
        self.model = self.model*(1-alpha) + surpise

                
    def condition(self, x) -> torch.Tensor:
        """
        Condition the model on the input x
        
        Returns:
        - surprise: the surprise metric
        """
        # prepare the grad. inner loop only updates the model
        k = self.key(x)
        v = self.value(x)
        
        s_t = self.surprise_metric((self.model@k.permute(1,0)).permute(1,0), v) # L1Loss
        # Compute gradients w.r.t. model params
        grads = torch.autograd.grad(s_t, self.model, create_graph=True, retain_graph=True)[0]
        self._update_memory(grads, eta=torch.tensor(0.9), alpha=torch.tensor(0.1))
        return s_t

    def forward(self, x):
        q = self.query(x)
        return (self.model@q.permute(1,0)).permute(1,0)
    
if __name__ == "__main__":
    x = torch.randn(12, 10) # tokens 1 x 10
    model = NeuralMemory(dim_in=10, dim_out=10)
    
    model.condition(x)
    model.condition(x)
    model.condition(x)

    y = model(x) 
    loss = nn.L1Loss()(y, x)

    # loss.backward() # d(MQx - x)/dw1   
    print(model.value.weight.grad)
    print(model.key.weight.grad)  
    print(model.query.weight.grad)
    
    vis = make_dot(loss, params=dict(model.named_parameters()), show_saved=True)
    # save
    vis.render("model", format="png", cleanup=True)
