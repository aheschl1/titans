Unofficial Implmentation for "Titans: Learning to Memorize at Test Time"
https://arxiv.org/pdf/2501.00663


Currently contains the implementation of the NeuralMemory module, as well as Memory as Context (MAC).
The NeuralMemory is implmented as a single matrix.

## NeuralMemory
```python
from pytitans.model.neural_memory import NeuralMemory
import torch

x = torch.randn(B, N, 10)

model = NeuralMemory(dim_in=10, dim_out=10, update_chunk_size=4, lr=0.0001)
x = model.condition(x) # condition
y = model(x)           # sample
```
One notible change from the paper is that eta and alpha are not data dependent; however, they are used as hyperparameters in the loss, and are updated during training.

## Memory as Context
```python
from pytitans.model.mac import MemoryAsContext
import torch

x = torch.randn(B, N, 10)

model = MemoryAsContext(dim_in=10)
y = model(x)           # sample
```