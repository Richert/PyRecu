import torch
from torch.nn import Module, Linear, Tanh, Softmax, Softmin, Sigmoid, Identity
from typing import Iterator
from .input_layer import LinearStatic


class OutputLayer(Module):

    def __init__(self, n: int, m: int, weights: torch.Tensor = None, trainable: bool = False, transform: str = None,
                 dtype: torch.dtype = torch.float64):

        super().__init__()

        # initialize output weights
        if trainable:
            self.layer = Linear(n, m, dtype=dtype)
        else:
            if weights is None:
                weights = torch.randn(m, n, dtype=dtype)
            elif weights.dtype != dtype:
                weights = torch.tensor(weights, dtype=dtype)
            self.layer = LinearStatic(weights)

        # define output function
        if transform is None:
            self.transform = Identity()
        elif transform == 'tanh':
            self.transform = Tanh()
        elif transform == 'softmax':
            self.transform = Softmax()
        elif transform == 'softmin':
            self.transform = Softmin()
        elif transform == 'sigmoid':
            self.transform = Sigmoid()

    def forward(self, x):
        projection = self.layer(x)
        return self.transform(projection)

    def parameters(self, recurse: bool = True) -> Iterator:
        return self.layer.parameters(recurse=recurse)
