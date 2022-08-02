import torch
from torch.nn import Sequential
from typing import Union
from .rnn_layer import RNNLayer, SRNNLayer
from .input_layer import InputLayer, LinearStatic
from .output_layer import OutputLayer
from pyrates import NodeTemplate
import numpy as np


class ReservoirModel:

    def __init__(self, rnn_layer: Union[RNNLayer, SRNNLayer]):

        self.rnn_layer = rnn_layer
        self.input_layer = None
        self.output_layer = None

    @classmethod
    def from_yaml(cls, node: Union[str, NodeTemplate], weights: np.ndarray, source_var: str, target_var: str,
                  input_var_ext: str, output_var: str, input_var_net: str = None, spike_var: str = None, **kwargs):

        # initialize rnn layer
        if input_var_net is None and spike_var is None:
            rnn_layer = RNNLayer.from_yaml(node, weights, source_var, target_var, input_var_ext, output_var, **kwargs)
        elif input_var_net is None or spike_var is None:
            raise ValueError('To define a reservoir with a spiking neural network layer, please provide both the '
                             'name of the variable that spikes should be stored in (`input_var_net`) as well as the '
                             'name of the variable that is used to define spikes (`spike_var`).')
        else:
            rnn_layer = SRNNLayer.from_yaml(node, weights, source_var, target_var, input_var_ext, input_var_net,
                                            output_var, spike_var, **kwargs)

        # initialize model
        return cls(rnn_layer)

    def add_input_layer(self, n: int, m: int, weights: torch.Tensor = None, trainable: bool = False,
                        dtype: torch.dtype = torch.float64) -> InputLayer:

        # initialize input layer
        input_layer = InputLayer(n, m, weights, trainable=trainable, dtype=dtype)

        # add layer to model
        self.input_layer = input_layer

        # return layer
        return self.input_layer

    def add_output_layer(self, n: int, m: int, weights: torch.Tensor = None, trainable: bool = False,
                         transform: str = None, dtype: torch.dtype = torch.float64) -> OutputLayer:

        # initialize output layer
        output_layer = OutputLayer(n, m, weights, trainable=trainable, transform=transform, dtype=dtype)

        # add layer to model
        self.output_layer = output_layer

        # return layer
        return self.output_layer

    def remove_input_layer(self):
        self.input_layer = None

    def remove_output_layer(self):
        self.output_layer = None

    def train(self, inputs: np.ndarray, targets: np.ndarray, optimizer: str, optimizer_kwargs: dict, loss: str,
              loss_kwargs: dict, lr: float = 1e-3, device: str = 'cpu', **kwargs):

        # preparations
        ##############

        # transform inputs into tensors
        inp_tensor = torch.tensor(inputs)
        target_tensor = torch.tensor(targets)
        if inp_tensor.shape[0] != target_tensor.shape[0]:
            raise ValueError('Wrong dimensions of input and target output. Please make shure that `inputs` and '
                             '`targets` agree in the first dimension.')

        # prepare model
        if isinstance(self.input_layer.layer, LinearStatic):
            self.rnn_layer.detach()
        model = self._compile(device)

        # initialize optimizer
        if optimizer == 'sgd':
            opt = torch.optim.SGD
        elif optimizer == 'adam':
            opt = torch.optim.Adam
        elif optimizer == 'adagrad':
            opt = torch.optim.Adagrad
        elif optimizer == 'lbfgs':
            opt = torch.optim.LBFGS
        elif optimizer == 'rmsprop':
            opt = torch.optim.RMSprop
        else:
            raise ValueError('Invalid optimizer choice. Please see the documentation of the `ReservoirModel.run()` '
                             'method for valid options.')
        optimizer = opt(model.parameters(), lr=lr, **optimizer_kwargs)

        # initialize loss function
        if loss == 'mse':
            from torch.nn import MSELoss
            l = MSELoss
        elif loss == 'l1':
            from torch.nn import L1Loss
            l = L1Loss
        elif loss == 'nll':
            from torch.nn import NLLLoss
            l = NLLLoss
        elif loss == 'ce':
            from torch.nn import CrossEntropyLoss
            l = CrossEntropyLoss
        elif loss == 'kld':
            from torch.nn import KLDivLoss
            l = KLDivLoss
        else:
            raise ValueError('Invalid loss function choice. Please see the documentation of the `ReservoirModel.run()` '
                             'method for valid options.')
        loss = l(**loss_kwargs)

        # optimization
        ##############

        for step in range(inp_tensor.shape[0]):

            # forward pass
            pred = model(inp_tensor[step, :])

            # loss calculation
            error = loss(pred, target_tensor[step, :])

            # error backpropagation
            optimizer.zero_grad()
            error.backward()
            optimizer.step()

    def test(self):

        pass

    def run(self, inputs: np.ndarray, device: str = 'cpu', sampling_steps: int = 1):

        # transform input into tensor
        inp_tensor = torch.tensor(inputs)

        # initialize model from layers
        model = self._compile(device)

        # forward input through static network
        results = []
        with torch.no_grad():
            for step in range(inp_tensor.shape[0]):
                output = model(inp_tensor[step, :])
                if step % sampling_steps == 0:
                    results.append(output.detach().numpy())

        return results

    def _compile(self, device: str = 'cpu') -> Sequential:
        in_layer = (self.input_layer,) if self.input_layer is not None else ()
        out_layer = (self.output_layer,) if self.output_layer is not None else ()
        layers = in_layer + (self.rnn_layer,) + out_layer
        model = Sequential(*layers)
        model.to(device)
        return model
