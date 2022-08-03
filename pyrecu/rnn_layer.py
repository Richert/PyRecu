from pyrates import NodeTemplate, CircuitTemplate
import torch
from torch.nn import Module
from typing import Callable, Union, Iterator
import numpy as np


class RNNLayer(Module):

    def __init__(self, rnn_func: Callable, rnn_args: tuple, input_ext: int, output: list, dt: float = 1e-3,
                 dtype: torch.dtype = torch.float64):

        super().__init__()
        self.y = torch.tensor(rnn_args[1], dtype=dtype)
        self.dy = torch.tensor(rnn_args[2], dtype=dtype)
        self.output = torch.tensor(output)
        self.dt = dt
        self.func = rnn_func
        self.args = rnn_args[3:]
        self._inp_ext = input_ext - 3

    @classmethod
    def from_yaml(cls, node: Union[str, NodeTemplate], weights: np.ndarray, source_var: str, target_var: str,
                  input_var: str, output_var: str, **kwargs):

        # extract keyword arguments for initialization
        dt = kwargs.pop('dt', 1e-3)

        # generate rnn template and function
        func, args, keys, template = cls._circuit_from_yaml(node, weights, source_var, target_var, step_size=dt,
                                                            **kwargs)

        # get variable indices
        input_idx = keys.index(input_var)
        out_indices, _ = template.get_variable_positions({'out': f"all/{output_var}"})

        return cls(func, args, input_idx, [val[0] for val in out_indices['out'].values()], dt=dt)

    def forward(self, x):
        self.args[self._inp_ext][:] = x
        self.dy = self.func(0, self.y, self.dy, *self.args)
        self.y = self.y + self.dt * self.dy
        return self.y[self.output]

    def parameters(self, recurse: bool = True) -> Iterator:
        for p in []:
            yield p

    def detach(self):

        self.y = self.y.detach()
        self.dy = self.dy.detach()
        self.args = tuple([arg.detach() for arg in self.args])

    @classmethod
    def _circuit_from_yaml(cls, node: Union[str, NodeTemplate], weights: np.ndarray, source_var: str, target_var: str,
                           **kwargs) -> tuple:

        # initialize base node template
        if type(node) is str:
            node = NodeTemplate.from_yaml(node)

        # initialize base circuit template
        n = weights.shape[0]
        nodes = {f'n{i}': node for i in range(n)}
        template = CircuitTemplate(name='reservoir', nodes=nodes)

        # add edges to network
        template.add_edges_from_matrix(source_var, target_var, nodes=list(nodes.keys()), weight=weights)

        # generate rnn function
        func, args, keys = template.get_run_func('rnn_layer', backend='torch', clear=False, **kwargs)

        return func, args, keys, template


class SRNNLayer(RNNLayer):

    def __init__(self, rnn_func: Callable, rnn_args: tuple, input_ext: int, input_net: int, output: list,
                 spike_var: list, spike_threshold: float = 1e2, spike_reset: float = -1e2, dt: float = 1e-3,
                 dtype: torch.dtype = torch.float64):

        super().__init__(rnn_func, rnn_args, input_ext, output, dt=dt, dtype=dtype)
        self._inp_net = input_net - 3
        self._thresh = spike_threshold
        self._reset = spike_reset
        self._var = torch.tensor(spike_var)

    @classmethod
    def from_yaml(cls, node: Union[str, NodeTemplate], weights: np.ndarray, source_var: str, target_var: str,
                  input_var_ext: str, input_var_net: str, output_var: str, spike_var: str, **kwargs):

        # extract keyword arguments for initialization
        dt = kwargs.pop('dt', 1e-3)
        kwargs_init = {}
        for key in ['spike_threshold', 'spike_reset']:
            if key in kwargs:
                kwargs_init[key] = kwargs.pop(key)

        # generate rnn template and function
        func, args, keys, template = cls._circuit_from_yaml(node, weights, source_var, target_var, step_size=dt,
                                                            **kwargs)

        # get variable indices
        input_ext_idx = keys.index(input_var_ext)
        input_net_idx = keys.index(input_var_net)
        var_indices, _ = template.get_variable_positions({'out': f"all/{output_var}", 'spike': f"all/{spike_var}"})

        return cls(func, args, input_ext_idx, input_net_idx,
                   [val[0] for val in var_indices['out'].values()],
                   [val[0] for val in var_indices['spike'].values()],
                   dt=dt, **kwargs_init)

    def forward(self, x):
        spikes = self.y[self._var] >= self._thresh
        self.args[self._inp_net][spikes] = 1.0
        self.args[self._inp_ext][:] = x
        self.dy = self.func(0, self.y, self.dy, *self.args)
        self.y = self.y + self.dt * self.dy
        self.reset(spikes)
        return self.y[self.output]

    def reset(self, spikes: torch.Tensor):
        self.y[self._var[spikes]] = self._reset
        self.args[self._inp_net][:] = 0.0
