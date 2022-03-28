import numpy as np
from numba import njit
import typing as tp


class RNN:

    def __init__(self, n_neurons: int, n_states: int, evolution_func: tp.Callable, *args, **kwargs):

        self.N = n_neurons
        self.u = kwargs.pop('u_init', np.zeros((n_states + n_neurons,)))
        self.du = np.zeros_like(self.u)
        self.t = kwargs.pop('t_init', 0.0)
        self.net_update = evolution_func
        self.func_kwargs = kwargs
        self.func_args = args
        self.state_records = []

    def run(self, T: float, dt: float, dts: float, outputs: tuple = None, inp: tp.Optional[np.ndarray] = None,
            W_in: np.ndarray = None, t_init: float = 0.0, cutoff: float = 0.0, verbose: bool = False):

        if not outputs:
            outputs = (np.arange(0, len(self.u)),)

        # initializations
        steps = int(np.round(T / dt))
        sampling_steps = int(np.round(dts / dt))
        store_steps = int(np.round((T - cutoff) / dts))
        start_step = steps - store_steps * sampling_steps
        self.t += t_init
        self.func_kwargs['dt'] = dt
        self.state_records = [np.zeros((store_steps, len(out))) for out in outputs]
        state_buffers = [np.zeros((len(out), sampling_steps)) for out in outputs]

        # retrieve recording variables
        ufunc, u, args, kwargs, results = self.net_update, self.u, self.func_args, self.func_kwargs, \
            self.state_records

        # define input projection function
        if inp is None:
            in_args = ()
            get_input = lambda s: 0.0
        elif W_in is None:
            in_args = (inp,)
            get_input = self._get_input
        else:
            in_args = (inp, W_in)
            get_input = self._project_input

        # integrate the system equations
        results = self._run(steps=steps, sampling_steps=sampling_steps, start_step=start_step, outputs=outputs,
                            state_buffers=state_buffers, results=results, ufunc=ufunc, u=u, N=self.N,
                            get_input=get_input, in_args=in_args, args=args, kwargs=kwargs)

        if verbose:
            print('Finished simulation. The state recordings are available under `state_records`.')
        return results

    @staticmethod
    def _run(steps: int, sampling_steps: int, start_step: int, outputs: tuple, state_buffers: list, results: list,
             ufunc: tp.Callable, u: np.ndarray, N: int, get_input: tp.Callable, in_args: tuple, args: tuple,
             kwargs: dict):

        sample = 0

        for step in range(steps):

            # call user-supplied update function
            u = ufunc(u, N, get_input(step, *in_args), *args, **kwargs)

            # store current network state
            buffer_step = step % sampling_steps
            store_results = step > start_step and buffer_step == 0
            for i, (out, buffer) in enumerate(zip(outputs, state_buffers)):
                buffer[:, buffer_step] = u[out]
                if store_results:
                    results[i][sample, :] = np.mean(buffer, axis=1)
            if store_results:
                sample += 1

        return results

    @staticmethod
    @njit(parallel=True)
    def _project_input(idx: int, inp: np.ndarray, w: np.ndarray, *args):
        return w @ inp[:, idx]

    @staticmethod
    @njit
    def _get_input(idx: int, inp: np.ndarray, *args):
        return inp[idx]
