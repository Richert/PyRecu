import sys
import time

from scipy.stats import rv_discrete, bernoulli
from scipy.signal import correlate, correlation_lags
import numpy as np
from numba import njit, prange
import typing as tp
from time import perf_counter


# class for RNN simulations
###########################

class RNN:

    def __init__(self, n_neurons: int, n_states: int, evolution_func: tp.Callable, *args, **kwargs):
        """Create an instance of a recurrent neural network that provides methods for numerical simulations.

        :param n_neurons: Number of neurons in the network.
        :param n_states: Length of the state vector of the network.
        :param evolution_func: Function that performs a single numerical integration step of the network of the form
            `f(u, n, I, *args, **kwargs)` with state vector `u`, number of neurons `n`, input `I` and
            user-supplied arguments.
        :param args: Positional arguments to be passed to the `evolution_func`.
        :param kwargs: Keyword arguments to be passed to the `evolution_func`.
        """

        self.N = n_neurons
        self.u = kwargs.pop('u_init', np.zeros((n_states,)))
        self.du = np.zeros_like(self.u)
        self.t = kwargs.pop('t_init', 0.0)
        self.net_update = evolution_func
        self.func_kwargs = kwargs
        self.func_args = args

    def run(self, T: float, dt: float, dts: float, outputs: dict, inp: np.ndarray = None, W_in: np.ndarray = None,
            t_init: float = 0.0, cutoff: float = 0.0, verbose: bool = True, decorator: tp.Optional[tp.Callable] = njit,
            disp_interval: float = None, **decorator_kwargs) -> dict:
        """Solve the initial value problem for the network.

        :param T: Upper time integration limit of the initial value problem.
        :param dt: Integration step-size.
        :param dts: Sampling step-size.
        :param outputs: Outputs which should be recorded. Key-value pairs where keys are arbitrary strings, and values
            are dictionaries with 2 key-value pairs: `idx` is used to provide a list/array of indices to the state
            variable vector and `avg` is used to provide a boolean to indicate whether the mean over these indices
            should be computed or not.
        :param inp: Array with inputs defined over time. If a 1D array, each neuron receives the same input. If a 2D
        array, `W_in` has to be provided as well to map inputs to neurons in the network.
        :param W_in: 2D array mapping input channels in `inp` to neurons in the network.
        :param t_init: Time at which to start solving the intial value problem.
        :param cutoff: Initial integration time that will be disregarded for all state recordings.
        :param verbose: If true, updates about the simulation status will be displayed.
        :param decorator: Decorator function that will be applied to the `evolution_func` and other `RNN` intrinsic
            functions that will be called multiple times throughout the integration process.
        :param disp_interval: Step-size at which the simulation progress will be displayed.
        :param decorator_kwargs: Optional keyword arguments to the `decorator` function.
        :return: Key-value pairs where the keys are the user-supplied keys of `outputs` and the values are state
            recordings in the form of `np.ndarray` objects.
        """

        # initializations
        steps = int(np.round(T / dt))
        sampling_steps = int(np.round(dts / dt))
        store_steps = int(np.round((T - cutoff) / dts))
        start_step = steps - store_steps * sampling_steps
        self.t += t_init
        self.func_kwargs['dt'] = dt

        # define recording variables
        state_records, state_buffers, state_averaging, state_indices, keys = [], [], [], [], []
        for key, out in outputs.items():
            if out["avg"]:
                x = np.zeros((store_steps, 1))
                x_buffer = np.zeros((1, sampling_steps))
            else:
                n_x = len(out['idx'])
                x = np.zeros((store_steps, n_x))
                x_buffer = np.zeros((n_x, sampling_steps))
            state_records.append(x)
            state_buffers.append(x_buffer)
            state_averaging.append(out["avg"])
            state_indices.append(np.asarray(out["idx"]))
            keys.append(key)

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

        # apply function decorator to all provided functions
        if decorator is None:
            decorator = lambda f, **kw: f
        ufunc = decorator(self.net_update, **decorator_kwargs)
        recfunc = decorator(self._record_njit if decorator == njit else self._record, **decorator_kwargs)
        if get_input == self._get_input and 'parallel' in decorator_kwargs:
            decorator_kwargs.pop('parallel')
        infunc = decorator(get_input, **decorator_kwargs)
        args = []
        for arg in self.func_args:
            if callable(arg):
                args.append(decorator(arg, **decorator_kwargs))
            else:
                args.append(arg)
        kwargs = {}
        for key, arg in self.func_kwargs.items():
            if callable(arg):
                kwargs[key] = decorator(arg, **decorator_kwargs)
            else:
                kwargs[key] = arg

        # retrieve simulation variables from object and dicts
        u, N = self.u, self.N
        recs, buffs, avgs, idxs = tuple(state_records), tuple(state_buffers), tuple(state_averaging), \
                                  tuple(state_indices)

        # final simulation variable initializations
        disp_interval = sampling_steps * 100 if disp_interval is None else int(np.round(disp_interval/dt))
        sample = 0

        # start simulation
        if verbose:
            print('Starting simulation.')
        t0 = perf_counter()
        for step in range(steps):

            # call user-supplied update function
            u = ufunc(u, N, infunc(step, *in_args), *args, **kwargs)

            # record states
            sample = recfunc(u, recs, buffs, avgs, idxs, sample, step, sampling_steps, start_step)

            # print simulation progress
            if verbose and step % disp_interval == 0:
                print(f'\r    Simulation progress: {step*100/steps} %', end='', file=sys.stdout)

        # final recording step
        if sample < store_steps:
            recfunc(u, recs, buffs, avgs, idxs, sample, sampling_steps, sampling_steps, 0)

        # summarize simulation process
        t1 = perf_counter()
        if verbose:
            print('')
            print(f'Finished simulation after {t1 - t0} s.')

        return {key: res for key, res in zip(keys, state_records)}

    @staticmethod
    def _project_input(idx: int, inp: np.ndarray, w: np.ndarray):
        return w @ inp[:, idx]

    @staticmethod
    def _get_input(idx: int, inp: np.ndarray):
        return inp[idx]

    @staticmethod
    def _record(u: np.ndarray, results: tuple, state_buffers: tuple, average: tuple, indices: tuple, sample: int,
                step: int, sampling_steps: int, start_step: int) -> int:
        buffer_step = step % sampling_steps
        store_results = step > start_step and buffer_step == 0
        for res, buffer, avg, idx in zip(results, state_buffers, average, indices):
            if avg:
                buffer[0, buffer_step] = np.mean(u[idx])
            else:
                buffer[:, buffer_step] = u[idx]
            if store_results:
                res[sample, :] = np.mean(buffer, 1)
        if store_results:
            sample += 1
        return sample

    @staticmethod
    def _record_njit(u: np.ndarray, results: tuple, state_buffers: tuple, average: tuple, indices: tuple, sample: int,
                     step: int, sampling_steps: int, start_step: int) -> int:
        buffer_step = step % sampling_steps
        store_results = step > start_step and buffer_step == 0
        for res, buffer, avg, idx in zip(results, state_buffers, average, indices):
            if avg:
                buffer[0, buffer_step] = u[idx].mean()
            else:
                buffer[:, buffer_step] = u[idx]
            if store_results:
                for i in prange(buffer.shape[0]):
                    res[sample, i] = buffer[i, :].mean()
        if store_results:
            sample += 1
        return sample


# data analysis functions
#########################


def wrap(idxs: np.ndarray, N: int) -> np.ndarray:
    idxs[idxs < 0] = N+idxs[idxs < 0]
    idxs[idxs >= N] = idxs[idxs >= N] - N
    return idxs


def circular_connectivity(N: int, p: float, spatial_distribution: rv_discrete) -> np.ndarray:
    C = np.zeros((N, N))
    n_conns = int(N*p)
    for n in range(N):
        idxs = spatial_distribution.rvs(size=n_conns)
        signs = 1 * (bernoulli.rvs(p=0.5, loc=0, size=n_conns) > 0)
        signs[signs == 0] = -1
        conns = wrap(n + idxs*signs, N)
        C[n, conns] = 1.0/n_conns
    return C


def sequentiality(signals: np.ndarray, **kwargs):

    # preparations
    N = signals.shape[0]
    m = signals.shape[1]
    mode = kwargs.pop('mode', 'full')
    if mode == 'valid':
        raise ValueError('Please choose correlation mode to be either `full` or `same`, since `valid` will does not '
                         'allow to evaluate the cross-correlation at multiple lags.')
    lags = correlation_lags(m, m, mode=mode)
    lags_pos = lags[lags > 0]
    zero_lag = np.argwhere(lags == 0)[0]

    # sum up cross-correlations over neurons and lags
    sym = 0
    asym = 0
    print('Starting sequentiality approximation...')
    t0 = time.perf_counter()
    for n1 in range(N):
        for n2 in range(N):
            cc = correlate(signals[n1], signals[n2], mode=mode, **kwargs)
            for l in lags_pos:
                sym += (cc[zero_lag+l]-cc[zero_lag-l])**2
                asym += (cc[zero_lag+l]+cc[zero_lag-l])**2
        print(f'\rProgress: {n1*100/N} %', end='', file=sys.stdout)
    t1 = time.perf_counter()
    print(f'Sequentiality approximation finished after {t1-t0} s.')

    # calculate sequentiality
    return np.sqrt(sym/asym)
