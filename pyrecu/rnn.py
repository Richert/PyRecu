import sys
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
from numba import njit, prange
import typing as tp
from time import perf_counter


# class for RNN simulations
###########################

class RNN:

    def __init__(self, n_neurons: int, n_states: int, evolution_func: tp.Callable, evolution_args: tuple,
                 callback_func: tp.Callable, callback_args: tuple, u_init: tp.Optional[np.ndarray] = None):
        """Create an instance of a recurrent neural network that provides methods for numerical simulations.

        :param n_neurons: Number of neurons in the network.
        :param n_states: Length of the state vector of the network.
        :param evolution_func: Function that performs a single numerical integration step of the network of the form
            `f(t, u, du, n, g, g_args, *args)` with time or integration step `t`, state vector `u`,
            number of neurons `n`, extrinsic input function `g` and user-supplied arguments `g_args` and `args`.
        :param evolution_args: Positional arguments to be passed to the `evolution_func`.
        :param callback_func: Function that can perform additional manipulations of the state vector `u` of the form
            `f(u, n, *args)` with state vector `u`, number of neurons `n`, input `I` and
            user-supplied arguments `args`.
        :param callback_args: Keyword arguments to be passed to the `callback_func`.
        """

        self.N = n_neurons
        self.u = u_init if u_init is not None else np.zeros((n_states,))
        self.du = np.zeros_like(self.u)
        self.t = 0.0
        self.net_update = evolution_func
        self.func_args = evolution_args
        self.callback = callback_func
        self.callback_args = callback_args

    def run(self, T: float, dt: float, dts: float, outputs: dict, inp: np.ndarray = None, W_in: np.ndarray = None,
            t_init: float = 0.0, cutoff: float = 0.0, solver: str = 'euler', solver_kwargs: dict = None,
            interp: str = 'linear', verbose: bool = True, decorator: tp.Optional[tp.Callable] = njit,
            decorate_run_func: bool = True, disp_interval: float = None, **decorator_kwargs) -> dict:
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
        :param solver: Numerical solver method to use for integrating the system equations. Can be `euler`or `heun` for
            fixed step-size solvers, or `scipy` for adaptive step-size solvers.
        :param interp: Interpolation type used to interpolate extrinsic inputs when using the `scipy` solver methods.
        :param solver_kwargs: Additional keyword arguments to be passed to `scipy.integrate.solve_ivp` if
            `solver='scipy'`.
        :param verbose: If true, updates about the simulation status will be displayed.
        :param decorator: Decorator function that will be applied to the `evolution_func` and other `RNN` intrinsic
            functions that will be called multiple times throughout the integration process.
        :param decorate_run_func: If True, decorator will be applied to run function as well. Else, it will only be
        applied to internal functions.
        :param disp_interval: Step-size at which the simulation progress will be displayed.
        :param decorator_kwargs: Optional keyword arguments to the `decorator` function.
        :return: Key-value pairs where the keys are the user-supplied keys of `outputs` and the values are state
            recordings in the form of `np.ndarray` objects.
        """

        if solver_kwargs is None:
            solver_kwargs = dict()

        # initializations
        steps = int(np.round(T / dt))
        sampling_steps = int(np.round(dts / dt))
        store_steps = int(np.round((T - cutoff) / dts))
        start_step = steps - store_steps * sampling_steps
        self.t += t_init

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

        # define integration function
        if solver == 'euler':
            make_step = self._euler_step
        elif solver == 'heun':
            make_step = self._heun_step
        elif solver == 'midpoint':
            make_step = self._midpoint_step
        elif solver == 'ralston':
            make_step = self._ralston_step
        else:
            raise ValueError('Invalid solver choice. See documentation of the `run` method for valid solver options.')

        # define function generator
        if decorator is None:
            decorator = lambda f, **kw: f

        # apply function decorator to all relevant functions
        ufunc = decorator(self.net_update, **decorator_kwargs) if decorate_run_func else self.net_update
        recfunc = decorator(self._record_njit if decorator == njit else self._record, **decorator_kwargs)
        callback = decorator(self.callback, **decorator_kwargs)
        make_step = decorator(make_step, **decorator_kwargs)
        infunc, inargs = self._get_input_func(inp, W_in, solver, T, t_init, steps, decorator, **decorator_kwargs)

        # retrieve simulation variables from instance and dicts
        u, N, fargs, cbargs = self.u, self.N, self.func_args, self.callback_args
        recs, buffs, avgs, idxs = tuple(state_records), tuple(state_buffers), tuple(state_averaging), \
                                  tuple(state_indices)

        # final simulation variable initializations
        disp_interval = sampling_steps * 100 if disp_interval is None else int(np.round(disp_interval/dt))
        sample = 0

        # start simulation
        if verbose:
            print('Starting simulation.')
        t0 = perf_counter()
        self._integrate(u, N, dt, steps, ufunc, infunc, callback, recfunc, make_step, inargs, fargs, cbargs, recs,
                        buffs, avgs, idxs, sample, sampling_steps, start_step, verbose, disp_interval, store_steps,
                        **solver_kwargs)

        # summarize simulation process
        t1 = perf_counter()
        if verbose:
            print('')
            print(f'Finished simulation after {t1 - t0} s.')

        return {key: res for key, res in zip(keys, state_records)}

    def _get_input_func(self, inp: np.ndarray, W_in: np.ndarray, solver: str, T: float, t0: float, steps: int,
                        decorator: tp.Callable, **kwargs) -> tuple:

        # define input function and function arguments
        decorator_kwargs = kwargs.copy()
        if inp is None:
            in_args = ()
            get_input = lambda s: 0.0
        elif W_in is None:
            decorator_kwargs.pop('parallel', None)
            if solver in ['euler', 'heun']:
                if solver == 'heun':
                    inp_tmp = np.zeros((inp.shape[0] + 1,))
                    inp_tmp[:-1] = inp
                    inp_tmp[-1] = inp[-1]
                    inp = inp_tmp
                in_args = (inp,)
                get_input = self._get_input
            else:
                time = np.linspace(t0, T, steps)
                in_args = (time, inp)
                get_input = self._get_continous_input
        else:
            if solver in ['euler', 'heun']:
                if solver == 'heun':
                    inp_tmp = np.zeros((inp.shape[0], inp.shape[1] + 1,))
                    inp_tmp[:, :-1] = inp
                    inp_tmp[:, -1] = inp[:, -1]
                    inp = inp_tmp
                in_args = (inp, W_in)
                get_input = self._project_input
            else:
                time = np.linspace(t0, T, steps)
                in_args = (time, inp, W_in)
                get_input = self._project_continuous_input

        # try applying the function decorator
        try:
            infunc = decorator(get_input, **decorator_kwargs)
        except ValueError:
            infunc = get_input
        return infunc, in_args

    @staticmethod
    def _project_input(idx: int, inp: np.ndarray, w: np.ndarray):
        return w @ inp[:, idx]

    @staticmethod
    def _get_input(idx: int, inp: np.ndarray):
        return inp[idx]

    @staticmethod
    def _project_continuous_input(t: float, time: np.ndarray, inp: np.ndarray, w: np.ndarray):
        inp_interp = np.asarray([np.interp(t, time, inp[:, idx]) for idx in range(inp.shape[1])])
        w @ inp_interp

    @staticmethod
    def _get_continous_input(t: float, time: np.ndarray, inp: np.ndarray):
        return np.interp(t, time, inp)

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

    @staticmethod
    def _integrate(u: np.ndarray, N: int, dt: float, steps: int, ufunc: tp.Callable, infunc: tp.Callable,
                   callback: tp. Callable, recfunc: tp.Callable, integrate: tp.Callable, inargs: tuple, fargs: tuple,
                   cbargs: tuple, recs: tuple, buffs: tuple, avgs: tuple, idxs: tuple, sample: int, sampling_steps: int,
                   start_step: int, verbose: bool, disp_interval: int, store_steps: int, **kwargs) -> None:

        for step in range(steps):

            # integrate vector-field
            u = integrate(u, N, dt, step, ufunc, infunc, callback, inargs, fargs, cbargs)

            # record states
            sample = recfunc(u, recs, buffs, avgs, idxs, sample, step, sampling_steps, start_step)

            # print simulation progress
            if verbose and step % disp_interval == 0:
                print(f'\r    Simulation progress: {step * 100 / steps} %', end='', file=sys.stdout)

        # final recording step
        if sample < store_steps:
            recfunc(u, recs, buffs, avgs, idxs, sample, sampling_steps, sampling_steps, 0)

    @staticmethod
    def _euler_step(u: np.ndarray, N: int, dt: float, step: int, ufunc: tp.Callable, infunc: tp.Callable,
                    cbfunc: tp.Callable, inargs: tuple, fargs: tuple, cbargs: tuple) -> np.ndarray:

        # spike detection and reset
        u, spikes = cbfunc(u, N, *cbargs)

        # call user-supplied vector-field function
        du = ufunc(step, u, N, spikes/dt, infunc, inargs, *fargs)

        # integrate
        u += dt * du

        return u

    @staticmethod
    def _heun_step(u: np.ndarray, N: int, dt: float, step: int, ufunc: tp.Callable, infunc: tp.Callable,
                   cbfunc: tp.Callable, inargs: tuple, fargs: tuple, cbargs: tuple) -> np.ndarray:

        # spike detection and reset
        u, spikes = cbfunc(u, N, *cbargs)
        rates = spikes/dt

        # first integration
        du = ufunc(step, u, N, rates, infunc, inargs, *fargs)
        u2 = u + dt*du

        # second integration
        du2 = ufunc(step+1, u2, N, rates, infunc, inargs, *fargs)
        return u + dt*0.5*(du+du2)

    @staticmethod
    def _midpoint_step(u: np.ndarray, N: int, dt: float, step: int, ufunc: tp.Callable, infunc: tp.Callable,
                       cbfunc: tp.Callable, inargs: tuple, fargs: tuple, cbargs: tuple) -> np.ndarray:

        # spike detection and reset
        u, spikes = cbfunc(u, N, *cbargs)
        rates = spikes / dt
        t = step*dt

        # intermediate integration
        du = ufunc(t, u, N, rates, infunc, inargs, *fargs)
        u2 = u + dt*0.5*du

        # final integration
        du2 = ufunc(t+0.5*dt, u2, N, rates, infunc, inargs, *fargs)
        return u + dt*du2

    @staticmethod
    def _ralston_step(u: np.ndarray, N: int, dt: float, step: int, ufunc: tp.Callable, infunc: tp.Callable,
                      cbfunc: tp.Callable, inargs: tuple, fargs: tuple, cbargs: tuple) -> np.ndarray:

        # spike detection and reset
        u, spikes = cbfunc(u, N, *cbargs)
        rates = spikes / dt
        t = step*dt

        # intermediate integration
        du = ufunc(t, u, N, rates, infunc, inargs, *fargs)
        u2 = u + dt*du*2/3

        # final integration
        du2 = ufunc(t + dt*2/3, u2, N, rates, infunc, inargs, *fargs)
        return u + dt*(du*1/4 + du2*3/4)
