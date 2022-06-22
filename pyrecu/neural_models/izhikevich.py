import numpy as np
from typing import Union, Callable

##########################
# vector-field functions #
##########################


def ik_nodim(t: Union[int, float], y: np.ndarray, N: int, rates: np.ndarray, infunc: Callable, inargs: tuple,
             etas: np.ndarray, g: float, tau: float, alpha: float, E_r: float, tau_s: float, b: float, a: float,
             d: float, J: float, W: np.ndarray) -> np.ndarray:
    """Calculates right-hand side update of a network of coupled Izhikevich neurons of the dimensionless form
     with heterogeneous background excitabilities."""

    # extract state variables from y
    m = 2*N
    v, u, s = y[:N], y[N:m], y[n:]

    # retrieve extrinsic input at time t
    inp = infunc(t, *inargs)

    # calculate state vector updates
    dy[:N] = (v**2 + alpha*v + etas + inp + g*s*tau*(E_r - v) - u)/tau
    dy[N:m] = a*(b*v - u) + d*rates
    dy[m:] = -s/tau_s + J*rates @ W

    return dy


def ik_nodim_ata(t: Union[int, float], y: np.ndarray, N: int, rates: np.ndarray, infunc: Callable, inargs: tuple,
             etas: np.ndarray, g: float, tau: float, alpha: float, E_r: float, tau_s: float, b: float, a: float,
             d: float, J: float) -> np.ndarray:
    """Calculates right-hand side update of a network of coupled Izhikevich neurons of the dimensionless form
     with heterogeneous background excitabilities."""

    # extract state variables from y
    m = 2*N
    v, u, s = y[:N], y[N:m], y[n:]

    # retrieve extrinsic input at time t
    inp = infunc(t, *inargs)

    # calculate state vector updates
    dy[:N] = (v**2 + alpha*v + etas + inp + g*s*tau*(E_r - v) - u)/tau
    dy[N:m] = a*(b*v - u) + d*rates
    dy[m:] = -s/tau_s + J*np.mean(rates)

    return dy


def ik(t: Union[int, float], y: np.ndarray, N: int, rates: np.ndarray, infunc: Callable, inargs: tuple, v_r: float,
       v_t: np.ndarray, k: float, E_r: float, C: float, g: float, tau_s: float, b: float, a: float, d: float, q: float,
       J: float, W: np.ndarray) -> np.ndarray:
    """Calculates right-hand side update of a network of coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities."""

    dy = np.zeros_like(y)

    # extract state variables from u
    m = 2*N
    v, u, s = y[:N], y[N:m], y[m:]

    # retrieve extrinsic input at time t
    inp = infunc(t, *inargs)

    # calculate state vector updates
    dy[:N] = (k*(v**2 - (v_r+v_t)*v + v_r*v_t) + inp + g*s*(E_r - v) + q*(np.mean(v)-v) - u)/C
    dy[N:m] = a*(b*(v-v_r) - u) + d*rates
    dy[m:] = -s/tau_s + J*rates @ W

    return dy


def ik_ata(t: Union[int, float], y: np.ndarray, N: int, rates: np.ndarray, infunc: Callable, inargs: tuple, v_r: float,
           v_t: np.ndarray, k: float, E_r: float, C: float, g: float, tau_s: float, b: float, a: float, d: float,
           q: float, J: float) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities."""

    dy = np.zeros_like(y)

    # extract state variables from u
    m = 2*N
    v, u, s = y[:N], y[N:m], y[m]

    # retrieve extrinsic input at time t
    inp = infunc(t, *inargs)

    # calculate vector field of the system
    dy[:N] = (k*(v**2 - (v_r+v_t)*v + v_r*v_t) + inp + g*s*(E_r - v) + q*(np.mean(v)-v) - u)/C
    dy[N:m] = a*(b*(v-v_r) - u) + d*rates
    dy[m] = -s/tau_s + J*np.mean(rates)

    return dy


#########################
# spike reset functions #
#########################


def ik_spike_reset(y: np.ndarray, N: int, spike_threshold: float, spike_reset: float):

    # extract relevant state variables
    v = y[:N]

    # find spikes
    spikes = v >= spike_threshold

    # apply discontinuities to state variables
    v[spikes] = spike_reset

    # overwrite state vector
    y[:N] = v

    return y, spikes
