import numpy as np
from typing import Union, Callable


##########################
# vector-field functions #
##########################


def qif(t: Union[int, float], y: np.ndarray, N: int, rates: np.ndarray, infunc: Callable, inargs: tuple,
        etas: np.ndarray, tau: float, tau_s: float, J: float, W: np.ndarray) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
    background excitabilities."""

    # extract state variables from u
    m = 2*N
    v, s = y[:N], y[N:m]

    # retrieve extrinsic input at time t
    inp = infunc(t, *inargs)

    # calculate state vector updates
    dy[:N] = (v**2 + etas + inp + s*tau)/tau
    dy[N:m] = -s/tau_s + J*rates @ W

    return dy


def qif_sfa(t: Union[int, float], y: np.ndarray, N: int, rates: np.ndarray, infunc: Callable, inargs: tuple,
            etas: np.ndarray, tau: float, alpha: float, tau_a: float, tau_s: float, J: float, W: np.ndarray
            ) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
    background excitabilities and mono-exponential spike-frequency-adaptation."""

    # extract state variables from u
    m = 2*N
    v, a, s = y[:N], u[N:m], u[m:m+N]

    # retrieve extrinsic input at time t
    inp = infunc(t, *inargs)

    # calculate state vector updates
    dy[:N] = (v**2 + etas + inp + x*tau - a)/tau
    dy[N:m] = alpha*rates - a/tau_a
    dy[m:m+N] = -s/tau_s + J*rates @ W

    return dy


def qif_ata(t: Union[int, float], y: np.ndarray, N: int, rates: np.ndarray, infunc: Callable, inargs: tuple,
            etas: np.ndarray, tau: float, tau_s: float, J: float) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
    background excitabilities."""

    # extract state variables from u
    v, s = y[:N], y[N]

    # retrieve extrinsic input at time t
    inp = infunc(t, *inargs)

    # calculate state vector updates
    dy[:N] = (v ** 2 + etas + inp + s * tau) / tau
    dy[N] = -s / tau_s + J * np.mean(rates)

    return dy


def qif_sfa_ata(t: Union[int, float], y: np.ndarray, N: int, rates: np.ndarray, infunc: Callable, inargs: tuple,
                etas: np.ndarray, tau: float, alpha: float, tau_a: float, tau_s: float, J: float, W: np.ndarray
                ) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
    background excitabilities and mono-exponential spike-frequency-adaptation."""

    # extract state variables from u
    m = 2*N
    v, a, s = y[:N], u[N:m], u[m]

    # retrieve extrinsic input at time t
    inp = infunc(t, *inargs)

    # calculate state vector updates
    dy[:N] = (v**2 + etas + inp + x*tau - a)/tau
    dy[N:m] = alpha*rates - a/tau_a
    dy[m] = -s/tau_s + J*np.mean(rates)

    return dy


#########################
# spike reset functions #
#########################


def qif_spike_reset(y: np.ndarray, N: int, spike_threshold: float, spike_reset: float):

    # extract relevant state variables
    v = y[:N]

    # find spikes
    spikes = v >= spike_threshold

    # apply discontinuities to state variables
    v[spikes] = spike_reset

    # overwrite state vector
    y[:N] = v

    return y, spikes
