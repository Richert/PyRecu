import numpy as np
from numba import njit


@njit
def qif(u: np.ndarray, N: int, inp: np.ndarray, C: np.ndarray, etas: np.ndarray, J: float, tau: float, tau_s: float,
        v_th: float, dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
    background excitabilities."""

    # extract state variables from u
    v, x = u[:N], u[N:2*N]

    # calculate network input
    spikes = v > v_th
    rates = spikes / dt
    s = rates @ C

    # calculate state vector updates
    v += dt * (v**2 + etas + inp + J*x*tau)/tau
    x += dt * (s[0, :] - x/tau_s)

    # reset membrane potential
    v[spikes] = -v[spikes]

    # store updated state variables
    u[:N] = v
    u[N:2*N] = x

    # store network spiking
    u[2*N:] = rates

    return u


@njit
def qif_sfa(u: np.ndarray, N: int, inp: np.ndarray, C: np.ndarray, etas: np.ndarray, J: float, tau: float, alpha: float,
            tau_a: float, tau_s: float, v_th: float, dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
    background excitabilities and mono-exponential spike-frequency-adaptation."""

    # extract state variables from u
    v, a, x = u[:N], u[N:2*N], u[2*N:3*N]

    # calculate network input
    spikes = v > v_th
    rates = spikes / dt
    s = rates @ C

    # calculate state vector updates
    v += dt * (v**2 + etas + inp + J*x*tau - a)/tau
    a += dt * (alpha*rates - a/tau_a)
    x += dt * (s[0, :] - x/tau_s)

    # reset membrane potential
    v[spikes] = -v[spikes]

    # store updated state variables
    u[:N] = v
    u[N:2*N] = a
    u[2*N:3*N] = x

    # store network spiking
    u[3*N:] = rates

    return u


@njit
def qif_ata(u: np.ndarray, N: int, inp: np.ndarray, etas: np.ndarray, J: float, tau: float, tau_s: float,
            v_th: float, dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
    background excitabilities."""

    # extract state variables from u
    v, x = u[:N], u[N:2*N]

    # calculate network input
    spikes = v > v_th
    rates = spikes / dt

    # calculate state vector updates
    v += dt * (v**2 + etas + inp + J*x*tau)/tau
    x += dt * (np.mean(rates) - x/tau_s)

    # reset membrane potential
    v[spikes] = -v[spikes]

    # store updated state variables
    u[:N] = v
    u[N:2*N] = x

    # store network spiking
    u[2*N:] = rates

    return u


@njit
def qif_sfa_ata(u: np.ndarray, N: int, inp: np.ndarray, etas: np.ndarray, J: float, tau: float, alpha: float,
                tau_a: float, tau_s: float, v_th: float, dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
    background excitabilities and mono-exponential spike-frequency-adaptation."""

    # extract state variables from u
    v, a, x = u[:N], u[N:2*N], u[2*N:3*N]

    # calculate network input
    spikes = v > v_th
    rates = spikes / dt

    # calculate state vector updates
    v += dt * (v**2 + etas + inp + J*x*tau - a) / tau
    a += dt * (alpha*rates - a / tau_a)
    x += dt * (np.mean(rates) - x / tau_s)

    # reset membrane potential
    v[spikes] = -v[spikes]

    # store updated state variables
    u[:N] = v
    u[N:2*N] = a
    u[2*N:3*N] = x

    # store network spiking
    u[3*N:] = rates

    return u
