import numpy as np
from numba import njit


@njit
def ik(u: np.ndarray, N: int, inp: np.ndarray, C: np.ndarray, etas: np.ndarray, J: float, tau: float, alpha: float,
       e_r: float, tau_s: float, b: float, tau_a: float, k: float, v_th: float, dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
    background excitabilities."""

    # extract state variables from u
    v, w, x = u[:N], u[N:2*N], u[2*N:3*N]

    # calculate network input
    spikes = v > v_th
    rates = spikes / dt
    s = rates @ C

    # calculate state vector updates
    v += dt * (v**2 + alpha*v + etas + inp + J*x*tau*(e_r - v) - w)/tau
    w += dt * ((b*v - w)/tau_a + k*rates)
    x += dt * (s[0, :] - x/tau_s)

    # reset membrane potential
    v[spikes] = -v[spikes]

    # store updated state variables
    u[:N] = v
    u[N:2*N] = w
    u[2*N:3*N] = x

    # store network spiking
    u[3*N:] = rates

    return u


@njit
def ik_ata(u: np.ndarray, N: int, inp: np.ndarray, etas: np.ndarray, J: float, tau: float, alpha: float, e_r: float,
           tau_s: float, b: float, tau_a: float, k: float, v_th: float, dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
    background excitabilities."""

    # extract state variables from u
    v, w, x = u[:N], u[N:2*N], u[2*N:3*N]

    # calculate network input
    spikes = v > v_th
    rates = spikes / dt

    # calculate state vector updates
    v += dt * (v**2 + alpha*v + etas + inp + J*x*tau*(e_r - v) - w)/tau
    w += dt * ((b*v - w)/tau_a + k*rates)
    x += dt * (np.mean(rates) - x/tau_s)

    # reset membrane potential
    v[spikes] = -v[spikes]

    # store updated state variables
    u[:N] = v
    u[N:2*N] = w
    u[2*N:3*N] = x

    # store network spiking
    u[3*N:] = rates

    return u