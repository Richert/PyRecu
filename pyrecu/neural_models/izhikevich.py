import numpy as np


def ik_nodim(y: np.ndarray, N: int, inp: np.ndarray, C: np.ndarray, etas: np.ndarray, J: float, g: float, tau: float,
             alpha: float, E_r: float, tau_s: float, b: float, tau_a: float, k: float, v_spike: float, v_reset: float,
             dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of coupled Izhikevich neurons of the dimensionless form
     with heterogeneous background excitabilities."""

    # extract state variables from u
    v, u, s = y[:N], y[N:2*N], y[2*N:]

    # calculate network input
    spikes = v >= v_th
    rates = spikes / dt
    s_in = s @ C

    # calculate state vector updates
    dv = (v**2 + alpha*v + etas + inp + g*s_in*tau*(E_r - v) - u)/tau
    du = (b*v - u)/tau_a + k*rates
    ds = J*rates - s/tau_s

    # update state variables
    v_new = v + dt * dv
    u_new = u + dt * du
    s_new = s + dt * ds

    # reset membrane potential and apply spike frequency adaptation
    v_new[spikes] = v_reset
    u_new[spikes] += d

    # store updated state variables
    y[:N] = v_new
    y[N:2 * N] = u_new
    y[2 * N:] = s_new


def ik_nodim_ata(y: np.ndarray, N: int, inp: np.ndarray, eta: np.ndarray, J: float, g: float, tau: float, alpha: float,
                 E_r: float, tau_s: float, b: float, a: float, d: float, v_spike: float, v_reset: float,
                 dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled Izhikevich neurons of the dimensionless form
     with heterogeneous background excitabilities."""

    # extract state variables from u
    v, u, s = y[:N], y[N:2*N], y[2*N:]

    # calculate network input
    spikes = v >= v_spike
    rates = spikes / dt
    s_in = np.mean(s)

    # calculate state vector updates
    dv = (v**2 - alpha*v + eta + inp + g*s_in*tau*(E_r - v) - u)/tau
    du = a*(b*v - u)
    ds = J*rates - s/tau_s

    # update state variables
    v_new = v + dt * dv
    u_new = u + dt * du
    s_new = s + dt * ds

    # reset membrane potential and apply spike frequency adaptation
    v_new[spikes] = v_reset
    u_new[spikes] += d

    # store updated state variables
    y[:N] = v_new
    y[N:2 * N] = u_new
    y[2 * N:] = s_new

    return y


def ik(y: np.ndarray, N: int, inp: np.ndarray, W: np.ndarray, v_r: float, v_t: np.ndarray, k: float, E_r: float,
       C: float, J: float, g: float, tau_s: float, b: float, a: float, d: float, v_spike: float, v_reset: float,
       q: float, dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities."""

    # extract state variables from u
    v, u, s = y[:N], y[N:2*N], y[2*N:]

    # calculate network input
    spikes = v >= v_spike
    rates = spikes / dt
    s_in = s @ W

    # calculate state vector updates
    dv = (k*(v**2 - (v_r+v_t)*v + v_r*v_t) + inp + g*s_in*(E_r - v) + q*(np.mean(v)-v) - u)/C
    du = a*(b*(v-v_r) - u)
    ds = J*rates - s/tau_s

    # update state variables
    v_new = v + dt * dv
    u_new = u + dt * du
    s_new = s + dt * ds

    # reset membrane potential and apply spike frequency adaptation
    v_new[spikes] = v_reset
    u_new[spikes] += d

    # store updated state variables
    y[:N] = v_new
    y[N:2 * N] = u_new
    y[2 * N:] = s_new

    return y


def ik_ata(y: np.ndarray, N: int, inp: np.ndarray, v_r: float, v_t: np.ndarray, k: float, E_r: float, C: float,
           J: float, g: float, tau_s: float, b: float, a: float, d: float, v_spike: float, v_reset: float,
           q: float, dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities."""

    # extract state variables from u
    v, u, s = y[:N], y[N:2*N], y[2*N:]

    # calculate network input
    spikes = v >= v_spike
    rates = spikes / dt
    s_in = s.mean()

    # calculate vector field of the system
    dv = (k*(v**2 - (v_r+v_t)*v + v_r*v_t) + inp + g*s_in*(E_r - v) + q*(np.mean(v)-v) - u)/C
    du = a*(b*(v-v_r) - u)
    ds = J*rates - s/tau_s

    # update state variables
    v_new = v + dt * dv
    u_new = u + dt * du
    s_new = s + dt * ds

    # reset membrane potential and apply spike frequency adaptation
    v_new[spikes] = v_reset
    u_new[spikes] += d

    # store updated state variables
    y[:N] = v_new
    y[N:2*N] = u_new
    y[2*N:] = s_new

    return y
