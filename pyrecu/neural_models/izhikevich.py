import numpy as np


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


def ik_ata(y: np.ndarray, N: int, inp: np.ndarray, eta: np.ndarray, J: float, g: float, tau: float, alpha: float,
           e_r: float, tau_s: float, b: float, a: float, d: float, v_spike: float, v_reset: float, dt: float = 1e-4
           ) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
    background excitabilities."""

    # extract state variables from u
    v, u, s = y[:N], y[N:2*N], y[2*N:3*N]

    # calculate network input
    spikes = v > v_spike
    rates = spikes / dt

    # calculate state vector updates
    v = v + dt * (v**2 - alpha*v + eta + inp + g*s*tau*(e_r - v) - u)/tau
    u = u + dt * (a*(b*v - u))
    s = s + dt * (np.mean(rates)*J - s/tau_s)

    # reset membrane potential
    v[spikes] = v_reset
    u[spikes] = u[spikes] + d

    # store updated state variables
    y[:N] = v
    y[N:2*N] = u
    y[2*N:3*N] = s

    # store network spiking
    y[3*N:] = rates

    return y


def ik2(y: np.ndarray, N: int, inp: np.ndarray, W: np.ndarray, v_r: float, v_t: np.ndarray, k: float, e_r: float,
        C: float, J: float, tau_s: float, b: float, a: float, d: float, v_spike: float, v_reset: float,
        dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
    background excitabilities."""

    # extract state variables from u
    v, u, x = y[:N], y[N:2*N], y[2*N:3*N]

    # calculate network input
    spikes = v > v_spike
    rates = spikes / dt
    s = rates @ W

    # calculate state vector updates
    v += dt * (k*(v**2 - (v_r+v_t)*v + v_r*v_t) + inp + J*x*(e_r - v) - u)/C
    u += dt * (a*(b*(v-v_r) - u))
    x += dt * (s[0, :] - x/tau_s)

    # reset membrane potential
    v[spikes] = v_reset
    u[spikes] += d

    # store updated state variables
    y[:N] = v
    y[N:2*N] = u
    y[2*N:3*N] = x

    # store network spiking
    y[3*N:] = rates

    return y


def ik2_ata(y: np.ndarray, N: int, inp: np.ndarray, v_r: float, v_t: np.ndarray, k: float, e_r: float, C: float,
            J: float, g: float, tau_s: float, b: float, a: float, d: float, v_spike: float, v_reset: float,
            g_e: float, dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled QIF neurons with heterogeneous
    background excitabilities."""

    # extract state variables from u
    v, u, s = y[:N], y[N:2*N], y[2*N:3*N]

    # calculate network input
    spikes = v >= v_spike
    rates = spikes / dt

    # calculate state vector updates
    v += dt * (k*(v**2 - (v_r+v_t)*v + v_r*v_t) + inp + g*s*(e_r - v) + g_e*(np.mean(v)-v) - u)/C
    u += dt * (a*(b*(v-v_r) - u))
    s += dt * (np.mean(rates)*J - s/tau_s)

    # reset membrane potential and apply spike frequency adaptation
    v[spikes] = v_reset
    u[spikes] += d

    # store updated state variables
    y[:N] = v
    y[N:2*N] = u
    y[2*N:3*N] = s

    # store network spiking
    y[3*N:] = rates

    return y
