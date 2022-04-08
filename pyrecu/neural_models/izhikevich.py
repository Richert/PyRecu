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


def ik_ei_ata(y: np.ndarray, N: int, inp: np.ndarray, ve_r: float, vi_r: float, ve_t: np.ndarray, vi_t: np.ndarray,
              ke: float, ki: float, Ce: float, Ci: float, ae: float, ai:float, be: float, bi: float, de: float,
              di:float, ve_spike: float, vi_spike: float, ve_reset: float, vi_reset: float, g_ampa: float,
              g_gaba: float, E_ampa: float, E_gaba: float, tau_ampa: float, tau_gaba: float, k_ee: float, k_ei: float,
              k_ie: float, k_ii: float, dt: float = 1e-4) -> np.ndarray:
    """Calculates right-hand side update of a network of all-to-all coupled Izhikevich neurons of the biophysical form
     with heterogeneous background excitabilities, split into an excitatory and an inhibitory population.
     """

    # preparatory calculations
    ##########################

    # extract state variables from u
    m = 4*N
    ve, ue, vi, ui = y[:N], y[N:2*N], y[2*N:3*N], y[3*N:m]
    se, si, re, ri = y[m], y[m+1], y[m+2], y[m+3]

    # extract inputs
    inp_e, inp_i = inp[0], inp[1]

    # calculate network firing rates
    spikes_e = ve >= ve_spike
    rates_e = np.mean(spikes_e / dt)
    spikes_i = vi >= vi_spike
    rates_i = np.mean(spikes_i / dt)

    # calculate vector field of the system
    ######################################

    # excitatory population
    d_ve = (ke*(ve**2 - (ve_r+ve_t)*ve + ve_r*ve_t) + inp_e + k_ee*g_ampa*se*(E_ampa-ve) + k_ie*g_gaba*si*(E_gaba-ve) - ue)/Ce
    d_ue = ae*(be*(ve-ve_r) - ue)
    d_se = rates_e - se/tau_ampa

    # inhibitory population
    d_vi = (ki * (vi**2 - (vi_r+vi_t)*vi + vi_r*vi_t) + inp_i + k_ei*g_ampa*se*(E_ampa-vi) + k_ii*g_gaba*si*(E_gaba-vi) - ui) / Ci
    d_ui = ai * (bi*(vi-vi_r) - ui)
    d_si = rates_i - si/tau_gaba

    # update state variables
    ########################

    # euler integration
    ve_new = ve + dt * d_ve
    ue_new = ue + dt * d_ue
    se_new = se + dt * d_se
    vi_new = vi + dt * d_vi
    ui_new = ui + dt * d_ui
    si_new = si + dt * d_si

    # reset membrane potential and apply spike frequency adaptation
    ve_new[spikes_e] = ve_reset
    ue_new[spikes_e] += de
    vi_new[spikes_i] = vi_reset
    ui_new[spikes_i] += di

    # store updated state variables
    y[:N] = ve_new
    y[N:2*N] = ue_new
    y[2*N:3*N] = vi_new
    y[3*N:m] = ui_new
    y[m] = se_new
    y[m+1] = si_new
    y[m+2] = rates_e
    y[m+3] = rates_i
    return y
