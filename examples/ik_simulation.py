import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu import RNN
from pyrecu.neural_models import ik_ata
import matplotlib.pyplot as plt
import pickle
plt.rcParams['backend'] = 'TkAgg'

# define parameters
###################

# model parameters
N = 10000
C = 100.0   # unit: pF
k = 0.7  # unit: None
v_r = -60.0  # unit: mV
v_t = -40.0  # unit: mV
v_spike = 50.0  # unit: mV
v_reset = -50.0  # unit: mV
v_delta = 0.01  # unit: mV
d = 0.0
a = 0.1
b = 0.0
tau_s = 6.0
J = 0.0
g = 0.0
q = 0.0
E_r = 0.0

# define lorentzian of etas
spike_thresholds = v_t+v_delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))

# define inputs
T = 210.0
cutoff = 0.0
dt = 1e-3
dts = 2e-1
inp = np.zeros((int(T/dt),)) + 1000.0
inp[int(600/dt):int(1600/dt)] -= 500.0

# run the model
###############

# initialize model
u_init = np.zeros((2*N+2,))
u_init[:N] -= 60.0
model = RNN(N, 2*N+2, ik_ata, C=C, k=k, v_r=v_r, v_t=spike_thresholds, v_spike=v_spike, v_reset=v_reset, d=d, a=a, b=b,
            tau_s=tau_s, J=J, g=g, E_r=E_r, q=q, u_init=u_init)

# define outputs
outputs = {'r': {'idx': np.asarray([2*N+1]), 'avg': False}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=inp, cutoff=cutoff, decorator=nb.njit)

# plot results
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(np.mean(res["r"], axis=1))
ax.set_xlabel('time')
ax.set_ylabel(r'$r(t)$')
plt.tight_layout()
plt.show()
