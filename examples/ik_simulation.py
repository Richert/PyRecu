import numba as nb
nb.config.THREADING_LAYER = 'omp'
nb.set_num_threads(4)
import numpy as np
from pyrecu import RNN
import matplotlib.pyplot as plt
from pyrecu.neural_models import ik_ata, ik_spike_reset
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
v_reset = -100.0  # unit: mV
v_delta = 1.0  # unit: mV
d = 20.0
a = 0.03
b = -20.0
tau_s = 6.0
J = 15.0
g = 1.0
q = 0.0
E_r = 0.0

# define lorentzian of etas
spike_thresholds = v_t+v_delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))

# define inputs
T = 1100.0
cutoff = 100.0
dt = 1e-2
dts = 1e-1
inp = np.zeros((int(T/dt),)) + 80.0
inp[int(300/dt):int(600/dt)] -= 500.0

# collect parameters
func_args = (v_r, spike_thresholds, k, E_r, C, g, tau_s, b, a, d, q, J)
callback_args = (v_spike, v_reset)

# run the model
###############

# initialize model
u_init = np.zeros((2*N+1,))
u_init[:N] -= 60.0
model = RNN(N, 2*N+1, ik_ata, func_args, ik_spike_reset, callback_args, u_init=u_init)

# define outputs
outputs = {'s': {'idx': np.asarray([2*N]), 'avg': False}}

# perform simulation
res = model.run(T=T, dt=dt, dts=dts, outputs=outputs, inp=inp, cutoff=cutoff, solver='ralston')

# plot results
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(np.mean(res["s"], axis=1))
ax.set_xlabel('time')
ax.set_ylabel(r'$s(t)$')
plt.tight_layout()
plt.show()
