import numba as nb
nb.config.THREADING_LAYER = 'tbb'
import numpy as np
from pyrecu import ik2_ata, RNN
import matplotlib.pyplot as plt


# define parameters
###################

N = 10000
C = 15.2   # unit: pF
k = 1.0  # unit: None
v_r = -80.0  # unit: mV
v_t = -30.0  # unit: mV
v_spike = 200.0  # unit: mV
v_reset = -1000.0  # unit: mV
v_delta = 1.0  # unit: mV
d = 9.0
a = 0.03
b = -20.0
tau_s = 5.0
J = 1.0
g = 8.0
g_e = 6.0
e_r = -60.0

# define lorentzian of etas
spike_thresholds = v_t+v_delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))

# define inputs
T = 1100.0
dt = 1e-4
dts = 1e-2
inp = np.zeros((int(T/dt),)) + 200.0
inp[int(30/dt):int(50/dt)] += 100.0

# perform simulation
model = RNN(N, 3*N, ik2_ata, C=C, k=k, v_r=v_r, v_t=spike_thresholds, v_spike=v_spike, v_reset=v_reset, d=d, a=a, b=b,
            tau_s=tau_s, J=J, g=g, e_r=e_r, g_e=g_e)
res = model.run(T=T, dt=dt, dts=dts, outputs=(np.arange(3*N, 4*N),), inp=inp, cutoff=10.0)[0]

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 8))
axes[0].plot(np.mean(res, axis=1))
spiking_data = []
for i in np.random.choice(np.arange(res.shape[-1]), size=100, replace=False):
    spikes = np.argwhere(res[:, i] > 0).squeeze()
    spiking_data.append(spikes if spikes.shape else np.asarray([]))

axes[1].eventplot(spiking_data, colors='k', lineoffsets=1.0, linelengths=1.0, linewidths=0.3)
plt.tight_layout()
plt.show()
