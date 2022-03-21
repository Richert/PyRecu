import numpy as np
from pyrecu import ik2_ata, RNN
import matplotlib.pyplot as plt

# define parameters
###################
N = 10000
C = 50.0   # unit: pF
k = 1.15  # unit: None
v_r = -80.0  # unit: mV
v_t = -30.0  # unit: mV
v_spike = 55.0  # unit: mV
v_reset = -55.0  # unit: mV
v_delta = 2.0  # unit: mV
d = 377.0
a = 0.05
b = -20.0
tau_s = 10.0
J = 0.0
e_r = 0.0

# define lorentzian of etas
spike_thresholds = v_t+v_delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))

# define inputs
T = 80.0
dt = 1e-4
dts = 1e-2
inp = np.zeros((int(T/dt),))
inp[int(10/dt):int(30/dt)] = 100.0

# perform simulation
model = RNN(N, 3*N, ik2_ata, C=C, k=k, v_r=v_r, v_t=spike_thresholds, v_spike=v_spike, v_reset=v_reset, d=d, a=a, b=b,
            tau_s=tau_s, J=J, e_r=e_r)
res = model.run(T=T, dt=dt, dts=dts, outputs=(np.arange(3*N, 4*N),), inp=inp)

# plot results
fig, axes = plt.subplots(nrows=2, figsize=(12, 8))
axes[0].plot(np.mean(res[0], axis=1))
axes[1].eventplot(res[0], colors='k', lineoffsets=1.0, linelengths=1.0, linewidths=0.3)
plt.tight_layout()
plt.show()
