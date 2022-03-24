import numpy as np
from pyrecu import ik_ata, RNN
import matplotlib.pyplot as plt

# define parameters
###################

N = 10000
tau = 1.0
alpha = 0.6215
eta = 0.1
Delta = 0.02
v_spike = 500.0  # unit: mV
v_reset = -500.0  # unit: mV
d = 0.02
a = 0.0077
b = -0.01
tau_s = 2.6
J = 1.2308
g = 2.6
e_r = 1.0

# define lorentzian of etas
etas = eta+Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))

# define inputs
T = 80.0
dt = 1e-4
dts = 1e-2
inp = np.zeros((int(T/dt),))
inp[int(20/dt):int(40/dt)] = 0.5

# perform simulation
model = RNN(N, 3*N, ik_ata, tau=tau, eta=etas, alpha=alpha, v_spike=v_spike, v_reset=v_reset, d=d, a=a, b=b,
            tau_s=tau_s, J=J, g=g, e_r=e_r)
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
