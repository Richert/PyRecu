import numpy as np
from pyrecu import ik_ata, RNN
import matplotlib.pyplot as plt

# define parameters
N = 10000
Delta = 20.0
tau = 20.0
eta = 2000.0
alpha = 100.0
e_r = 1.0
b = 0.2
tau_a = 50.0
k = 2.0
tau_s = 10.0
J = 0.0

# define lorentzian of etas
etas = eta+Delta*np.tan((np.pi/2)*(2.*np.arange(1, N+1)-N-1)/(N+1))

# define inputs
T = 80.0
dt = 1e-3
dts = 1e-2
inp = np.zeros((int(T/dt),))
inp[int(10/dt):int(30/dt)] = 100.0

# perform simulation
model = RNN(N, 3*N, ik_ata, tau=tau, etas=etas, alpha=alpha, e_r=e_r, b=b, tau_a=tau_a, k=k, tau_s=tau_s, J=J,
            v_th=200.0)
res = model.run(T=T, dt=dt, dts=dts, outputs=(np.arange(0, N),), inp=inp)

# plot results
plt.plot(np.mean(res[0], axis=1))
plt.show()
