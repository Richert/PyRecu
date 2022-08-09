import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from time import perf_counter


class Reservoir(nn.Module):

    def __init__(self, n: int, W: torch.Tensor, tau: float, v_th: float, v_r: float, dt: float = 1e-3):
        super().__init__()
        self.u = torch.zeros((n,))
        self.s = torch.zeros_like(self.u)
        self.W = W
        self.tau = tau
        self.v_th = v_th
        self.v_r = v_r
        self.dt = dt
        self.spikes = torch.zeros_like(self.u)

    def forward(self, x):
        spikes = self.u >= self.v_th
        self.spikes[spikes] = 1.0
        self.u[:] += self.dt * (self.u**2 + x + self.s)
        self.s[:] += self.dt * (-self.s/self.tau) + self.W @ self.spikes
        self.spike_reset(spikes)
        return self.s

    def parameters(self, recurse: bool = True):
        for param in []:
            yield param

    def spike_reset(self, spikes: torch.Tensor):
        self.u[spikes] = self.v_r
        self.spikes[:] = 0.0


m = 2
n = 100
W = torch.rand(n, n)
W /= torch.max(torch.real(torch.linalg.eigvals(W)))
in_layer = nn.Linear(m, n, bias=False)
reservoir = Reservoir(n, W, tau=0.5, v_th=100.0, v_r=-100.0, dt=1e-3)
out_layer = nn.Linear(n, 1)
model = nn.Sequential(in_layer, reservoir, out_layer, nn.Tanh())
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

steps = 50000
dur = 100
inputs = torch.zeros((steps, m))
targets = torch.zeros((steps, 1))
for step in range(int(steps/dur)):
    for k in range(m):
        inputs[step*dur:(step+1)*dur, k] = torch.rand(1)*10.0
    targets[step*dur:(step+1)*dur, 0] = 1.0 if inputs[step*dur, 0] > inputs[step*dur, 1] else -1.0

train_steps = 48000
errors = []
t0 = perf_counter()
for step in range(train_steps):
    pred = model(inputs[step, :])
    mse = loss(pred, targets[step])
    optimizer.zero_grad()
    mse.backward(retain_graph=True)
    optimizer.step()
    errors.append(mse.item())
t1 = perf_counter()
print(f'Finished optimization after {t1-t0} s.')
y = []
s = []
with torch.no_grad():
    for step in range(steps-train_steps):
        pred = model(inputs[train_steps+step, :])
        y.append(pred.numpy())
        s.append(np.mean(reservoir.s.numpy()))

fig, ax = plt.subplots(nrows=3, figsize=(10, 8))
ax[0].plot(targets[train_steps:])
ax[0].plot(y)
ax[1].plot(s)
ax[2].plot(errors)
plt.show()
