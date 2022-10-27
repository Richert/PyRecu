import numba as nb
nb.config.THREADING_LAYER = 'omp'
import numpy as np
from pyrecu.util import cross_corr_njit
import time

# config
njit_kwargs = {'fastmath': True, 'parallel': True}
nb.set_num_threads(8)
mode = 'same'
method = 'direct'

# define inputs
n = 1000
m = 1000
signal = np.random.randn(n, m)

# numba-wrap functions
cross_corr_njit = nb.njit(cross_corr_njit, **njit_kwargs)
cross_corr_njit(n, signal[:, :100], mode=mode, method=method)

# time function definition
def time_func(f, reps, *args):
    t0 = time.perf_counter()
    for _ in range(reps):
        f(*args)
    t1 = time.perf_counter()
    return (t1-t0)/reps


# time function calls
reps = 10
t_direct = time_func(cross_corr_njit, reps, *(n, signal, mode, method))
t_numba = time_func(cross_corr_njit, reps, *(n, signal, mode, method))
print(f'Direct cross-correlation estimation took {t_direct} s on average.')
print(f'Numba-accelerated cross-correlation estimation took {t_numba} s on average.')
