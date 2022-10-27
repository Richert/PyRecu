import time
from copy import deepcopy
from scipy.stats import rv_discrete, bernoulli
from scipy.signal import correlation_lags, correlate
import numpy as np
from numba import njit, prange, objmode
from typing import Callable, Union
import sys


# helper functions
##################


def _get_unique_key(key: int, keys: list) -> int:
    if key in keys:
        return np.max(keys)+1
    return key


def _split_into_modules(A: np.ndarray, P: np.ndarray, modules: dict, m: int, iteration: int = 0, max_iter: int = 100,
                        min_nodes: int = 2) -> dict:
    modules_new = {}
    B = A - P
    for key, (nodes, Q_old) in deepcopy(modules).items():

        if Q_old > 0:

            # calculate difference matrix
            B_tmp = B[nodes, :][:, nodes]
            if iteration > 0:
                for n in range(B_tmp.shape[0]):
                    B_tmp[n, n] -= np.sum(B_tmp[n, :])

            # eigenvalue decomposition of difference matrix
            eigs, vectors = np.linalg.eig(B_tmp)
            idx_max = np.argmax(eigs)

            # sort nodes into two modules
            idx = vectors[:, idx_max] >= 0
            s = np.zeros_like(nodes)
            s[idx == True] = 1
            s[idx == False] = -1

            # calculate modularity
            Q_new = s.T @ B_tmp @ s

            # decide whether to do another split or not
            if Q_new > 0 and iteration < max_iter and np.sum(idx == True) >= min_nodes \
                    and np.sum(idx == False) >= min_nodes:

                modules_new[_get_unique_key(key, list(modules_new))] = (nodes[np.argwhere(s > 0)[:, 0]], Q_new)
                modules_new[_get_unique_key(key, list(modules_new))] = (nodes[np.argwhere(s < 0)[:, 0]], Q_new)
                modules_new = _split_into_modules(A, P, modules_new, m, iteration + 1)

            else:
                modules_new[_get_unique_key(key, list(modules_new))] = (nodes, -1.0)

        else:
            modules_new[_get_unique_key(key, list(modules_new))] = (nodes, -1.0)

    return modules_new


def _get_modules(A: np.ndarray, **kwargs) -> dict:

    # calculating the null matrix to compare against
    P = np.zeros_like(A)
    N = A.shape[0]
    m = int(np.sum(np.tril(A) > 0))
    for n1 in range(N):
        for n2 in range(N):
            k1 = np.sum(A[n1, :] > 0)
            k2 = np.sum(A[n2, :] > 0)
            P[n1, n2] = k1 * k2 / (2 * m)

    # find modules
    return _split_into_modules(A, P, {1: (np.arange(0, N), 1)}, m, 0, **kwargs)


def _wrap(idxs: np.ndarray, N: int) -> np.ndarray:
    idxs[idxs < 0] = N+idxs[idxs < 0]
    idxs[idxs >= N] = idxs[idxs >= N] - N
    return idxs


def _sequentiality_calculation(N: int, signals: np.ndarray, lags: np.ndarray, zero_lag: int,
                               corr: Callable = np.correlate) -> tuple:
    sym = 0
    asym = 0
    for n1 in range(N):
        for n2 in range(N):
            cc = corr(signals[n1], signals[n2])
            for l in lags:
                sym += (cc[zero_lag + l] - cc[zero_lag - l]) ** 2
                asym += (cc[zero_lag + l] + cc[zero_lag - l]) ** 2
        #print(f'\rProgress: {n1 * 100 / N} %')
    return sym, asym


def _dft(x: np.ndarray) -> np.ndarray:
    N = x.shape[0]
    dft = np.zeros(N, dtype=np.complex64)
    for i in range(N):
        series_element = 0
        for n in prange(N):
            series_element += x[n] * np.exp(-2j * np.pi * i * n * (1 / N))
        dft[i] = series_element
    return dft


def _idft(x: np.ndarray) -> np.ndarray:
    N = x.shape[0]
    idft = np.zeros(N, dtype=np.complex64)
    for i in range(N):
        series_element = 0
        for n in prange(N):
            series_element += x[n] * np.exp(2j * np.pi * i * n * (1 / N))
        idft[i] = series_element / N
    return idft


def cross_corr(N: int, signals: np.ndarray, mode: str = 'same', method: str = 'direct') -> np.ndarray:
    C = np.zeros((N, N))
    for n2 in range(N):
        s2 = signals[n2]
        for n1 in range(n2+1, N):
            s1 = signals[n1]
            C[n1, n2] = np.max(correlate(s1, s2, mode=mode, method=method))
            C[n2, n1] = C[n1, n2]
        print(f'\r      Progress: {n2 * 100 / N} %', end='', file=sys.stdout)
    return C


def cross_corr_njit(N: int, signals: np.ndarray, mode: str = 'same', method: str = 'direct'):
    C = np.zeros((N, N))
    for n2 in prange(N):
        s2 = signals[n2]
        for n1 in prange(n2+1, N):
            s1 = signals[n1]
            with objmode(c_tmp='double'):
                c_tmp = np.max(correlate(s1, s2, mode=mode, method=method))
            C[n1, n2] = c_tmp
            C[n2, n1] = c_tmp
    return C


def corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    m = y.shape[0]-n
    c = np.zeros((m+1,))
    for i in range(m+1):
        c_i = 0
        for k in prange(n):
            c_i += x[k]*y[k+i]
        c[i] = c_i
    return c


def corr_dft(x: np.ndarray, y: np.ndarray, dft: Callable, idft: Callable) -> np.ndarray:
    x_dft = dft(x)
    y_dft = dft(y)
    return np.real(idft(x_dft * np.conjugate(y_dft)))


# connectivity generation functions
###################################


def circular_connectivity(N: int, p: float, spatial_distribution: rv_discrete) -> np.ndarray:
    C = np.zeros((N, N))
    n_conns = int(N*p)
    for n in range(N):
        idxs = spatial_distribution.rvs(size=n_conns)
        signs = 1 * (bernoulli.rvs(p=0.5, loc=0, size=n_conns) > 0)
        signs[signs == 0] = -1
        conns = _wrap(n + idxs*signs, N)
        C[n, conns] = 1.0/n_conns
    return C


def random_connectivity(N: int, p: float, normalize: bool = True) -> np.ndarray:
    C = np.zeros((N, N))
    n_conns = int(N * p)
    positions = np.arange(start=0, stop=N)
    for n in range(N):
        idxs = np.random.permutation(positions)[:n_conns]
        C[n, idxs] = 1.0/n_conns if normalize else 1.0
    return C


# data analysis functions
#########################


def sequentiality(signals: np.ndarray, decorator: Callable = njit, **kwargs) -> float:
    """Estimates the sequentiality of the dynamics of a system using the method proposed by Bernacchia et al. (2022).

    :param signals: `N x T` matrix containing the dynamics of `N` units sampled at `T` time steps.
    :param  decorator: Decorator function that will be applied to the intrinsic function used for cross-correlation
        calculation.
    :param kwargs: Additional keyword arguments to be passed to the decorator function.
    :return: Estimate of the sequentiality of the system dynamcis.
    """

    # preparations
    N = signals.shape[0]
    m = signals.shape[1]
    lags = correlation_lags(m, m, mode='valid')
    lags_pos = lags[lags > 0]
    zero_lag = np.argwhere(lags == 0)[0]
    if decorator is None:
        c_func = np.correlate
        seq_func = _sequentiality_calculation
    else:
        c_func = decorator(corr, **kwargs)
        seq_func = decorator(_sequentiality_calculation, **kwargs)

    # sum up cross-correlations over neurons and lags
    print('Starting sequentiality approximation...')
    t0 = time.perf_counter()
    sym, asym = seq_func(N, signals, lags_pos, zero_lag, corr=c_func)
    t1 = time.perf_counter()
    print(f'Sequentiality approximation finished after {t1-t0} s.')

    # calculate sequentiality
    return np.sqrt(sym/asym)


def modularity(signals: np.ndarray, threshold: float = 0.1, min_connections: int = 2, min_nodes: int = 2,
               max_iter: int = 100, cross_corr_method: str = 'direct', decorator: Union[Callable, None] = njit,
               **kwargs) -> tuple:
    """Calculates the modularity of a system of interconnected units, by creating an adjacency matrix from the maximum
    cross-correlation between all units, thresholding it, and using the Newman (2006) community detection method.

    :param signals: N x T matrix with N units and T samples of the dynamics of the units.
    :param threshold: Value in the half-open interval `(0,1]` that indicates which fraction of all pair-wise
        correlations between the system units should be kept to create the adjacency matrix.
    :param min_connections: Indicates how many incoming connections a unit should at least have to still be considered
        part of the network.
    :param min_nodes: Minimum number of nodes that a module should contain.
    :param max_iter: Maximum number of successive splits into submodules..
    :param cross_corr_method: Can be `direct` or `fft`, depending on whether the cross-correlation should be calculated
        in time or frequency space.
    :param decorator: Decorator function applied to the cross-correlation calculation function.
    :param kwargs: Additional keyword arguments for the decorator function.
    :return: Tuple with 3 entries: (1) The dictionary containing the module structure, (2) the adjacency matrix used to
    identify the module structure, (3) the set of nodes that survived thresholding etc. and are part of the adjacency
        matrix.
    """

    # preparations
    N = signals.shape[0]
    if decorator is None:
        cc_func = cross_corr
    elif decorator is njit:
        cc_func = decorator(cross_corr_njit, **kwargs)
    else:
        cc_func = decorator(cross_corr, **kwargs)

    # calculate correlation matrix
    print('1. Calculating the correlation matrix...')
    t0 = time.perf_counter()
    C = cc_func(N, signals, method=cross_corr_method, mode='same')
    t1 = time.perf_counter()
    print(f'\n        ...finished after {t1-t0} s.')

    # preprocess correlation matrix
    print('2. Turning correlation matrix into adjacency graph...')
    t0 = time.perf_counter()
    C_abs = np.abs(C)
    theta = np.sort(C_abs, axis=None)[int(N**2*(1-threshold))]
    idx = np.argwhere(C_abs > theta)
    A = np.zeros_like(C)
    A[idx[:, 0], idx[:, 1]] = 1.0
    connected_nodes = np.sum(A, axis=1) >= min_connections
    A1 = A[connected_nodes, :][:, connected_nodes]
    t1 = time.perf_counter()
    print(f'        ...finished after {t1-t0} s.')

    # calculate modularity
    print('3. Estimating the modularity of the adjacency graph...')
    t0 = time.perf_counter()
    modules = _get_modules(A1, min_nodes=min_nodes, max_iter=max_iter)
    t1 = time.perf_counter()
    print(f'        ...finished after {t1-t0} s.')

    print(fr'Result: Community finding algorithm revealed an optimal split into $m = {len(modules.keys())}$ modules.')
    return modules, A1, np.argwhere(connected_nodes).squeeze()


def sort_via_modules(A: np.ndarray, modules: dict) -> np.ndarray:
    """Sorts adjacancy matrices such as the one generated by `modularity()` according to the module dictionary returned
    by `modularity()`.

    :param A: Adjacancy matrix (N x N)
    :param modules: Dictionary containing module indices as keys and node lists in the value tuple.
    :return: N x N sorted matrix containing the module keys as entries.
    """

    C = np.zeros_like(A)
    node_order = []

    # collect node order and create new matrix that contains the module indices as entries
    for module, (nodes_tmp, _) in modules.items():
        node_order.extend(list(nodes_tmp))
        idx = np.ix_(nodes_tmp, np.arange(0, A.shape[1]))
        idx2 = A[idx] > 0
        C_tmp = np.zeros(shape=idx2.shape, dtype=np.int32)
        C_tmp[idx2] = module
        C[idx] = C_tmp

    # sort the new matrix according to the collected node order
    C1 = np.zeros_like(C)
    for i, n in enumerate(node_order):
        C1[i, :] = C[n, :]
    C2 = np.zeros_like(C)
    for i, n in enumerate(node_order):
        C2[:, i] = C1[:, n]

    return C2
