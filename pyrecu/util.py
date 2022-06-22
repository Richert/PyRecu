import time
from copy import deepcopy
from scipy.stats import rv_discrete, bernoulli
from scipy.signal import correlate, correlation_lags
import numpy as np
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


# data analysis functions
#########################


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


def sequentiality(signals: np.ndarray, **kwargs):

    # preparations
    N = signals.shape[0]
    m = signals.shape[1]
    mode = kwargs.pop('mode', 'full')
    if mode == 'valid':
        raise ValueError('Please choose correlation mode to be either `full` or `same`, since `valid` will does not '
                         'allow to evaluate the cross-correlation at multiple lags.')
    lags = correlation_lags(m, m, mode=mode)
    lags_pos = lags[lags > 0]
    zero_lag = np.argwhere(lags == 0)[0]

    # sum up cross-correlations over neurons and lags
    sym = 0
    asym = 0
    print('Starting sequentiality approximation...')
    t0 = time.perf_counter()
    for n1 in range(N):
        for n2 in range(N):
            cc = correlate(signals[n1], signals[n2], mode=mode, **kwargs)
            for l in lags_pos:
                sym += (cc[zero_lag+l]-cc[zero_lag-l])**2
                asym += (cc[zero_lag+l]+cc[zero_lag-l])**2
        print(f'\rProgress: {n1*100/N} %', end='', file=sys.stdout)
    t1 = time.perf_counter()
    print(f'Sequentiality approximation finished after {t1-t0} s.')

    # calculate sequentiality
    return np.sqrt(sym/asym)


def modularity(signals: np.ndarray, threshold: float = 0.1, min_connections: int = 2, **kwargs) -> tuple:

    # preparations
    N = signals.shape[0]
    mode = kwargs.pop('mode', 'full')
    method = kwargs.pop('method', 'auto')

    # calculate correlation matrix
    C = np.zeros((N, N))
    print('1. Calculating the correlation matrix...')
    for n1 in range(N):
        for n2 in range(N):
            if n1 != n2:
                C[n1, n2] = np.max(correlate(signals[n1], signals[n2], mode=mode, method=method))
        print(f'\r      Progress: {n1 * 100 / N} %', end='', file=sys.stdout)
    print('')

    # preprocess correlation matrix
    print('2. Turning correlation matrix into adjacency graph...')
    idx = np.argwhere(C > threshold)
    A = np.zeros_like(C)
    A[idx[:, 0], idx[:, 1]] = 1.0
    connected_nodes = np.sum(A, axis=1) >= min_connections
    A1 = A[connected_nodes, :][:, connected_nodes]
    print('        ...finished.')

    # calculate modularity
    print('3. Estimating the modularity of the adjacency graph...')
    modules = _get_modules(A1, **kwargs)
    print('        ...finished.')

    print(fr'Result: Community finding algorithm revealed an optimal split into $m = {len(modules.keys())}$ modules.')
    return modules, A1, np.argwhere(connected_nodes)
