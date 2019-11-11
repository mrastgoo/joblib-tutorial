import time

import numpy as np
from scipy.linalg import svd

from joblib import Memory


def func(X, n_components=2):
    U, s, Vh = svd(X)
    return s[:n_components]


X = np.random.randn(1000, 5000)

start = time.time()
func(X)
stop = time.time()
print('Function without caching - Elapsed time: {:.4f} s'.format(stop - start))

memory = Memory(location='cachedir', verbose=0)
func_cached = memory.cache(func)

# Instead of using the func_cached we could also use the decorator 
# @memory.cache
# def func(X, n_components=2):
#     U, s, Vh = svd(X)
#     return s[:n_components]

# In this case func is performing like func_cache


start = time.time()
func_cached(X)
stop = time.time()
print('Computing the results - Elapsed time: {:.4f} s'.format(stop - start))

start = time.time()
func_cached(X)
stop = time.time()
print('Loading the resuluts without computing again, Elapsed time: {:.4f} s'.format(stop - start))

X = np.random.randn(1000, 5000)

start = time.time()
func_cached(X)
stop = time.time()
print('The input has changes so it computes again - Elapsed time: {:.4f} s'.format(stop - start))
