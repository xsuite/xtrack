import xtrack as xt
import numpy as np
from time import time

# rng = xt.RandomUniform()
rng = xt.RandomUniformAccurate()


n_gen = 0
min_val = 2.
max_val = -2.
min_non_zero = 2
n_zeros = 0
t0 = time()
while True:
    xx = np.squeeze(rng.generate(n_seeds=1, n_samples=1e8))

    min_val = min(min_val, xx.min())
    max_val = max(max_val, xx.max())
    min_non_zero = min(min_non_zero, xx[xx!=0].min())
    n_zeros += len(xx[xx==0])
    n_gen += len(xx)

    print(f'T={time()-t0:.2f}s, '
          f'Generated {n_gen:.4e} numbers, '
          f'max: 1 - {1 - max_val:.2e}, '
          f'min: {min_val:.2e}, '
          f'min non zero: {min_non_zero:.2e}, '
          f'n zeros: {n_zeros}')
