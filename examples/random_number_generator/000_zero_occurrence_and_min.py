import xtrack as xt
import numpy as np

rng = xt.RandomUniform()

n_gen = 0
min_val = 2.
min_non_zero = 2
n_zeros = 0
while True:
    xx = np.squeeze(rng.generate(n_seeds=1, n_samples=1e8))

    min_val = min(min_val, xx.min())
    min_non_zero = min(min_non_zero, xx[xx!=0].min())
    n_zeros += len(xx[xx==0])
    n_gen += len(xx)

    print(f'Generated {n_gen:.1e} numbers, '
          f'min: {min_val}, '
          f'min non zero: {min_non_zero}, '
          f'n zeros: {n_zeros}')
