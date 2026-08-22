from time import perf_counter

import numpy as np

import xpart as xp
import xtrack as xt

NUM_PART = 10_000
NUM_TURNS = 10
NEMITT_SCAN = [2.5e-6, 25e-6, 100e-6, 500e-6]

env = xt.load('sps_env.json')
sps_thin = env.lines['sps_thin']
sps_with_limits = env.lines['sps_with_limits']

for nemitt in NEMITT_SCAN:
    print(f'=> Testing nemitt = {nemitt}...')
    x_norm, px_norm = xp.generate_2D_gaussian(NUM_PART)
    y_norm, py_norm = xp.generate_2D_gaussian(NUM_PART)

    p0 = sps_thin.build_particles(
        x_norm=x_norm,
        px_norm=px_norm,
        y_norm=y_norm,
        py_norm=py_norm,
        nemitt_x=nemitt,
        nemitt_y=nemitt,
        zeta=0.0,
        delta=0.0,
    )

    p_thin = p0.copy()
    p_limits = p0.copy()

    print('Tracking with no limits... ', end='')
    t0 = perf_counter()
    sps_thin.track(p_thin, num_turns=NUM_TURNS)
    print(f'took {perf_counter() - t0:.3f} s')

    print('Tracking with limits... ', end='')
    t0 = perf_counter()
    sps_with_limits.track(p_limits, num_turns=NUM_TURNS)
    print(f'took {perf_counter() - t0:.3f} s, lost {np.sum(p_limits.state < 1)} particles')
