import numpy as np

import xtrack as xt


def test_bend_kick_bend_noise_10000_vs_1_slice():
    length = 14
    angle = 2 * np.pi / 1000
    knl = [0, 0, 0]
    x_test = 1e-4

    env = xt.Environment()
    line = env.new_line(components=[
        env.new(
            'b', 'Bend', length=length, angle=angle, knl=knl,
            model='bend-kick-bend', integrator='yoshida4',
            num_multipole_kicks=1,
        )
    ])
    line.set_particle_ref('positron', energy0=1e9)

    x_by_slices = []
    for n_slices in [1, 10000]:
        line.configure_bend_model(
            edge='full', core='bend-kick-bend',
            num_multipole_kicks=7 * n_slices)
        tw = line.twiss(betx=1, bety=1, x=x_test)
        x_by_slices.append(tw.x[-1])

    error = abs(x_by_slices[1] - x_by_slices[0])
    assert error < 1e-16
