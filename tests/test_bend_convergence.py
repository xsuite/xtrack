import numpy as np
import pytest

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


@pytest.mark.parametrize(
    'model, converged_slice',
    [
        ('rot-kick-rot-low-order', 50),
        ('rot-kick-rot', 2),
        ('rot-kick-rot-high-order', 1),
    ],
)
def test_one_bend_rot_kick_rot_models_converged_at_expected_num_slices(
        model, converged_slice):
    length = 14
    angle = 2 * np.pi / 1200
    knl = [0, 0, 2.]
    x_test = 1e-4

    env = xt.Environment()
    line = env.new_line(components=[
        env.new(
            'b', 'Bend', length=length, angle=angle, knl=knl,
            model=model, integrator='yoshida4',
            num_multipole_kicks=1,
        )
    ])
    line.set_particle_ref('positron', energy0=1e9)

    line.configure_bend_model(
        edge='full', core=model, num_multipole_kicks=7 * converged_slice)
    tw_converged = line.twiss(betx=1, bety=1, x=x_test)

    line.configure_bend_model(
        edge='full', core=model, num_multipole_kicks=7 * 10000)
    tw_reference = line.twiss(betx=1, bety=1, x=x_test)

    error = abs(tw_converged.x[-1] - tw_reference.x[-1])
    assert error < 1e-14


@pytest.mark.parametrize(
    'model',
    [
        'rot-kick-rot-low-order',
        'rot-kick-rot',
    ],
)
def test_one_bend_on_zero_orbit_is_zero(model):
    length = 14
    angle = 2 * np.pi / 1200
    knl = [0, 0, 2.]

    env = xt.Environment()
    line = env.new_line(components=[
        env.new(
            'b', 'Bend', length=length, angle=angle, knl=knl,
            model=model, integrator='yoshida4',
            num_multipole_kicks=1,
        )
    ])
    line.set_particle_ref('positron', energy0=1e9)

    for n_slices in [10000, 20000]:
        line.configure_bend_model(
            edge='full', core=model, num_multipole_kicks=7 * n_slices)
        tw = line.twiss(betx=1, bety=1, x=0)
        assert abs(tw.x[-1]) < 1e-16
