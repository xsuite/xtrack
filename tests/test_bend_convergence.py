import numpy as np
import pytest

import xobjects as xo
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

    line.configure_bend_model(
        edge='full', core='bend-kick-bend', num_multipole_kicks=7)
    tw_1_slice = line.twiss(betx=1, bety=1, x=x_test)

    line.configure_bend_model(
        edge='full', core='bend-kick-bend', num_multipole_kicks=7 * 10000)
    tw_10000_slices = line.twiss(betx=1, bety=1, x=x_test)

    xo.assert_allclose(tw_10000_slices.x[-1], tw_1_slice.x[-1],
                       rtol=0, atol=1e-16)


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

    line.configure_bend_model(
        edge='full', core='bend-kick-bend', num_multipole_kicks=7 * 10)
    tw_bend_kick_bend = line.twiss(betx=1, bety=1, x=x_test)

    xo.assert_allclose(tw_converged.x[-1], tw_reference.x[-1],
                       rtol=0, atol=1e-14)
    xo.assert_allclose(tw_converged.px[-1], tw_reference.px[-1],
                       rtol=0, atol=1e-14)
    xo.assert_allclose(tw_converged.y[-1], tw_reference.y[-1],
                       rtol=0, atol=1e-14)
    xo.assert_allclose(tw_converged.py[-1], tw_reference.py[-1],
                       rtol=0, atol=1e-14)

    xo.assert_allclose(tw_converged.x[-1], tw_bend_kick_bend.x[-1],
                       rtol=0, atol=1e-14)
    xo.assert_allclose(tw_converged.px[-1], tw_bend_kick_bend.px[-1],
                       rtol=0, atol=1e-14)
    xo.assert_allclose(tw_converged.y[-1], tw_bend_kick_bend.y[-1],
                       rtol=0, atol=1e-14)
    xo.assert_allclose(tw_converged.py[-1], tw_bend_kick_bend.py[-1],
                       rtol=0, atol=1e-14)


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

    line.configure_bend_model(
        edge='full', core=model, num_multipole_kicks=7 * 10000)
    tw_10000_slices = line.twiss(betx=1, bety=1, x=0)
    xo.assert_allclose(tw_10000_slices.x[-1], 0, rtol=0, atol=1e-16)

    line.configure_bend_model(
        edge='full', core=model, num_multipole_kicks=7 * 20000)
    tw_20000_slices = line.twiss(betx=1, bety=1, x=0)
    xo.assert_allclose(tw_20000_slices.x[-1], 0, rtol=0, atol=1e-16)
