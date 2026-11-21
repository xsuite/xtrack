import pathlib
from contextlib import nullcontext

import numpy as np
import pytest
from scipy.constants import c as clight

import xobjects as xo
import xtrack as xt


def _make_line():
    return xt.Line(
        elements=[xt.LineSegmentMap(
            length=100., qx=.31, qy=.32, qs=.02, bets=10.,
            betx=3., bety=5., dx=.7, dy=.2,
            x_ref=1e-3, y_ref=-2e-3, dqx=2., dqy=3.,
            energy_increment=-1e5)],
        element_names=['map'],
        particle_ref=xt.Particles(p0c=2e9, mass0=xt.ELECTRON_MASS_EV))


def _check_closure(line, tw):
    particle = tw.particle_on_co.copy()
    line.track(particle)
    for coord in ('x', 'px', 'y', 'py', 'delta'):
        xo.assert_allclose(getattr(particle, coord), getattr(tw, coord)[0],
                           rtol=0, atol=2e-10)
        xo.assert_allclose(getattr(tw, coord)[-1], getattr(tw, coord)[0],
                           rtol=0, atol=2e-10)
    xo.assert_allclose(particle.zeta, tw.zeta[-1],
                       rtol=0, atol=1e-12)


@pytest.mark.parametrize('constraint', [{}, {'zeta0': .002}, {'delta0': .001}])
def test_twiss_5d_constraints_and_optics(constraint):
    line = _make_line()
    tw = line.twiss5d(**constraint)
    assert tw.method == '5d'
    _check_closure(line, tw)
    zeta_slip = tw.zeta[-1] - tw.zeta[0]
    assert abs(zeta_slip) > 1e-4
    if 'delta0' in constraint:
        xo.assert_allclose(tw.delta[0], constraint['delta0'], rtol=0, atol=1e-12)
    else:
        xo.assert_allclose(tw.zeta[0], constraint.get('zeta0', 0), rtol=0, atol=1e-12)
    xo.assert_allclose(tw.dqx, 2., rtol=0, atol=1e-6)
    xo.assert_allclose(tw.dqy, 3., rtol=0, atol=1e-6)
    xo.assert_allclose(tw.qs, .02, rtol=0, atol=1e-8)

    # Explicitly subtracting the reference slip makes this an ordinary 6D
    # periodic map, with the same orbit and full linearized optics.
    reset_line = xt.Line(
        elements=[line['map'].copy(), xt.TimeDelay(shift_zeta=zeta_slip)],
        particle_ref=line.particle_ref.copy())
    tw6 = reset_line.twiss(chrom=False, co_guess=tw.particle_on_co)
    for coord in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
        xo.assert_allclose(getattr(tw6, coord)[0], getattr(tw, coord)[0],
                           rtol=0, atol=1e-10)
    for quantity in ('qx', 'qy', 'qs'):
        xo.assert_allclose(getattr(tw6, quantity), getattr(tw, quantity),
                           rtol=0, atol=1e-8)
    xo.assert_allclose(tw6.R_matrix, tw.R_matrix, rtol=1e-7, atol=1e-8)

    orbit = line.find_closed_orbit(method='5d', **constraint)
    xo.assert_allclose(orbit.delta, tw.delta[0], rtol=0, atol=1e-10)
    xo.assert_allclose(orbit.zeta, tw.zeta[0], rtol=0, atol=1e-10)
    only_orbit = line.twiss(method='5d', only_orbit=True, **constraint)
    _check_closure(line, only_orbit)


def test_twiss_5d_matches_synchronous_6d():
    line = _make_line()
    tw6 = line.twiss()
    for constraint in ({'zeta0': tw6.zeta[0]}, {'delta0': tw6.delta[0]}):
        tw5 = line.twiss(method='5d', **constraint)
        xo.assert_allclose(tw5.zeta[-1], tw5.zeta[0], rtol=0, atol=1e-10)
        for quantity in ('x', 'px', 'y', 'py', 'zeta', 'delta',
                         'betx', 'bety', 'qx', 'qy', 'qs', 'dqx', 'dqy'):
            xo.assert_allclose(getattr(tw5, quantity), getattr(tw6, quantity),
                               rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('method', [None, '4d', '5d', '6d'])
@pytest.mark.parametrize('provide_twiss', [False, True])
def test_orbit_correction_twiss_method(method, provide_twiss):
    line = xt.Line(
        elements=[
            xt.Multipole(knl=[2e-5, .02]),
            xt.LineSegmentMap(length=50, qx=.17, qy=.18, qs=.002,
                              bets=10, energy_increment=-5e4),
            xt.Multipole(knl=[-1e-5, -.02]),
            xt.LineSegmentMap(length=50, qx=.14, qy=.15, qs=.002,
                              bets=10, energy_increment=-5e4)],
        element_names=['q1', 'arc1', 'q2', 'arc2'],
        particle_ref=xt.Particles(p0c=2e9, mass0=xt.ELECTRON_MASS_EV))
    quads = ['q1', 'q2']
    tw_ref = line.twiss(method=method or '4d') if provide_twiss else None
    correction = line.correct_trajectory(
        method=method, twiss_table=tw_ref, run=False,
        monitor_names_x=quads, corrector_names_x=quads)
    assert correction.twiss_table.method == (method or '4d')
    before = correction.x_correction._get_tw_orbit()
    assert before.method == (method or '6d')
    assert np.max(np.abs(before['x', quads])) > 1e-6

    if method == '6d':
        kicks_before = correction.x_correction.get_kick_values().copy()
        for delta0 in (0., .001):
            with pytest.raises(ValueError, match='delta0.*6d'):
                correction.correct(planes='x', delta0=delta0)
            np.testing.assert_array_equal(
                correction.x_correction.get_kick_values(), kicks_before)

    correction.correct(planes='x', n_iter=3, verbose=False)
    after = correction.x_correction._get_tw_orbit()
    assert np.max(np.abs(after['x', quads])) < 1e-10
    xo.assert_allclose([line[name].knl[1] for name in quads], [.02, -.02],
                       rtol=0, atol=0)
    assert line.twiss_default == {}
    if method == '5d':
        _check_closure(line, after)
        assert abs(after.zeta[-1] - after.zeta[0]) > 1e-8
        off_momentum = correction.x_correction._get_tw_orbit(delta0=.001)
        assert off_momentum.method == '5d'
        xo.assert_allclose(off_momentum.delta[0], .001, rtol=0, atol=1e-12)


@pytest.mark.parametrize('constraint', [
    {'delta0': 0, 'zeta0': 0}, {'zeta_shift': 1e-3},
    {'freeze_longitudinal': True}, {'search_for_t_rev': True},
    {'num_turns': 2}, {'co_search_at': 'map'},
])
def test_twiss_5d_invalid_constraints(constraint):
    line = _make_line()
    line.build_tracker()
    expected_warning = (
        pytest.warns(FutureWarning, match='`freeze_longitudinal` keyword is deprecated')
        if 'freeze_longitudinal' in constraint else nullcontext())
    with expected_warning:
        with pytest.raises(ValueError, match='5d'):
            line.twiss(method='5d', **constraint)
    with pytest.raises(ValueError, match='5d'):
        line.find_closed_orbit(method='5d', **constraint)


def test_twiss_5d_with_radiation():
    filename = (pathlib.Path(__file__).parent
                / '../test_data/clic_dr/line_for_taper.json')
    line = xt.Line.from_json(filename)
    line.particle_ref.p0c = 4e9
    line.build_tracker()
    line['rf3'].voltage = 0
    line['rf'].voltage = 0
    for name in ('rf1', 'rf2a', 'rf2b'):
        line[name].voltage *= 2
    line['rf1'].harmonic = line['rf1'].frequency * line.get_length() / clight
    line['rf1'].frequency = 0
    line.configure_radiation(model='mean')
    line.compensate_radiation_energy_loss(verbose=False)

    tw6 = line.twiss(chrom=False)
    tw5 = line.twiss(method='5d', delta0=tw6.delta[0], chrom=False)
    xo.assert_allclose(tw5.zeta[-1], tw5.zeta[0], rtol=0, atol=1e-10)
    xo.assert_allclose(tw5.betx, tw6.betx, rtol=1e-6, atol=1e-8)
    xo.assert_allclose(tw5.qs, tw6.qs, rtol=1e-6, atol=0)

    # Use the same RF settings in the untapered lattice. The 5D orbit must
    # balance radiation loss without changing any magnet or cavity setting.
    taper_setter = line.attr._cache['delta_taper'].multisetter
    taper_setter.set_values(np.zeros_like(taper_setter.get_values()))
    rf_before = {name: line[name].to_dict()
                 for name in ('rf', 'rf1', 'rf2a', 'rf2b', 'rf3')}
    for radiation_method in ('kick_as_co', 'full'):
        for constraint in ({}, {'delta0': 0.}):
            tw = line.twiss(method='5d', radiation_method=radiation_method,
                            **constraint)
            _check_closure(line, tw)
            assert abs(tw.zeta[-1] - tw.zeta[0]) > 1e-8
            assert np.ptp(tw.delta) > 1e-3
            assert tw.qs > 0
            assert np.isfinite(tw.dqx)
            if constraint:
                xo.assert_allclose(tw.delta[0], 0, rtol=0, atol=1e-12)
            else:
                xo.assert_allclose(tw.zeta[0], 0, rtol=0, atol=1e-12)
    assert np.all(taper_setter.get_values() == 0)
    assert rf_before == {name: line[name].to_dict() for name in rf_before}
