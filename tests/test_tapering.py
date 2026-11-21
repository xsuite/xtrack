import json
import pathlib
import pytest

import numpy as np
from scipy.constants import c as clight
from xobjects.test_helpers import (
    allow_kernel_compilation, skip_if_forbid_compile)

import xobjects as xo
import xtrack as xt


test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

CONFIGURATIONS_TO_TEST = [
    {'radiation_method': None, 'p0_correction': True, 'cavity_preserve_angle': False, 'beta_rtol': 2e-2, 'q_atol': 5e-4},
    {'radiation_method': 'kick_as_co', 'p0_correction': True, 'cavity_preserve_angle': False, 'beta_rtol': 2e-2, 'q_atol': 5e-4},
    {'radiation_method': 'kick_as_co', 'p0_correction': True, 'cavity_preserve_angle': True, 'beta_rtol': 1e-3, 'q_atol': 5e-4},
    {'radiation_method': 'scale_as_co', 'p0_correction': True, 'cavity_preserve_angle': True, 'beta_rtol': 2e-4, 'q_atol': 5e-4},
    {'radiation_method': 'full', 'p0_correction': False, 'cavity_preserve_angle': False, 'beta_rtol': 2e-2, 'q_atol': 5e-4},
    {'radiation_method': 'full', 'p0_correction': True, 'cavity_preserve_angle': False, 'beta_rtol': 2e-2, 'q_atol': 5e-4},
    {'radiation_method': 'full', 'p0_correction': True, 'cavity_preserve_angle': True, 'beta_rtol': 2e-4, 'q_atol': 5e-4},
]

@pytest.mark.parametrize('conf', CONFIGURATIONS_TO_TEST)
@allow_kernel_compilation(skip_when_forbid_compile=False)
def test_tapering_and_twiss_with_radiation(conf):

    if conf['radiation_method'] == 'scale_as_co':
        skip_if_forbid_compile()

    if conf['cavity_preserve_angle']:
        skip_if_forbid_compile()

    filename = test_data_folder / 'clic_dr/line_for_taper.json'


    with open(filename, 'r') as f:
        line = xt.Line.from_dict(json.load(f))

    # Use harmonic for one of the cavities
    t_rev = line.get_length() / clight
    harm = line['rf1'].frequency * t_rev
    line['rf1'].frequency = 0
    line['rf1'].harmonic = harm

    line.build_tracker()

    # Initial twiss (no radiation)
    line.configure_radiation(model=None)
    tw_no_rad = line.twiss(method='4d')

    assert tw_no_rad.radiation_method == None

    # Enable radiation
    line.configure_radiation(model='mean')
    # - Set cavity lags to compensate energy loss
    # - Taper magnet strengths
    line.compensate_radiation_energy_loss(delta0=0)

    print(f'Running test with conf: {conf}')

    line.config.XTRACK_CAVITY_PRESERVE_ANGLE = conf['cavity_preserve_angle']

    extra_kwargs = {}
    if conf['radiation_method'] == 'kick_as_co' and conf['cavity_preserve_angle']:
        extra_kwargs['matrix_stability_tol'] = 0.1

    if conf['radiation_method'] == 'scale_as_co':
        extra_kwargs['matrix_stability_tol'] = 0.1
        extra_kwargs['use_full_inverse'] = True

    print('Twiss with radiation')
    # Twiss(es) with radiation
    tw = line.twiss(radiation_method=conf['radiation_method'],
                    radiation_analysis=True, **extra_kwargs)
    print('Done')

    assert tw.radiation_method == (conf['radiation_method'] or 'kick_as_co')

    # Check twiss at_s
    print('Twiss at_s')
    i_ele = len(tw.s) // 3
    tws = line.twiss(radiation_method=conf['radiation_method'], radiation_analysis=True, **extra_kwargs).rows[i_ele]
    print('Done')

    line.config.XTRACK_CAVITY_PRESERVE_ANGLE = False

    if conf['p0_correction']:
        p0corr = 1 + tw.delta
    else:
        p0corr = 1

    # mask for taperable elements
    tt = line.get_table().rows[:-1] # remove endpoint
    mask_taperable = (tt.element_type == 'Multipole') | (tt.element_type == 'DipoleEdge')
    assert np.sum(mask_taperable) == 17420

    delta_taper = line.attr['delta_taper']
    xo.assert_allclose(delta_taper[mask_taperable],
        0.5*(tw.delta[:-1] + tw.delta[1:])[mask_taperable], rtol=0, atol=1e-5)

    xo.assert_allclose(tw.delta[0], 0, rtol=0, atol=1e-5)
    xo.assert_allclose(tw.delta[-1], 0, rtol=0, atol=1e-5)

    xo.assert_allclose(tw.qx, tw_no_rad.qx, rtol=0, atol=conf['q_atol'])
    xo.assert_allclose(tw.qy, tw_no_rad.qy, rtol=0, atol=conf['q_atol'])

    xo.assert_allclose(tw.dqx, tw_no_rad.dqx, rtol=0, atol=1.5e-2*tw.qx)
    xo.assert_allclose(tw.dqy, tw_no_rad.dqy, rtol=0, atol=1.5e-2*tw.qy)

    xo.assert_allclose(tw.dqx, tw_no_rad.dqx, rtol=0, atol=1.5e-2*tw.qx)
    xo.assert_allclose(tw.y, tw_no_rad.y, rtol=0, atol=1e-6)

    xo.assert_allclose(tw.betx*p0corr, tw_no_rad.betx, rtol=conf['beta_rtol'], atol=0)
    xo.assert_allclose(tw.bety*p0corr, tw_no_rad.bety, rtol=conf['beta_rtol'], atol=0)

    xo.assert_allclose(tw.dx, tw.dx, rtol=0.0, atol=0.1e-3)

    xo.assert_allclose(tw.dy, tw.dy, rtol=0.0, atol=0.1e-3)

    xo.assert_allclose(tw.s[i_ele], tws.s, rtol=0, atol=1e-6)
    xo.assert_allclose(tw.x[i_ele], tws.x, rtol=0, atol=1e-6)
    xo.assert_allclose(tw.y[i_ele], tws.y, rtol=0, atol=1e-6)
    xo.assert_allclose(tw.betx[i_ele], tws.betx, rtol=1e-3, atol=0)
    xo.assert_allclose(tw.bety[i_ele], tws.bety, rtol=1e-3, atol=0)

    eneloss = tw.energy_loss
    assert eneloss/line.particle_ref.energy0 > 0.01
    xo.assert_allclose(
        line['rf'].voltage*np.sin(line['rf'].phase + line['rf'].phase_taper + (line['rf'].lag + line['rf'].lag_taper)/180*np.pi),
        eneloss/4, rtol=3e-5)
    xo.assert_allclose(
        line['rf1'].voltage*np.sin(line['rf1'].phase + line['rf1'].phase_taper + (line['rf1'].lag + line['rf1'].lag_taper)/180*np.pi),
        eneloss/4, rtol=3e-5)
    xo.assert_allclose(
        line['rf2a'].voltage*np.sin(line['rf2a'].phase + line['rf2a'].phase_taper + (line['rf2a'].lag + line['rf2a'].lag_taper)/180*np.pi),
        eneloss/4*0.6, rtol=3e-5)
    xo.assert_allclose(
        line['rf2b'].voltage*np.sin(line['rf2b'].phase + line['rf2b'].phase_taper + (line['rf2b'].lag + line['rf2b'].lag_taper)/180*np.pi),
        eneloss/4*0.4, rtol=3e-5)
    xo.assert_allclose(
        line['rf3'].voltage*np.sin(line['rf3'].phase + line['rf3'].phase_taper + (line['rf3'].lag + line['rf3'].lag_taper)/180*np.pi),
        eneloss/4, rtol=3e-5)

def test_tapering_zero_mean():

    filename = test_data_folder / 'clic_dr/line_for_taper.json'
    with open(filename, 'r') as f:
        line = xt.Line.from_dict(json.load(f))

    line.build_tracker()

    line['rf3'].voltage = 0. # desymmetrize the rf
    line['rf'].voltage = 0.  # desymmetrize the rf

    line['rf1'].voltage *= 2
    line['rf2b'].voltage *= 2
    line['rf2a'].voltage *= 2

    # Use harmonic for one of the cavities
    t_rev = line.get_length() / clight
    harm = line['rf1'].frequency * t_rev
    line['rf1'].frequency = 0
    line['rf1'].harmonic = line['rf1'].frequency * t_rev

    line.particle_ref.p0c = 4e9  # eV

    line.configure_radiation(model=None)
    tw_no_rad = line.twiss(method='4d')

    ###############################################
    # Enable radiation and compensate energy loss #
    ###############################################

    line.configure_radiation(model='mean')

    # - Set cavity lags to compensate energy loss
    # - Taper magnet strengths to avoid optics and orbit distortions
    line.compensate_radiation_energy_loss(max_iter=100, delta0='zero_mean')

    ##############################
    # Twiss to check the results #
    ##############################

    tw = line.twiss(method='6d')

    tw.delta # contains the momentum deviation along the ring

    #!end-doc-part

    p0corr = 1 + tw.delta

    delta_ave = np.trapezoid(tw.delta, tw.s)/tw.s[-1]
    xo.assert_allclose(delta_ave, 0, rtol=0, atol=1e-6)

    xo.assert_allclose(tw.qx, tw_no_rad.qx, rtol=0, atol=5e-4)
    xo.assert_allclose(tw.qy, tw_no_rad.qy, rtol=0, atol=5e-4)

    xo.assert_allclose(tw.dqx, tw_no_rad.dqx, rtol=0, atol=1.5e-2*tw.qx)
    xo.assert_allclose(tw.dqy, tw_no_rad.dqy, rtol=0, atol=1.5e-2*tw.qy)

    xo.assert_allclose(tw.x, tw_no_rad.x, rtol=0, atol=1e-7)
    xo.assert_allclose(tw.y, tw_no_rad.y, rtol=0, atol=1e-7)

    xo.assert_allclose(tw.betx*p0corr, tw_no_rad.betx, rtol=2e-2, atol=0)
    xo.assert_allclose(tw.bety*p0corr, tw_no_rad.bety, rtol=2e-2, atol=0)

    xo.assert_allclose(tw.dx, tw.dx, rtol=0.0, atol=0.1e-3)

    xo.assert_allclose(tw.dy, tw.dy, rtol=0.0, atol=0.1e-3)


def _make_line_for_phase_compensation():
    line = xt.Line.from_json(test_data_folder / 'clic_dr/line_for_taper.json')
    line.particle_ref.p0c = 4e9
    line.build_tracker()
    line['rf3'].voltage = 0
    line['rf'].voltage = 0
    for name in ('rf1', 'rf2a', 'rf2b'):
        line[name].voltage *= 2
    line['rf1'].harmonic = line['rf1'].frequency * line.get_length() / clight
    line['rf1'].frequency = 0
    line['rf1'].lag = 13.
    line['rf2a'].phase = .23
    line['rf'].phase_taper = .17  # Inactive cavity settings should be preserved.
    line.configure_radiation(model='mean')
    return line


@pytest.mark.parametrize('delta0', [.001, 'zero_mean'])
@pytest.mark.parametrize('mode', [True, False, 'selected', 'empty'])
def test_phase_compensation_taper_selection(delta0, mode, monkeypatch):
    line = _make_line_for_phase_compensation()
    line.compensate_radiation_energy_loss(delta0=0, verbose=False)
    taper_setter = line.attr._cache['delta_taper'].multisetter
    taper_before = taper_setter.get_values().copy()
    taper_names = np.array(line.element_names)[line.attr._cache['delta_taper'].mask]
    if mode == 'selected':
        taper_magnets = list(taper_names[::4])
        selected = np.isin(taper_names, taper_magnets)
    elif mode == 'empty':
        taper_magnets = []
        selected = np.zeros(len(taper_names), dtype=bool)
    else:
        taper_magnets = mode
        selected = np.full(len(taper_names), mode, dtype=bool)

    attrs = ('_own_voltage', '_own_frequency', '_own_harmonic',
             '_own_lag', '_own_phase', '_own_lag_taper')
    setters = {name: line.attr._cache[name].multisetter for name in attrs}
    before = {name: setter.get_values().copy() for name, setter in setters.items()}
    config_before = dict(line.config)
    flags_before = dict(line.tracker.track_flags.flags)
    find_closed_orbit = line.find_closed_orbit
    searches = []
    co_guess = line.particle_ref.copy()
    guess_before = np.array([getattr(co_guess, key)[0]
                             for key in ('x', 'px', 'y', 'py', 'zeta', 'delta')])

    def check_settings(**kwargs):
        searches.append(kwargs['method'])
        if kwargs['method'] == '4d':
            assert kwargs['co_guess'] is co_guess
        for name, setter in setters.items():
            np.testing.assert_array_equal(setter.get_values(), before[name])
        np.testing.assert_array_equal(
            taper_setter.get_values()[~selected], taper_before[~selected])
        assert line.tracker.track_flags.XS_FLAG_SR_TAPER == (mode is True)
        return find_closed_orbit(**kwargs)

    monkeypatch.setattr(line, 'find_closed_orbit', check_settings)
    line.compensate_radiation_energy_loss(
        taper_magnets=taper_magnets, delta0=delta0,
        verbose=False, record_iterations=True, co_guess=co_guess)
    monkeypatch.setattr(line, 'find_closed_orbit', find_closed_orbit)
    np.testing.assert_array_equal(
        [getattr(co_guess, key)[0] for key in ('x', 'px', 'y', 'py', 'zeta', 'delta')],
        guess_before)
    assert searches[0] == '4d'
    if mode is True:
        assert searches == ['4d']  # Keep the automatic-tapering fast path.
    else:
        assert searches[1:] == ['5d'] * len(line._tapering_iterations)
    assert dict(line.config) == config_before
    assert dict(line.tracker.track_flags.flags) == flags_before
    for name, setter in setters.items():
        np.testing.assert_array_equal(setter.get_values(), before[name])
    np.testing.assert_array_equal(taper_setter.get_values()[~selected], taper_before[~selected])
    assert line['rf'].phase_taper == .17

    tw = line.twiss5d(zeta0=0, chrom=False)
    tolerance = 1e-6 if mode is True else 1e-9
    if delta0 == 'zero_mean':
        delta_mean = np.trapezoid(tw.delta, tw.s) / line.get_length()
        xo.assert_allclose(delta_mean, 0, rtol=0, atol=tolerance)
    else:
        xo.assert_allclose(tw.delta[0], delta0, rtol=0, atol=tolerance)
    for coord in ('x', 'px', 'y', 'py', 'delta'):
        values = getattr(tw, coord)
        xo.assert_allclose(values[-1], values[0], rtol=0, atol=2e-10)
    assert np.ptp(tw.delta) > 1e-3
    if np.any(selected):
        expected = (0.5 * (tw.delta[:-1] + tw.delta[1:]))[line.attr._cache['delta_taper'].mask]
        xo.assert_allclose(taper_setter.get_values()[selected], expected[selected],
                           rtol=0, atol=tolerance)
        assert np.max(np.abs(taper_setter.get_values()[selected] - taper_before[selected])) > 1e-5

    energy_change = np.diff(tw.ptau) * tw.p0c
    cavity_mask = line.attr._cache['_own_voltage'].mask
    loss = -energy_change[~cavity_mask].sum()
    voltage = before['_own_voltage']
    xo.assert_allclose(energy_change[cavity_mask], loss * voltage / voltage.sum(),
                       rtol=3e-5 if mode is True else 0, atol=1.)


@pytest.mark.parametrize('failure', ['iteration_limit', 'insufficient_voltage'])
def test_phase_compensation_restores_state_on_failure(failure):
    line = _make_line_for_phase_compensation()
    taper_setter = line.attr._cache['delta_taper'].multisetter
    taper_before = taper_setter.get_values().copy()
    taper_names = np.array(line.element_names)[line.attr._cache['delta_taper'].mask]
    kwargs = {'taper_magnets': list(taper_names[::4])}
    error, message = RuntimeError, 'Maximum number of iterations'
    if failure == 'iteration_limit':
        kwargs['max_iter'] = 1
    else:
        for name in ('rf1', 'rf2a', 'rf2b'):
            line[name].voltage = 1.
        error, message = ValueError, 'Insufficient RF voltage'
    phase_setter = line.attr._cache['_own_phase_taper'].multisetter
    phase_before = phase_setter.get_values().copy()
    config_before = dict(line.config)
    flags_before = dict(line.tracker.track_flags.flags)
    with pytest.raises(error, match=message):
        line.compensate_radiation_energy_loss(verbose=False, **kwargs)
    np.testing.assert_array_equal(phase_setter.get_values(), phase_before)
    np.testing.assert_array_equal(taper_setter.get_values(), taper_before)
    assert dict(line.config) == config_before
    assert dict(line.tracker.track_flags.flags) == flags_before


@pytest.mark.parametrize('selection', [['missing'], ['rf'], 'q1'])
def test_phase_compensation_invalid_selection(selection):
    line = xt.Line(
        elements={'q1': xt.Quadrupole(length=1, k1=.1), 'rf': xt.Cavity(voltage=1e6)},
        particle_ref=xt.Particles(p0c=4e9, mass0=xt.ELECTRON_MASS_EV))
    line.build_tracker()
    with pytest.raises(ValueError, match='taper'):
        line.compensate_radiation_energy_loss(taper_magnets=selection)
