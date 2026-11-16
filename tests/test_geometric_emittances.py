import numpy as np
import pytest

import xtrack as xt
from xobjects.test_helpers import allow_kernel_compilation


@pytest.fixture
def line(temp_context_default_func):
    line = xt.Line(elements=[xt.LineSegmentMap(
        length=100, qx=0.28, qy=0.31, betx=3., bety=5.,
        longitudinal_mode='linear_fixed_qs', qs=0.01, bets=20.,
        det_xx=100., det_yy=150., det_xy=20.)], element_names=['ip'])
    line.particle_ref = xt.Particles(p0c=2e9)
    line.build_tracker()
    return line


def emittances(line):
    bg = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
    return dict(nemitt_x=2e-6, nemitt_y=3e-6), dict(gemitt_x=2e-6/bg, gemitt_y=3e-6/bg)


@pytest.mark.parametrize('open_twiss', [False, True])
def test_twiss_geometric_emittances(line, open_twiss):
    norm, geom = emittances(line)
    kwargs = dict(chrom=True, _keep_initial_particles=True)
    if open_twiss:
        kwargs.update(init=line.twiss().get_twiss_init('ip'))
    tn = line.twiss(**norm, **kwargs)
    tg = line.twiss(**geom, **kwargs)
    for key in ('W_matrix', 'betx', 'bety', 'dx', 'mux', 'muy'):
        np.testing.assert_allclose(tn[key], tg[key], rtol=1e-12, atol=1e-14)
    # Verify that the propagation probes, not only linear optics, are identical.
    for key in ('x', 'px', 'y', 'py'):
        np.testing.assert_allclose(getattr(tn._initial_particles, key),
                                   getattr(tg._initial_particles, key), rtol=1e-12, atol=1e-15)
    line.twiss_default.update(geom)
    np.testing.assert_allclose(line.twiss().betx, tg.betx)


@pytest.mark.parametrize('mode', ['polar', 'uniform_action_grid'])
@pytest.mark.parametrize('linear_rescale', [False, True])
@allow_kernel_compilation
def test_footprint_geometric_emittances(line, mode, linear_rescale):
    norm, geom = emittances(line)
    kwargs = dict(mode=mode, n_turns=64, n_fft=1024, keep_tracking_data=True,
                  freeze_longitudinal=True)
    kwargs.update(dict(n_r=2, n_theta=2) if mode == 'polar' else
                  dict(n_x_norm=2, n_y_norm=2))
    if linear_rescale:
        line.vars['detuning_scale'] = 1.
        line.element_refs['ip'].det_xx = 100. * line.vars['detuning_scale']
        kwargs['linear_rescale_on_knobs'] = xt.LinearRescale(
            knob_name='detuning_scale', v0=0.5, dv=0.1)
    fn = line.get_footprint(**norm, **kwargs)
    fg = line.get_footprint(**geom, **kwargs)
    for key in ('qx', 'qy', 'x_norm_2d', 'y_norm_2d'):
        np.testing.assert_allclose(getattr(fn, key), getattr(fg, key), rtol=1e-12)
    for key, value in geom.items():
        assert getattr(fn, key) == pytest.approx(value)
        assert getattr(fg, key) == pytest.approx(value)
    if mode == 'uniform_action_grid':
        for key in ('Jx_grid', 'Jy_grid', 'Jx_2d', 'Jy_2d'):
            np.testing.assert_allclose(getattr(fn, key), getattr(fg, key), rtol=1e-12)
        stability_kwargs = dict(n_points_stability_diagram=3,
                                n_points_interpolate=16, max_iter=2)
        sn = fn.get_stability_diagram(**stability_kwargs)
        sg = fg.get_stability_diagram(**stability_kwargs)
        assert np.all(np.isfinite(sn))
        np.testing.assert_allclose(sn, sg, rtol=1e-12, atol=1e-15)
    for key in ('x', 'px', 'y', 'py'):
        np.testing.assert_allclose(getattr(fn.tracking_data, key),
                                   getattr(fg.tracking_data, key), rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize('p0c', [2e9, 7e12])
@pytest.mark.parametrize('input_form', ['normalized', 'geometric', 'mixed'])
def test_footprint_geometric_action_grid(p0c, input_form):
    particle_ref = xt.Particles(p0c=p0c)
    bg = particle_ref.beta0[0] * particle_ref.gamma0[0]
    geom = dict(gemitt_x=2e-6/bg, gemitt_y=3e-6/bg)
    inputs = dict(normalized=dict(nemitt_x=2e-6, nemitt_y=3e-6),
                  geometric=geom,
                  mixed=dict(nemitt_x=2e-6, gemitt_y=geom['gemitt_y']))[input_form]
    fp = xt.Footprint(**inputs, particle_ref=particle_ref,
                      mode='uniform_action_grid', x_norm_range=(1, 3),
                      y_norm_range=(2, 4), n_x_norm=3, n_y_norm=4)
    np.testing.assert_allclose(fp.Jx_grid, geom['gemitt_x'] * np.array([0.5, 2.5, 4.5]),
                               rtol=1e-13, atol=0)
    np.testing.assert_allclose(fp.Jy_grid, geom['gemitt_y'] * np.array([2., 4., 6., 8.]),
                               rtol=1e-13, atol=0)


def test_footprint_emittances_without_reference_particle():
    geom = dict(gemitt_x=2e-6, gemitt_y=3e-6)
    fp = xt.Footprint(**geom, mode='uniform_action_grid')
    assert fp.gemitt_x == geom['gemitt_x']
    assert fp.gemitt_y == geom['gemitt_y']
    for inputs in (dict(nemitt_x=2e-6, nemitt_y=3e-6),
                   dict(gemitt_x=2e-6, nemitt_y=3e-6)):
        with pytest.raises(ValueError, match='particle_ref.*normalized.*geometric'):
            xt.Footprint(**inputs)


def test_detuning_geometric_emittances(line):
    norm, geom = emittances(line)
    dn = line.get_amplitude_detuning_coefficients(**norm)
    dg = line.get_amplitude_detuning_coefficients(**geom)
    for key in dn:
        assert dg[key] == pytest.approx(dn[key], rel=1e-7, abs=1e-5)
    defaults = line.get_amplitude_detuning_coefficients()
    explicit = line.get_amplitude_detuning_coefficients(nemitt_x=1e-6, nemitt_y=1e-6)
    assert defaults == explicit


def test_momentum_acceptance_geometric_emittances(line):
    norm, geom = emittances(line)
    kwargs = dict(elements=['ip'], n_turns=2, x_norm_offset=1., y_norm_offset=2.,
                  delta_step_size=0.01, delta_negative_limit=-0.02, delta_positive_limit=0.02)
    tn = line.get_local_momentum_acceptance(**norm, **kwargs)
    tg = line.get_local_momentum_acceptance(**geom, **kwargs)
    for key in ('delta_neg', 'delta_pos'):
        np.testing.assert_allclose(tn[key], tg[key])


@pytest.mark.parametrize('method', ['twiss', 'get_footprint',
                                   'get_amplitude_detuning_coefficients',
                                   'get_local_momentum_acceptance'])
@pytest.mark.parametrize('plane', ['x', 'y'])
def test_conflicting_emittances(line, method, plane):
    kwargs = dict(nemitt_x=2e-6, nemitt_y=3e-6)
    kwargs[f'gemitt_{plane}'] = 1e-6
    with pytest.raises(ValueError, match=f'both nemitt_{plane} and gemitt_{plane}'):
        getattr(line, method)(**kwargs)


@pytest.mark.parametrize('source', ['table', 'init'])
@pytest.mark.parametrize('plane', ['x', 'y', 'zeta'])
def test_geometric_normalized_coordinates(line, source, plane):
    tw = line.twiss()
    if source == 'init':
        tw = tw.get_twiss_init('ip')
    bg = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
    particles = line.build_particles(x_norm=[1., 2.], y_norm=[-1., 0.5],
                                    zeta_norm=[0.1, 0.2], gemitt_x=2e-6,
                                    gemitt_y=3e-6, gemitt_zeta=1e-4)
    kwargs = {f'gemitt_{plane}': 1e-6}
    ng = tw.get_normalized_coordinates(particles, **kwargs)
    nn = tw.get_normalized_coordinates(particles, **{f'nemitt_{plane}': 1e-6*bg})
    for key in ('x_norm', 'px_norm', 'y_norm', 'py_norm', 'zeta_norm', 'pzeta_norm'):
        np.testing.assert_allclose(ng[key], nn[key], rtol=1e-13, atol=1e-15)
    assert ng[f'gemitt_{plane}'] == 1e-6
    with pytest.raises(ValueError, match='Cannot provide both'):
        tw.get_normalized_coordinates(particles, **kwargs, **{f'nemitt_{plane}': 1e-6})


@pytest.mark.parametrize('method', ['rays', 'bisection', 'exact'])
@allow_kernel_compilation
def test_aperture_geometric_emittances(method):
    env = xt.load(string='''m: marker, apertype=circle, aperture={0.02};
                            seq: sequence,l=1; m,at=0; endsequence;''',
                  format='madx', install_limits=False)
    line = env['seq']
    line.particle_ref = xt.Particles(p0c=2e9)
    tw = line.twiss4d(betx=3, bety=5)
    norm, geom = emittances(line)
    aperture = line.aperture.from_associated_apertures()
    aperture.halo_params.update(norm)
    an = aperture.get_aperture_sigmas_at_element('m', twiss=tw, method=method)
    assert np.all(np.isfinite(an[0].n1))
    assert np.all(an[0].n1 > 0)
    aperture.halo_params.update(nemitt_x=None, nemitt_y=None, **geom)
    ag = aperture.get_aperture_sigmas_at_element('m', twiss=tw, method=method)
    np.testing.assert_allclose(an[0].n1, ag[0].n1, rtol=1e-12)
    aperture.halo_params['nemitt_x'] = 1e-6
    with pytest.raises(ValueError, match='Cannot provide both'):
        aperture.get_aperture_sigmas_at_element('m', twiss=tw, method=method)

    aperture.halo_params['nemitt_x'] = None
    expected_sigmas = aperture.get_aperture_sigmas_for_twiss(tw, method=method)
    envelope_kwargs = dict(sliced_twiss=tw, sigmas=1., envelopes_num_points=32,
                           include_aper_tols=False, polygons=False, extents=True)
    expected_envelope = aperture.get_envelope_for_twiss(**envelope_kwargs)

    # Geometric inputs need no reference energy once the optics are supplied.
    line.particle_ref = None
    sigmas = aperture.get_aperture_sigmas_for_twiss(tw, method=method)
    envelope = aperture.get_envelope_for_twiss(**envelope_kwargs)
    np.testing.assert_allclose(sigmas.n1, expected_sigmas.n1, rtol=1e-12)
    for key in ('min_x', 'max_x', 'min_y', 'max_y'):
        np.testing.assert_allclose(envelope[key], expected_envelope[key], rtol=1e-12)

    # Normalized inputs, including implicit defaults, still require an energy.
    message = r'line\.particle_ref must be set to convert normalized halo emittances'
    for inputs in (norm, {}, dict(gemitt_x=geom['gemitt_x'], nemitt_y=norm['nemitt_y'])):
        aperture.halo_params.update(nemitt_x=None, nemitt_y=None,
                                    gemitt_x=None, gemitt_y=None)
        aperture.halo_params.update(inputs)
        with pytest.raises(ValueError, match=message):
            aperture.get_aperture_sigmas_for_twiss(tw, method=method)
        with pytest.raises(ValueError, match=message):
            aperture.get_envelope_for_twiss(**envelope_kwargs)


@allow_kernel_compilation
def test_aperture_geometric_defaults_and_plot_labels():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    env = xt.load(string='''m: marker, apertype=circle, aperture={0.02};
                            seq: sequence,l=1; m,at=0; endsequence;''',
                  format='madx', install_limits=False)
    line = env['seq']
    line.particle_ref = xt.Particles(p0c=2e9)
    tw = line.twiss4d(betx=3, bety=5)
    aperture = line.aperture.from_associated_apertures()
    defaults = aperture.get_aperture_sigmas_for_twiss(tw)
    aperture.halo_params.update(nemitt_x=1e-6, nemitt_y=1e-6)
    explicit = aperture.get_aperture_sigmas_for_twiss(tw)
    np.testing.assert_allclose(defaults.n1, explicit.n1, rtol=1e-12)

    bg = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
    geom = dict(gemitt_x=1e-6/bg, gemitt_y=1e-6/bg)
    aperture.halo_params.update(nemitt_x=None, nemitt_y=None, **geom)
    original_params = aperture.halo_params.copy()
    twiss_init = tw.get_twiss_init('m')
    fig, axs = plt.subplots(2)
    fig_transverse, ax = plt.subplots()
    try:
        aperture.plot_extents(s_positions=[0., 0.5], sigmas=1.,
                              twiss_init=twiss_init, axs=axs, with_progress=False)
        aperture.plot_transverse(s_positions=[0.], sigmas=1.,
                                 twiss_init=twiss_init, ax=ax, with_progress=False)
        for title in (fig._suptitle.get_text(), ax.get_title()):
            for plane in ('x', 'y'):
                assert rf'\varepsilon_{{g,{plane}}}' in title
                assert f'{geom[f"gemitt_{plane}"]:.3e}' in title
        assert aperture.halo_params == original_params
    finally:
        plt.close(fig)
        plt.close(fig_transverse)
