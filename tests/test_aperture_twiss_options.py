import numpy as np
import pytest

import xobjects as xo
import xtrack as xt


def _make_line_without_rf():
    env = xt.Environment()
    env.new('qf', 'Quadrupole', length=1.0, k1=0.12)
    env.new('qd', 'Quadrupole', length=1.0, k1=-0.12)
    env.new('drift', 'Drift', length=2.0)
    env.new('bend', 'Bend', length=2.0, angle=0.02)
    line = env.new_line(components=['qf', 'drift', 'qd', 'bend'])
    line.set_particle_ref('proton', p0c=3e9)
    profile = env.aperture.new_profile('round', 'Circle', radius=0.03)
    pipe = env.aperture.new_pipe('pipe', positions=[
        profile.place(at=0.0), profile.place(at=1.0),
    ])
    line.aperture.place_pipe('pipe', pipe, at='qf')
    line.aperture.halo_params.update(nemitt_x=2.5e-6, nemitt_y=2.5e-6)
    return line


@pytest.mark.parametrize('quantity', ['aperture_sigmas', 'hvd_aperture_sigmas', 'envelope'])
@pytest.mark.parametrize('location', ['s', 'element'])
def test_aperture_twiss_options_without_rf(quantity, location):
    line = _make_line_without_rf()
    aperture = line.aperture
    options = {'method': '4d', 'delta0': 1e-3}
    kwargs = {'sigmas': 3} if quantity == 'envelope' else {}
    if quantity == 'aperture_sigmas':
        kwargs['method'] = 'exact'
    position = [0.2, 0.6] if location == 's' else 'qf'
    compute = getattr(aperture, f'get_{quantity}_at_{location}')

    table, twiss = compute(position, twiss_options=options,
                           with_progress=False, **kwargs)

    assert options == {'method': '4d', 'delta0': 1e-3}
    assert aperture.line.twiss_default == {}
    xo.assert_allclose(twiss.delta, 1e-3, atol=1e-12, rtol=0)
    xo.assert_allclose(table.s, twiss.s, atol=1e-12, rtol=0)

    # The same options as line defaults must give the same result.
    aperture.line.twiss_default.update(options)
    reference, reference_twiss = compute(position, with_progress=False, **kwargs)
    for column in ('betx', 'bety', 'x', 'dx'):
        xo.assert_allclose(twiss[column], reference_twiss[column], atol=1e-12, rtol=0)
    column = {'aperture_sigmas': 'n1', 'hvd_aperture_sigmas': 'n1_horizontal',
              'envelope': 'cross_section'}[quantity]
    assert np.all(np.isfinite(table[column]))
    xo.assert_allclose(table[column], reference[column], atol=1e-12, rtol=0)


@pytest.mark.parametrize('plot', ['plot_extents', 'plot_transverse'])
@pytest.mark.parametrize('sigmas', [None, 3])
def test_aperture_plot_twiss_options_without_rf(plot, sigmas):
    from matplotlib.figure import Figure

    line = _make_line_without_rf()
    aperture = line.aperture
    options = {'method': '4d', 'delta0': 1e-3}
    figure = Figure()
    axes = {'axs': figure.subplots(2)} if plot == 'plot_extents' else {
        'ax': figure.subplots()}
    getattr(aperture, plot)(s_positions=[0.2, 0.6], sigmas=sigmas,
                           twiss_options=options, with_progress=False, **axes)
    assert options == {'method': '4d', 'delta0': 1e-3}
    assert aperture.line.twiss_default == {}
    assert all(ax.lines for ax in figure.axes)


def test_aperture_twiss_options_initial_conditions():
    line = _make_line_without_rf()
    aperture = line.aperture
    init = xt.TwissInit(betx=2, bety=3)
    options = {'method': '4d', 'init': init}
    _, twiss = aperture.get_aperture_sigmas_at_s(
        [0.0], twiss_options=options, with_progress=False)
    xo.assert_allclose(twiss.betx, 2, atol=1e-12, rtol=0)
    xo.assert_allclose(twiss.bety, 3, atol=1e-12, rtol=0)
    assert options == {'method': '4d', 'init': init}

    with pytest.raises(ValueError, match='Provide either twiss_init'):
        aperture.get_aperture_sigmas_at_s(
            [0.0], twiss_init=init, twiss_options=options)
    with pytest.raises(ValueError, match='require reverse=False'):
        aperture.get_aperture_sigmas_at_s(
            [0.0], twiss_options={'method': '4d', 'reverse': True})
