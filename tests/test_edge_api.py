import json

import numpy as np
import pytest
import xobjects as xo

import xtrack as xt

BEND_MODELS = ['linear', 'full', 'no-mult', 'suppressed']
STRAIGHT_MODELS = ['full', 'no-mult', 'suppressed']


def _new(cls, **kw):
    if cls is xt.RBend:
        return cls(length_straight=1, **kw)
    return cls(length=1, **kw)


@pytest.mark.parametrize(
    'cls, models, default_model, default_active',
    [
        (xt.Bend, BEND_MODELS, 'linear', 1),
        (xt.RBend, BEND_MODELS, 'linear', 1),
        (xt.Magnet, BEND_MODELS, 'linear', 1),
        (xt.Quadrupole, STRAIGHT_MODELS, 'full', 0),
        (xt.Sextupole, STRAIGHT_MODELS, 'full', 0),
        (xt.Octupole, STRAIGHT_MODELS, 'full', 0),
        (xt.UniformSolenoid, STRAIGHT_MODELS, 'suppressed', 1),
    ],
)
def test_edge_models_round_trip(cls, models, default_model, default_active):
    assert cls.get_available_edge_models() == models
    el = _new(cls)
    for face in ('entry', 'exit'):
        assert getattr(el, f'edge_{face}_model') == default_model
        assert getattr(el, f'edge_{face}_active') == default_active
    assert not any('edge' in kk for kk in el.to_dict())

    for model in models:
        el = _new(cls, edge_entry_model=model, edge_exit_active=False)
        assert el.edge_entry_model == model
        assert el.edge_entry_active == default_active
        assert el.edge_exit_active == 0
        el2 = cls.from_dict(json.loads(json.dumps(el.to_dict(), cls=xo.JEncoder)))
        assert el2.edge_entry_model == model
        assert el2.edge_exit_model == default_model
        assert el2.edge_exit_active == 0
        # Switching the face off and on keeps the model
        el.edge_entry_active = False
        el.edge_entry_active = True
        assert el.edge_entry_model == model


def test_invalid_edge_model():
    with pytest.raises(ValueError, match='soft-only'):
        xt.Bend(length=1, edge_entry_model='soft-only')
    with pytest.raises(ValueError, match='linear'):
        xt.Quadrupole(length=1).edge_exit_model = 'linear'
    with pytest.raises(ValueError, match='full'):
        xt.UniformSolenoid(length=1, edge_entry_model='linear')


def test_old_bend_dict_with_model_index():
    el = xt.Bend.from_dict(
        {
            '__class__': 'Bend',
            'length': 1,
            'angle': 0.1,
            'edge_entry_model': 2,
            'edge_exit_model': -1,
            'edge_exit_active': 0,
        }
    )
    assert el.edge_entry_model == 'no-mult'
    assert el.edge_exit_model == 'suppressed'
    assert el.edge_exit_active == 0


def _track(element):
    line = xt.Line(elements=[element])
    line.set_particle_ref('proton', p0c=1e9)
    line.build_tracker()
    p = line.build_particles(
        x=[1e-3, -2e-3],
        px=[1e-4, 2e-4],
        y=[2e-3, 1e-3],
        py=[-1e-4, 3e-4],
        delta=[1e-3, -2e-3],
    )
    line.track(p)
    return np.array([p.x, p.px, p.y, p.py, p.zeta, p.delta])


def test_solenoid_multipole_fringe_only_if_full():
    def sol(**kw):
        return xt.UniformSolenoid(length=1, ks=0.1, knl=[0, 0.05, 0.3], **kw)

    default = _track(sol())
    np.testing.assert_array_equal(
        default,
        _track(sol(edge_entry_model='suppressed', edge_exit_model='suppressed')),
    )
    full = _track(sol(edge_entry_model='full', edge_exit_model='full'))
    assert np.max(np.abs(default - full)) > 1e-12


def test_dipole_only_is_deprecated_name_of_no_mult():
    def bend(model):
        return xt.Bend(
            length=1, angle=0.1, k1=0.2, knl=[0, 0, 0.3], edge_entry_angle=0.05,
            edge_entry_fint=0.5, edge_entry_hgap=0.03,
            edge_entry_model=model, edge_exit_model=model,
        )

    with pytest.warns(FutureWarning, match='no-mult'):
        old = bend('dipole-only')
    assert old.edge_entry_model == 'no-mult'
    assert old.to_dict()['edge_entry_model'] == 'no-mult'
    np.testing.assert_array_equal(_track(old), _track(bend('no-mult')))

    line = xt.Line(elements={'b': bend('linear')}, element_names=['b'])
    with pytest.warns(FutureWarning, match='no-mult'):
        line.configure_bend_model(edge='dipole-only')
    assert line['b'].edge_entry_model == 'no-mult'


def _magnet_with_gradient(cls, **kw):
    """A magnet of each kind with a k1 (from knl) and a sextupole component,
    both faces active"""
    geo = dict(length_straight=1) if cls is xt.RBend else dict(length=1)
    if cls in (xt.Bend, xt.RBend):
        geo['angle'] = 0.1
    if cls is xt.UniformSolenoid:
        geo['ks'] = 0.1
    return cls(
        **geo, knl=[0, 0.3, 0.5], edge_entry_active=True, edge_exit_active=True, **kw
    )


SOFT_QUAD_MAGNETS = [
    xt.Bend, xt.RBend, xt.Magnet, xt.Quadrupole, xt.Sextupole, xt.Octupole,
    xt.UniformSolenoid,
]


@pytest.mark.parametrize('cls', SOFT_QUAD_MAGNETS)
def test_soft_quadrupole_fringe_on_all_magnets(cls):
    qint = dict(
        edge_entry_qint1=0.2, edge_entry_qint2=0.1,
        edge_exit_qint1=0.15, edge_exit_qint2=-0.05,
    )

    def track(model, **kw):
        return _track(_magnet_with_gradient(
            cls, edge_entry_model=model, edge_exit_model=model, **kw))

    # The soft fringe acts with the models full and no-mult only
    np.testing.assert_array_equal(
        track('suppressed', **qint), track('suppressed'))
    for model in ('full', 'no-mult'):
        assert np.max(np.abs(track(model, **qint) - track(model))) > 1e-10
    # no-mult has no hard multipole fringe
    assert np.max(np.abs(track('full') - track('no-mult'))) > 1e-10

    if cls in (xt.Magnet, xt.UniformSolenoid):
        return  # thick slicing only: no edge slices
    for model in ('full', 'no-mult'):
        el = _magnet_with_gradient(
            cls, edge_entry_model=model, edge_exit_model=model, **qint)
        assert el.to_dict()['edge_exit_qint2'] == -0.05
        _check_edge_slices(_sliced_line('m', el), 'm')


def test_soft_quadrupole_fringe_reversed_line():
    # A FODO cell whose focusing quadrupole has different soft fringes on
    # its two faces. The reversed line (as for a counter-rotating beam: the
    # gradients change sign, so the planes swap) swaps the faces, and has
    # the optics of the forward line read backwards.
    from xtrack.environment import _reverse_element

    env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
    faces = dict(edge_entry_active=True, edge_exit_active=True)
    line = env.new_line(components=[
        env.new('qf', xt.Quadrupole, length=0.4, k1=1.2, k1s=0.05, **faces,
                edge_entry_qint1=0.1, edge_entry_qint2=0.03,
                edge_exit_qint1=-0.06, edge_exit_qint2=0.01),
        env.new('d1', xt.Drift, length=2),
        env.new('qd', xt.Quadrupole, length=0.4, k1=-1.1, **faces),
        env.new('d2', xt.Drift, length=2.5),
    ])
    reversed_line = line.copy()
    for nn in reversed_line.element_names:
        _reverse_element(reversed_line, nn)
    reversed_line.element_names = reversed_line.element_names[::-1]
    assert reversed_line['qf'].edge_entry_qint1 == -0.06

    tw = line.twiss4d()
    tw_rev = reversed_line.twiss4d()
    # The soft fringe changes betx by about 0.07; swapping it back on the
    # reversed line would change the optics there by 0.04
    for forward, backward, sign in (
        (tw.betx, tw_rev.bety, 1), (tw.bety, tw_rev.betx, 1),
        (tw.alfx, tw_rev.alfy, -1), (tw.alfy, tw_rev.alfx, -1),
    ):
        np.testing.assert_allclose(
            sign * backward[::-1], forward, rtol=0, atol=1e-7)


def test_soft_quadrupole_fringe_not_exported_to_mad():
    from xtrack.mad_writer import to_madng_sequence

    env = xt.Environment()
    line = env.new_line(components=[
        env.new('q1', xt.Quadrupole, length=1, k1=0.1, edge_exit_qint2=0.1),
        env.new('q2', xt.Quadrupole, length=1, k1=0.1),
    ])
    line.slice_thick_elements([xt.Strategy(slicing=xt.Teapot(2))])
    for to_mad in (line.to_madx_sequence, to_madng_sequence):
        with pytest.warns(UserWarning, match=r"1 magnets have one, e.g. \['q1'\]"):
            to_mad(line)


def _sliced_line(name, element):
    line = xt.Line(elements={name: element}, element_names=[name])
    line.set_particle_ref('proton', p0c=1e9)
    line.slice_thick_elements([xt.Strategy(slicing=xt.Teapot(1))])
    line.build_tracker()
    return line


def _check_edge_slices(line, name):
    """Edge slices track as their equivalent elements; returns these."""
    out = []
    for face in ('entry', 'exit'):
        edge = line.element_dict[f'{name}..{face}_map']
        eq = edge.get_equivalent_element()
        tracked = []
        for el in (edge, eq):
            p = line.build_particles(
                x=[1e-3, -2e-3],
                px=[1e-4, 2e-4],
                y=[2e-3, 1e-3],
                py=[-1e-4, 3e-4],
                delta=[1e-3, -2e-3],
            )
            el.track(p)
            tracked.append(np.array([p.x, p.px, p.y, p.py]))
        np.testing.assert_allclose(tracked[1], tracked[0], rtol=0, atol=1e-16)
        out.append(eq)
    return out


@pytest.mark.parametrize(
    'cls, model, kw, equivalent',
    [
        (xt.Bend, 'linear', dict(k1=0.2), xt.DipoleEdge),
        (xt.Bend, 'no-mult', dict(k1=0.2), xt.DipoleEdge),
        (
            xt.Bend,
            'no-mult',
            dict(k1=0.2, edge_entry_qint1=0.2, edge_exit_qint1=0.2),
            xt.MagnetEdge,
        ),
        (
            xt.Bend,
            'full',
            dict(edge_entry_qint2=0.1, edge_exit_qint2=0.1),
            xt.MagnetEdge,
        ),
        (xt.Bend, 'full', dict(), xt.DipoleEdge),
        (xt.Bend, 'full', dict(k1=0.2), xt.MagnetEdge),
        (
            xt.Bend,
            'full',
            dict(k0=0.12, k2=0.5, knl=[0.01, 0, 0.3], knl_rel=[0, 0, 0.1]),
            xt.MagnetEdge,
        ),
        (xt.RBend, 'full', dict(k1=0.2), xt.MagnetEdge),
    ],
)
def test_bend_edge_slice_equivalent_element(cls, model, kw, equivalent):
    geo = dict(length_straight=1) if cls is xt.RBend else dict(length=1)
    bend = cls(
        **geo,
        angle=0.1,
        edge_entry_angle=0.05,
        edge_entry_fint=0.5,
        edge_entry_hgap=0.03,
        edge_exit_angle=0.02,
        edge_exit_fint=0.4,
        edge_exit_hgap=0.03,
        edge_entry_model=model,
        edge_exit_model=model,
        **kw,
    )
    line = _sliced_line('b', bend)
    for eq in _check_edge_slices(line, 'b'):
        assert isinstance(eq, equivalent)


@pytest.mark.parametrize(
    'kw',
    [
        dict(),
        dict(edge_entry_qint1=0.2, edge_entry_qint2=0.1),
        dict(edge_entry_model='no-mult', edge_entry_qint1=0.2, edge_entry_qint2=0.1),
        dict(edge_entry_model='no-mult'),
        dict(edge_entry_model='no-mult', knl=[0.01, 0, 0.3]),
        dict(edge_entry_qint1=0.2, k1s=0.1, knl=[0, 0, 0.3]),
    ],
)
def test_quadrupole_edge_slice_equivalent_element(kw):
    # Same settings at the exit
    kw.update(
        {
            kk.replace('edge_entry_', 'edge_exit_'): vv
            for kk, vv in kw.items()
            if kk.startswith('edge_entry_')
        }
    )
    quad = xt.Quadrupole(
        length=1, k1=0.5, edge_entry_active=True, edge_exit_active=True, **kw
    )
    line = _sliced_line('q', quad)
    _check_edge_slices(line, 'q')


MIXED_BEAM = dict(
    x=[1e-3, -2e-3, 0, 3e-3, 2e-3],
    px=[2e-4, 1e-4, 0, -3e-4, 1e-4],
    y=[-1e-3, 2e-3, 0, 1e-3, -2e-3],
    py=[1e-4, -2e-4, 0, 2e-4, 3e-4],
    zeta=[1e-3, -1e-3, 0, 2e-3, 0],
    delta=[1e-3, -2e-3, 0, 5e-3, -1e-2],
    charge_ratio=[1, 1, 0.5, 2, -1],
    mass_ratio=[1, 1, 0.25, 3, 1],
)
COORDS = ['x', 'px', 'y', 'py', 'zeta', 'delta', 'state']


@pytest.mark.parametrize('backtrack', [False, True])
@pytest.mark.parametrize(
    'make',
    [
        lambda: xt.Bend(
            length=1,
            angle=0.1,
            k1=0.2,
            knl=[0, 0, 0.3],
            edge_entry_model='full',
            edge_exit_model='full',
            edge_entry_angle=0.05,
            edge_exit_angle=0.02,
            edge_entry_fint=0.5,
            edge_entry_hgap=0.03,
        ),
        lambda: xt.Bend(
            length=1,
            angle=0.1,
            edge_entry_angle=0.05,
            edge_entry_fint=0.5,
            edge_entry_hgap=0.03,
        ),
        lambda: xt.RBend(
            length_straight=1,
            angle=0.1,
            k1=0.1,
            rbend_model='straight-body',
            edge_entry_model='full',
            edge_exit_model='full',
        ),
        lambda: xt.Quadrupole(
            length=1,
            k1=0.5,
            k1s=0.1,
            knl_rel=[0, 0, 0.2],
            edge_entry_active=True,
            edge_exit_active=True,
            edge_entry_qint1=0.2,
            edge_entry_qint2=0.1,
        ),
        lambda: xt.Sextupole(
            length=0.5, k2=3.0, edge_entry_active=True, edge_exit_active=True
        ),
        lambda: xt.UniformSolenoid(length=1, ks=0.4),
    ],
    ids=['bend_full', 'bend_linear', 'rbend_straight', 'quad', 'sext', 'solenoid'],
)
def test_mixed_beam_tracks_as_single_particles(make, backtrack):
    # The maps of the faces and the body are prepared once per element: each
    # particle (own chi, energy, coordinates) must still get its own map
    line = xt.Line(elements=[make()], element_names=['e'])
    line.particle_ref = xt.Particles(p0c=7e9, mass0=xt.PROTON_MASS_EV)
    line.build_tracker()

    def particles(ii=slice(None)):
        kw = {
            kk: np.atleast_1d(np.array(vv, dtype=float)[ii])
            for kk, vv in MIXED_BEAM.items()
        }
        return xt.Particles(
            p0c=7e9,
            mass0=xt.PROTON_MASS_EV,
            chi=kw['charge_ratio'] / kw['mass_ratio'],
            **kw,
        )

    beam = particles()
    line.track(beam, backtrack=backtrack)
    beam.sort(interleave_lost_particles=True)
    assert len(set(beam.chi)) > 1

    for ii in range(len(MIXED_BEAM['x'])):
        single = particles(ii)
        line.track(single, backtrack=backtrack)
        for cc in COORDS:
            xo.assert_allclose(
                getattr(beam, cc)[ii], getattr(single, cc)[0], rtol=0, atol=0
            )


def test_edge_bits_from_c_header():
    # The bits are defined once, in edge_flags.h, and read by _edge.py
    from xtrack.beam_elements import _edge

    assert set(_edge._EDGE_BITS) == {
        'EDGE_ACTIVE',
        'EDGE_LINEAR',
        'EDGE_BEND',
        'EDGE_MULT',
        'EDGE_QSAD',
    }
    assert sorted(_edge._EDGE_BITS.values()) == [1 << ii for ii in range(5)]
    for name, value in _edge._EDGE_BITS.items():
        assert getattr(_edge, name) == value
