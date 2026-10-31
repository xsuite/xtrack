import os
import re

import numpy as np
import pytest

import xtrack as xt

pymadng = pytest.importorskip('pymadng')

P0C = 10e9
X0 = np.array(
    [
        [2e-3, 1e-4, 1e-3, -2e-4, 0, 1e-3],
        [-1e-3, -3e-4, -2e-3, 1e-4, 0, -5e-4],
        [3e-3, 2e-4, -1.5e-3, 3e-4, 0, 0],
    ]
)


@pytest.fixture(scope='module')
def mad():
    # MADNG_PATH selects a MAD-NG binary other than the one bundled with pymadng
    with pymadng.MAD(mad_path=os.environ.get('MADNG_PATH') or None) as mad:
        yield mad


def _madng_track(mad, element, frngmax=2):
    mad.send(f"""
    local sequence, beam, track in MAD
    local sbend, quadrupole, sextupole in MAD.element
    local seq = sequence 'seq' {{ refer='entry', {element} }}
    local X0 = py:recv()
    local _, mflw = track {{sequence=seq, beam=beam {{particle='proton',
                           pc={P0C / 1e9}}}, X0=X0, observe=0, nslice=1,
                           frngmax={frngmax}}}
    local out = MAD.matrix(#mflw, 4)
    local names = {{'x', 'px', 'y', 'py'}}
    for i=1,#mflw do for j=1,4 do out:set(i, j, mflw[i][names[j]]) end end
    py:send(out)
    """)
    mad.send([list(xx) for xx in X0])
    return np.array(mad.recv())


def _xt_track(element):
    line = xt.Line(elements=[element])
    line.set_particle_ref('proton', p0c=P0C)
    line.build_tracker()
    p = line.build_particles(
        x=X0[:, 0], px=X0[:, 1], y=X0[:, 2], py=X0[:, 3], zeta=0, ptau=X0[:, 5]
    )
    line.track(p)
    return np.array([p.x, p.px, p.y, p.py]).T


def _without_fringe(element):
    element = element.copy()
    element.edge_entry_active = False
    element.edge_exit_active = False
    return element


def _fringe_effect(mad, ng_element, xt_element, frngmax=2):
    """Effect of the fringes (fringe on - fringe off) in both codes."""
    ng_off = re.sub(r'fringe=\d+', 'fringe=0', ng_element)
    d_ng = _madng_track(mad, ng_element, frngmax) - _madng_track(mad, ng_off, frngmax)
    d_xt = _xt_track(xt_element) - _xt_track(_without_fringe(xt_element))
    return d_xt, d_ng


def _faces(model):
    return {
        'edge_entry_model': model,
        'edge_exit_model': model,
        'edge_entry_angle': 0.02,
        'edge_exit_angle': -0.015,
        'edge_entry_fint': 0.5,
        'edge_exit_fint': 0.4,
        'edge_entry_hgap': 0.03,
        'edge_exit_hgap': 0.03,
    }


NG_BEND = 'l=2, angle=0.05, e1=0.02, e2=-0.015, fint=0.5, fintx=0.4, hgap=0.03'


@pytest.mark.parametrize(
    'ng_fringe, xt_model',
    [
        (1, 'no-mult'),
        (3, 'full'),
    ],
)
def test_bend_fringe_vs_madng(mad, ng_fringe, xt_model):
    # Without k1 both codes track the body exactly; MAD-NG keeps the wedge
    # for fringe=0, so compare the full tracking
    out_ng = _madng_track(mad, f"sbend 'b' {{at=0, {NG_BEND}, fringe={ng_fringe}}}")
    out_xt = _xt_track(xt.Bend(length=2, angle=0.05, **_faces(xt_model)))
    np.testing.assert_allclose(out_xt, out_ng, rtol=0, atol=5e-10)


def test_hard_edge_bend_fringe_vs_madng(mad):
    # fint = 0: the hard-edge dipole fringe
    faces = _faces('no-mult')
    faces.update(edge_entry_fint=0, edge_exit_fint=0)
    out_ng = _madng_track(
        mad,
        "sbend 'b' {at=0, l=2, angle=0.05, e1=0.02, e2=-0.015, fint=0, "
        "fintx=0, hgap=0.03, fringe=1}",
    )
    out_xt = _xt_track(xt.Bend(length=2, angle=0.05, **faces))
    np.testing.assert_allclose(out_xt, out_ng, rtol=0, atol=5e-10)


def _pymadng_version():
    return tuple(int(vv) for vv in re.findall(r'\d+', pymadng.__version__)[:3])


@pytest.mark.xfail(
    not os.environ.get('MADNG_PATH') and _pymadng_version() <= (0, 9, 6),
    reason='MAD-NG bundled with pymadng <= 0.9.6 lacks the xsuite/xsuite#962 fix',
)
def test_bend_fringe_large_edge_angle_vs_madng(mad):
    # 42 deg faces with a finite gap (xsuite/xsuite#962): the finite-gap
    # term scales as pz/(1+delta)^2, which needs a MAD-NG with that fix
    ang = np.deg2rad(42)
    face = {
        f'edge_{ff}_{kk}': vv
        for ff in ('entry', 'exit')
        for kk, vv in {'model': 'no-mult', 'angle': ang, 'fint': 0.9, 'hgap': 0.025}.items()
    }
    out_ng = _madng_track(
        mad,
        f"sbend 'b' {{at=0, l=2.2, angle=0.3, e1={ang}, e2={ang}, "
        f'fint=0.9, hgap=0.025, fringe=1}}',
    )
    out_xt = _xt_track(xt.Bend(length=2.2, angle=0.3, **face))
    np.testing.assert_allclose(out_xt, out_ng, rtol=0, atol=5e-10)


def test_quadrupole_fringe_vs_madng(mad):
    d_xt, d_ng = _fringe_effect(
        mad,
        "quadrupole 'q' {at=0, l=1, k1=0.3, fringe=2}",
        xt.Quadrupole(length=1, k1=0.3, edge_entry_model='full', edge_exit_model='full'),
    )
    assert np.max(np.abs(d_ng)) > 1e-10
    np.testing.assert_allclose(d_xt, d_ng, rtol=0, atol=1e-6 * np.max(np.abs(d_ng)))


def _ng_quad(strengths, f1, f2, faces='both'):
    """MAD-NG quadrupole with a soft fringe on both faces, or different ones
    (two halves, each with one face)."""
    if faces == 'both':
        return f"quadrupole 'q' {{at=0, l=1, {strengths}, f1={f1}, f2={f2}, fringe=6}}"
    (f1a, f1b), (f2a, f2b) = f1, f2
    return (
        f"quadrupole 'qa' {{at=0, l=0.5, {strengths}, f1={f1a}, f2={f2a}, "
        f"fringe=6, kill_exi_fringe=true}}, "
        f"quadrupole 'qb' {{at=0.5, l=0.5, {strengths}, f1={f1b}, f2={f2b}, "
        f"fringe=6, kill_ent_fringe=true}}"
    )


@pytest.mark.parametrize(
    'strengths, xt_strengths, f1, f2',
    [
        ('k1=0.3', {'k1': 0.3}, 0.2, 0.1),
        ('k1=0.3', {'k1': 0.3}, 0, 0.1),
        ('k1=0.3', {'k1': 0.3}, -0.2, 0),
        ('k1=-0.3', {'k1': -0.3}, 0.2, 0.1),
        # The skew gradient is not integrated exactly in the body: enough
        # kicks to bring the body below the soft fringe tolerance
        (
            'k1=0.2, k1s=0.15',
            {'k1': 0.2, 'k1s': 0.15, 'num_multipole_kicks': 1000},
            0.2,
            0.1,
        ),
        ('k1=0.3', {'k1': 0.3}, (0.2, 0.05), (0.1, -0.03)),
    ],
)
def test_quad_lw_fringe_vs_madng(mad, strengths, xt_strengths, f1, f2):
    # Effect of the soft fringe alone (qint on - qint off, at the full
    # model) in both codes. The particles are off momentum.
    faces = 'both' if np.isscalar(f1) else 'different'
    f1_entry, f1_exit = (f1, f1) if faces == 'both' else f1
    f2_entry, f2_exit = (f2, f2) if faces == 'both' else f2
    no_qint = 0 if faces == 'both' else (0, 0)
    d_ng = _madng_track(mad, _ng_quad(strengths, f1, f2, faces)) - _madng_track(
        mad, _ng_quad(strengths, no_qint, no_qint, faces)
    )

    def quad(**qint):
        return xt.Quadrupole(
            length=1,
            edge_entry_model='full',
            edge_exit_model='full',
            **xt_strengths,
            **qint,
        )

    d_xt = _xt_track(
        quad(
            edge_entry_qint1=f1_entry,
            edge_entry_qint2=f2_entry,
            edge_exit_qint1=f1_exit,
            edge_exit_qint2=f2_exit,
        )
    ) - _xt_track(quad())
    assert np.max(np.abs(d_ng)) > 5e-7
    np.testing.assert_allclose(d_xt, d_ng, rtol=0, atol=1e-6 * np.max(np.abs(d_ng)))


def test_sextupole_fringe_vs_madng(mad):
    # xtrack applies the multipole fringe to all orders; MAD-NG up to
    # frngmax - 1 (frngmax = 3 includes the sextupole)
    d_xt, d_ng = _fringe_effect(
        mad,
        "sextupole 's' {at=0, l=0.5, k2=20, frngmax=3, fringe=2}",
        xt.Sextupole(length=0.5, k2=20, edge_entry_model='full', edge_exit_model='full'),
        frngmax=3,
    )
    assert np.max(np.abs(d_ng)) > 1e-12
    # The sextupole body is nonlinear and integrated differently by the
    # two codes, so the difference does not isolate the fringe exactly
    np.testing.assert_allclose(d_xt, d_ng, rtol=0, atol=1e-2 * np.max(np.abs(d_ng)))
