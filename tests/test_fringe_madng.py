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
    return dict(
        edge_entry_model=model,
        edge_exit_model=model,
        edge_entry_angle=0.02,
        edge_exit_angle=-0.015,
        edge_entry_fint=0.5,
        edge_exit_fint=0.4,
        edge_entry_hgap=0.03,
        edge_exit_hgap=0.03,
    )


NG_BEND = 'l=2, angle=0.05, e1=0.02, e2=-0.015, fint=0.5, fintx=0.4, hgap=0.03'


@pytest.mark.parametrize(
    'ng_fringe, xt_model',
    [
        (1, 'dipole-only'),
        (3, 'full'),
    ],
)
def test_bend_fringe_vs_madng(mad, ng_fringe, xt_model):
    # Without k1 both codes track the body exactly; MAD-NG keeps the wedge
    # for fringe=0, so compare the full tracking
    out_ng = _madng_track(mad, f"sbend 'b' {{at=0, {NG_BEND}, fringe={ng_fringe}}}")
    out_xt = _xt_track(xt.Bend(length=2, angle=0.05, **_faces(xt_model)))
    np.testing.assert_allclose(out_xt, out_ng, rtol=0, atol=5e-10)


def test_bend_fringe_large_edge_angle_vs_madng(mad):
    # 42 deg faces with a finite gap (xsuite/xsuite#962): the finite-gap
    # term scales as pz/(1+delta)^2, which needs a MAD-NG with that fix
    ang = np.deg2rad(42)
    face = {
        f'edge_{ff}_{kk}': vv
        for ff in ('entry', 'exit')
        for kk, vv in dict(model='dipole-only', angle=ang, fint=0.9, hgap=0.025).items()
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
        xt.Quadrupole(length=1, k1=0.3, edge_entry_active=True, edge_exit_active=True),
    )
    assert np.max(np.abs(d_ng)) > 1e-10
    np.testing.assert_allclose(d_xt, d_ng, rtol=0, atol=1e-6 * np.max(np.abs(d_ng)))


def test_qsad_fringe_vs_madng(mad):
    d_xt, d_ng = _fringe_effect(
        mad,
        "quadrupole 'q' {at=0, l=1, k1=0.3, f1=0.2, f2=0.1, fringe=6}",
        xt.Quadrupole(
            length=1,
            k1=0.3,
            edge_entry_active=True,
            edge_exit_active=True,
            edge_entry_qint1=0.2,
            edge_entry_qint2=0.1,
            edge_exit_qint1=0.2,
            edge_exit_qint2=0.1,
        ),
    )
    assert np.max(np.abs(d_ng)) > 1e-6
    np.testing.assert_allclose(d_xt, d_ng, rtol=0, atol=1e-6 * np.max(np.abs(d_ng)))


def test_sextupole_fringe_vs_madng(mad):
    # xtrack applies the multipole fringe to all orders; MAD-NG up to
    # frngmax - 1 (frngmax = 3 includes the sextupole)
    d_xt, d_ng = _fringe_effect(
        mad,
        "sextupole 's' {at=0, l=0.5, k2=20, frngmax=3, fringe=2}",
        xt.Sextupole(length=0.5, k2=20, edge_entry_active=True, edge_exit_active=True),
        frngmax=3,
    )
    assert np.max(np.abs(d_ng)) > 1e-12
    # The sextupole body is nonlinear and integrated differently by the
    # two codes, so the difference does not isolate the fringe exactly
    np.testing.assert_allclose(d_xt, d_ng, rtol=0, atol=1e-2 * np.max(np.abs(d_ng)))
