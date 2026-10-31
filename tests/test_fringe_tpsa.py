import madng_tpsa
import numpy as np
import pytest
import xobjects as xo

import xtrack as xt
import xtrack.tpsa as xtpsa

P0C = 7e12
MASS0 = xt.PROTON_MASS_EV
COORDS = ('x', 'px', 'y', 'py', 'zeta', 'delta')
X0 = {'x': 1e-3, 'px': 1.5e-4, 'y': -1e-3, 'py': 1e-4, 'zeta': 1e-3, 'delta': 2e-3}


def _faces(**kw):
    return {
        f'edge_{face}_{kk}': vv for face in ('entry', 'exit') for kk, vv in kw.items()
    }


FULL = _faces(model='full', angle=0.05, fint=0.5, hgap=0.02)
LINEAR = _faces(angle=0.05, fint=0.5, hgap=0.02)
# The fringes of straight magnets need a model: the default is "suppressed"
QUAD_HARD = _faces(model='full')
QUAD_LW = _faces(model='full', qint1=0.1, qint2=0.05)

ELEMENTS = {
    'bend_full': lambda: xt.Bend(
        length=1, angle=0.05, k1=0.02, knl=[0, 0, 0.01], **FULL
    ),
    'bend_linear': lambda: xt.Bend(length=1, angle=0.05),
    'quad_lw': lambda: xt.Quadrupole(length=1, k1=0.1, **QUAD_LW),
    'rbend_straight': lambda: xt.RBend(
        length_straight=1, angle=0.05, rbend_model='straight-body', **FULL
    ),
}


def _line(name):
    line = xt.Line(elements=[ELEMENTS[name]()])
    line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS0)
    line.build_tracker()
    return line


def _orbit(line, coords, backtrack):
    p = line.build_particles(**coords)
    line.track(p, backtrack=backtrack)
    return np.array([float(getattr(p, c)[0]) for c in COORDS])


def _fd_jacobian(line, coords, backtrack, h=1e-7):
    jac = np.zeros((6, 6))
    for j, c in enumerate(COORDS):
        plus, minus = dict(coords), dict(coords)
        plus[c] += h
        minus[c] -= h
        jac[:, j] = (_orbit(line, plus, backtrack) - _orbit(line, minus, backtrack)) / (
            2 * h
        )
    return jac


# TPSA tracking has no backtracking, so only the forward maps are checked
@pytest.mark.parametrize('name', list(ELEMENTS))
def test_fringe_tpsa_matches_scalar(name):
    line = _line(name)
    m = xtpsa.ParticlesTpsa(order=2, p0c=P0C, mass0=MASS0, **X0)
    line.track(m)

    xo.assert_allclose(m.const_part, _orbit(line, X0, False), rtol=0, atol=1e-15)
    xo.assert_allclose(m.jacobian(), _fd_jacobian(line, X0, False), rtol=0, atol=1e-7)


# A strength that is a TPSA parameter of value zero keeps its sensitivity
# through the fringes: the maps of zero strengths must not be skipped
SENSITIVITY_CASES = {
    'bend_k0_full': (lambda v: xt.Bend(length=1, angle=0.05, k0=v, **FULL), 'k0'),
    'bend_k0_linear': (lambda v: xt.Bend(length=1, angle=0.05, k0=v, **LINEAR), 'k0'),
    'quad_k1_mult': (lambda v: xt.Quadrupole(length=1, k1=v, **QUAD_HARD), 'k1'),
    'quad_k1_lw': (lambda v: xt.Quadrupole(length=1, k1=v, **QUAD_LW), 'k1'),
    # knl_rel is relative to k1 * length, in the body and in the fringes
    'quad_k1_knl_rel': (
        lambda v: xt.Quadrupole(
            length=1, k1=v, knl_rel=[0, 0, 0.5], ksl_rel=[0, 0, -0.2], **QUAD_HARD
        ),
        'k1',
    ),
}


@pytest.mark.parametrize('name', list(SENSITIVITY_CASES))
def test_fringe_strength_sensitivity(name):
    value = 0.0
    make, field = SENSITIVITY_CASES[name]

    def scalar_orbit(v):
        line = xt.Line(elements=[make(v)])
        line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS0)
        line.build_tracker()
        return _orbit(line, X0, False)

    element = make(value)
    line = xt.Line(elements=[element])
    line.particle_ref = xt.Particles(p0c=P0C, mass0=MASS0)
    descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
    setattr(element, field, descriptor.param(1, value))
    line.build_tracker()
    m = xtpsa.ParticlesTpsa(descriptor=descriptor, p0c=P0C, mass0=MASS0, **X0)
    line.track(m)

    h = 1e-5
    fd = (scalar_orbit(value + h) - scalar_orbit(value - h)) / (2 * h)
    xo.assert_allclose(m.param_jacobian()[:, 0], fd, rtol=1e-4, atol=1e-9)

