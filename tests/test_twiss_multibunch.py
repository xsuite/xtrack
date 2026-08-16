# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest

import xfields as xf
import xobjects as xo
import xtrack as xt


ZETA_BUNCHES = np.array([0.0, 0.2, 0.5])


@pytest.fixture
def multibunch_twiss_line(temp_context_default_func):
    opposing = xt.Particles(
        p0c=7e12,
        x=[0.2e-3, -0.35e-3, 0.5e-3],
        y=[-0.1e-3, 0.25e-3, 0.15e-3],
        zeta=ZETA_BUNCHES,
        weight=[0.8e11, 1.1e11, 1.4e11],
    )
    bb = xf.BeamBeamBiGaussianMultibunch2D(
        other_particles=opposing,
        zeta_match_tol=0.02,
        zeta_period=0.8,
        other_beam_q0=1,
        other_beam_beta0=float(opposing.beta0[0]),
        other_beam_sigma_x=[1.0e-3, 1.2e-3, 0.9e-3],
        other_beam_sigma_y=[1.3e-3, 0.8e-3, 1.1e-3],
    )
    arc = xt.LineSegmentMap(
        length=80,
        qx=0.31,
        qy=0.32,
        betx=[12, 12],
        bety=[15, 15],
    )
    line = xt.Line(
        elements=[bb, arc],
        element_names=['bb', 'arc'],
        particle_ref=xt.Particles(p0c=7e12),
    )
    line.build_tracker()
    return line


def test_twiss_multibunch_fast_modes_against_full(multibunch_twiss_line):
    line = multibunch_twiss_line
    names = ['slot_0', 'slot_1', 'slot_2']
    full = line.twiss_multibunch(
        zeta_bunches=ZETA_BUNCHES,
        bunch_names=names,
        mode='full',
        show_progress=False,
    )
    fast = line.twiss_multibunch(
        zeta_bunches=ZETA_BUNCHES,
        bunch_names=names,
        mode='fast',
        show_progress=False,
        co_tol=1e-12,
    )
    fast_orbit = line.twiss_multibunch(
        zeta_bunches=ZETA_BUNCHES,
        bunch_names=names,
        mode='fast_orbit',
        show_progress=False,
        co_tol=1e-12,
    )

    assert len(full) == len(fast) == len(fast_orbit) == len(ZETA_BUNCHES)
    assert fast.bunch_names == names
    assert fast.bunch('slot_1') is fast[1]
    xo.assert_allclose(fast.zeta_bunches, ZETA_BUNCHES, rtol=0, atol=0)

    for column in ('x', 'px', 'y', 'py'):
        xo.assert_allclose(
            fast[column], full[column], rtol=0, atol=5e-13)
        xo.assert_allclose(
            fast_orbit[column], full[column], rtol=0, atol=5e-13)
    for column in (
            'betx', 'alfx', 'bety', 'alfy', 'mux', 'muy',
            'dx', 'dpx', 'dy', 'dpy'):
        xo.assert_allclose(
            fast[column], full[column], rtol=2e-10, atol=2e-10)

    xo.assert_allclose(fast.qx, full.qx, rtol=0, atol=2e-10)
    xo.assert_allclose(fast.qy, full.qy, rtol=0, atol=2e-10)
    # fast_orbit has no accumulated phase advance, so qx/qy are fractional.
    # This toy ring has tunes below one and can be compared directly to full.
    xo.assert_allclose(fast_orbit.qx, full.qx, rtol=0, atol=2e-10)
    xo.assert_allclose(fast_orbit.qy, full.qy, rtol=0, atol=2e-10)
    xo.assert_allclose(
        fast['x', 'bb'], full['x', 'bb'], rtol=0, atol=5e-13)


def test_twiss_multibunch_reads_active_particle_zeta(multibunch_twiss_line):
    line = multibunch_twiss_line
    bunch_particles = xt.Particles(
        p0c=7e12,
        zeta=[0.5, 99.0, 0.0, 0.2],
        state=[1, 0, 1, 1],
    )
    from_particles = line.twiss_multibunch(
        particles=bunch_particles,
        mode='fast_orbit',
        show_progress=False,
        co_tol=1e-12,
    )
    explicit = line.twiss_multibunch(
        zeta_bunches=[0.5, 0.0, 0.2],
        mode='fast_orbit',
        show_progress=False,
        co_tol=1e-12,
    )

    xo.assert_allclose(
        from_particles.zeta_bunches, [0.5, 0.0, 0.2], rtol=0, atol=0)
    for column in ('x', 'px', 'y', 'py'):
        xo.assert_allclose(
            from_particles[column], explicit[column], rtol=0, atol=0)
    xo.assert_allclose(from_particles.qx, explicit.qx, rtol=0, atol=0)
    xo.assert_allclose(from_particles.qy, explicit.qy, rtol=0, atol=0)


def test_twiss_multibunch_rejects_invalid_inputs(multibunch_twiss_line):
    line = multibunch_twiss_line
    particles = xt.Particles(p0c=7e12, zeta=[0.0])

    with pytest.raises(ValueError, match='Provide exactly one'):
        line.twiss_multibunch()
    with pytest.raises(ValueError, match='Provide exactly one'):
        line.twiss_multibunch(zeta_bunches=[0.0], particles=particles)
    with pytest.raises(ValueError, match='No bunches'):
        line.twiss_multibunch(zeta_bunches=[])
    with pytest.raises(ValueError, match='Unknown mode'):
        line.twiss_multibunch(zeta_bunches=[0.0], mode='unknown')
    with pytest.raises(ValueError, match="requires method='4d'"):
        line.twiss_multibunch(
            zeta_bunches=[0.0], mode='fast', method='6d')
    with pytest.raises(ValueError, match='not supported'):
        line.twiss_multibunch(
            zeta_bunches=[0.0], mode='fast', freeze_longitudinal=True)
    with pytest.raises(ValueError, match='cannot be provided'):
        line.twiss_multibunch(zeta_bunches=[0.0], zeta0=0.0)
