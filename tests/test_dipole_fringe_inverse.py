# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import allow_no_prebuilt_kernels, for_all_test_contexts


_DELTA_VALUES = np.array([-0.4, -0.2, 0.0, 0.1, 0.4, 0.8])
_COORDINATES = ('x', 'px', 'y', 'py', 'zeta', 'ptau', 'delta', 's')


def _assert_round_trip(particle_before, particle_after, atol=2e-12):
    particle_before.move(_context=xo.context_default)
    particle_after.move(_context=xo.context_default)

    assert np.all(particle_after.state == 1)
    for coordinate in _COORDINATES:
        xo.assert_allclose(
            getattr(particle_after, coordinate),
            getattr(particle_before, coordinate),
            rtol=0,
            atol=atol,
        )


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_full_dipole_edge_forward_backtrack_delta(test_context):
    """Round-trip the PSB comparison fringe over a broad delta range."""

    edge = xt.DipoleEdge(
        k=0.12,
        fint=100.0,
        hgap=0.035,
        model='full',
    )
    line = xt.Line(elements=[edge])
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, beta0=0.5)
    line.reset_s_at_end_turn = False
    line.build_tracker(_context=test_context)

    particle_before = line.build_particles(
        x=0.0,
        px=0.5,
        y=0.01,
        py=0.001,
        zeta=0.0,
        delta=_DELTA_VALUES,
    )
    particle_after = particle_before.copy(_context=test_context)

    line.track(particle_after)
    line.track(particle_after, backtrack=True)

    _assert_round_trip(particle_before, particle_after)


@for_all_test_contexts
@allow_no_prebuilt_kernels
def test_bend_dipole_only_edges_forward_backtrack_delta(test_context):
    """Exercise the same inverse through the thick-magnet edge machinery."""

    bend = xt.Bend(
        length=1.0,
        angle=0.1,
        k1=0.2,
        edge_entry_model='dipole-only',
        edge_exit_model='dipole-only',
        edge_entry_fint=0.5,
        edge_exit_fint=0.5,
        edge_entry_hgap=0.03,
        edge_exit_hgap=0.03,
    )
    line = xt.Line(elements=[bend])
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, beta0=0.7)
    line.reset_s_at_end_turn = False
    line.build_tracker(_context=test_context)

    particle_before = line.build_particles(
        x=0.01,
        px=0.02,
        y=0.03,
        py=0.04,
        zeta=0.05,
        delta=_DELTA_VALUES,
    )
    particle_after = particle_before.copy(_context=test_context)

    line.track(particle_after)
    line.track(particle_after, backtrack=True)

    _assert_round_trip(particle_before, particle_after)
