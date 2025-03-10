# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2023.                 #
# ######################################### #
import numpy as np
import pytest

import xtrack as xt
import xobjects as xo
import xpart as xp


def test_check_is_active_sorting_openmp():
    test_context = xo.ContextCpu(omp_num_threads=5)

    class TestElement(xt.BeamElement):
        _xofields = {
            'states': xo.Int64[:],
        }

        _extra_c_sources = ["""
            #define XT_OMP_SKIP_REORGANIZE

            /*gpufun*/
            void TestElement_track_local_particle(
                TestElementData el,
                LocalParticle* part0
            ) {
                //start_per_particle_block (part0->part)
                    int64_t state = check_is_active(part);
                    int64_t id = LocalParticle_get_particle_id(part);
                    TestElementData_set_states(el, id, state);
                //end_per_particle_block
            }
        """]

    el = TestElement(
        _context=test_context,
        states=np.zeros(18, dtype=np.int64),
    )
    particles = xp.Particles(
        _context=test_context,
        state=[
            1, 0, 1, 0, 1,  # should be reordered to 1, 1, 1, 0, 0
            0, 0, 0, 0, 0,  # should be left intact
            0, 1, 0, 1, 1,  # should be reordered to 1, 1, 1, 0, 0
            1, 1, 0,        # should be left intact
        ],
        _capacity=22,       # there are 4 particles that are unallocated
        _no_reorganize=True,
    )

    el.track(particles)

    # We have five threads, so the particles should be split into chunks
    # of 5, 5, 5, 3 + 2 (unallocated), 2 (unallocated).
    assert len(particles.state) == 22

    # Check that each chunk is reorganized correctly.
    # First batch:
    assert np.all(particles.state[0:5] == [1, 1, 1, 0, 0])
    assert set(particles.particle_id[0:3]) == {0, 2, 4}
    assert set(particles.particle_id[3:5]) == {1, 3}

    # Second batch:
    assert np.all(particles.state[5:10] == [0, 0, 0, 0, 0])
    # Don't reorder if not needed:
    assert np.all(particles.particle_id[5:10] == [5, 6, 7, 8, 9])

    # Third batch:
    assert np.all(particles.state[10:15] == [1, 1, 1, 0, 0])
    assert set(particles.particle_id[10:13]) == {11, 13, 14}
    assert set(particles.particle_id[13:15]) == {10, 12}

    # Fourth batch:
    assert np.all(particles.state[15:20] == [1, 1, 0, -999999999, -999999999])
    # Don't reorder if not needed:
    assert np.all(particles.particle_id[15:18] == [15, 16, 17])

    # Fifth batch (unallocated):
    assert np.all(particles.state[20:22] == [-999999999, -999999999])


@pytest.mark.parametrize(
    'test_context',
    [
        xo.ContextCpu(),
        xo.ContextCpu(omp_num_threads=4),
    ]
)
def test_check_is_active_sorting_cpu_default(test_context):
    class TestElement(xt.BeamElement):
        _xofields = {
            'states': xo.Int64[:],
        }

        _extra_c_sources = ["""
            /*gpufun*/
            void TestElement_track_local_particle(
                TestElementData el,
                LocalParticle* part0
            ) {
                //start_per_particle_block (part0->part)
                    int64_t state = check_is_active(part);
                    int64_t id = LocalParticle_get_particle_id(part);
                    TestElementData_set_states(el, id, state);
                //end_per_particle_block
            }
        """]

    el = TestElement(
        _context=test_context,
        states=np.zeros(18, dtype=np.int64),
    )
    particles = xp.Particles(
        _context=test_context,
        state=[
            1, 0, 1, 0, 1,
            0, 0, 0, 0, 0,
            0, 1, 0, 1, 1,
            1, 1, 0,
        ],
        _no_reorganize=True,
    )
    # We want to simulate a situation where a recount is needed, so we put a
    # value of active particles to be equal to the total number of particles:
    particles._num_active_particles = 18

    el.track(particles)

    # Here we don't reorganize by batches, so we just check the whole array
    # to see if it's sensible:
    assert np.all(particles.state == ([1] * 8) + ([0] * 10))
    assert set(particles.particle_id[:8]) == {0, 2, 4, 11, 13, 14, 15, 16}
    assert set(particles.particle_id[8:]) == {1, 3, 5, 6, 7, 8, 9, 10, 12, 17}

def test_particles_energy_coordinates():
    p = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1,
                    kinetic_energy0=50e6, delta=0.1)

    beta0 = p.beta0[0]
    gamma0 = p.gamma0[0]
    mass0_ev = p.mass0
    energy0 = p.energy0[0]
    energy = p.energy[0]
    p0c = p.p0c[0]

    delta = p.delta[0]
    ptau = p.ptau[0]
    pzeta = p.pzeta[0]

    beta = beta0 * p.rvv[0]
    gamma = energy / energy0 * gamma0

    xo.assert_allclose(gamma0, 1 / np.sqrt(1 - beta0**2), rtol=0, atol=1e-14)
    xo.assert_allclose(gamma, 1 / np.sqrt(1 - beta**2), rtol=0, atol=1e-14)
    xo.assert_allclose(energy, mass0_ev * gamma, rtol=0, atol=1e-6) # 1e-6 eV
    xo.assert_allclose(energy0, mass0_ev * gamma0, rtol=0, atol=1e-6) # 1e-6 eV

    Pc = p0c * (1 + delta)
    xo.assert_allclose(Pc, mass0_ev * gamma * beta, rtol=0, atol=1e-6) # 1e-6 eV

    # Definitions delta/ptau/pzeta
    xo.assert_allclose(delta, (Pc - p0c) / p0c, rtol=0, atol=1e-14)
    xo.assert_allclose(ptau, (energy - energy0) / energy0 / beta0, rtol=0, atol=1e-14)
    xo.assert_allclose(pzeta,(energy - energy0) / energy0 / beta0**2, rtol=0, atol=1e-14)
    xo.assert_allclose(pzeta,(energy - energy0) / p0c / beta0, rtol=0, atol=1e-14)

    # Conversions
    xo.assert_allclose(delta, np.sqrt(ptau**2 + 2*ptau/beta0 + 1) - 1, rtol=0, atol=1e-14)
    xo.assert_allclose(delta, np.sqrt(beta0**2 * pzeta**2 + 2*pzeta + 1) - 1, rtol=0, atol=1e-14)
    xo.assert_allclose(delta, beta * ptau + (beta - beta0)/beta0, rtol=0, atol=1e-14)
    xo.assert_allclose(delta, beta * beta0 * pzeta + (beta - beta0)/beta0, rtol=0, atol=1e-14)

    # More relations from the physics manual
    xo.assert_allclose(gamma, gamma0 * (1 + beta0 * ptau), rtol=0, atol=1e-14)
    xo.assert_allclose(beta, np.sqrt(1 - (1 - beta0**2)/(1 + beta0 * ptau)**2), rtol=0, atol=1e-14)

    # Check relations for small energy deviations
    p_small = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1,
                        kinetic_energy0=50e6, delta=1e-4)
    delta_small = p_small.delta[0]
    ptau_small = p_small.ptau[0]
    pzeta_small = p_small.pzeta[0]
    beta_small = beta0 * p_small.rvv[0]

    xo.assert_allclose(delta_small, ptau_small/beta0, rtol=0, atol=1e-8)
    xo.assert_allclose(delta_small, pzeta_small, rtol=0, atol=1e-8)
    xo.assert_allclose(beta_small, beta0 + (1 - beta0**2) * ptau_small, rtol=0, atol=1e-8)
