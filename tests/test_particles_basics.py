# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
import xpart as xp

from xobjects.test_helpers import for_all_test_contexts


def _check_consistency_energy_variables(particles):
    # Check consistency between beta0 and gamma0
    xo.assert_allclose(particles.gamma0, 1/np.sqrt(1 - particles.beta0**2),
                       rtol=1e-14, atol=1e-14)

    # Assert consistency of p0c
    xo.assert_allclose(particles.p0c,
                       particles.mass0 * particles.beta0 * particles.gamma0,
                       rtol=1e-14, atol=1e-14)

    # Check energy0 property (consistency of p0c and gamma0)
    xo.assert_allclose(particles.energy0, particles.mass0 * particles.gamma0,
                       atol=1e-14, rtol=1e-14)

    # Check consistency of rpp and delta
    xo.assert_allclose(particles.rpp, 1./(particles.delta + 1),
                       rtol=1e-14, atol=1e-14)

    beta = particles.beta0 * particles.rvv
    gamma = 1/np.sqrt(1 - beta**2)
    pc = particles.mass * gamma * beta

    # Check consistency of delta with rvv
    xo.assert_allclose(particles.delta, (pc/particles.mass_ratio - particles.p0c)/(particles.p0c),
                       rtol=1e-14, atol=1e-14)

    # Check consistency of ptau with rvv
    energy = particles.mass * gamma
    xo.assert_allclose(particles.ptau, (energy/particles.mass_ratio - particles.energy0)/particles.p0c,
                       rtol=1e-14, atol=1e-14)

    # Check consistency of pzeta
    energy = particles.mass * gamma
    xo.assert_allclose(particles.pzeta, (energy/particles.mass_ratio  - particles.energy0)/(particles.beta0 * particles.p0c),
                       rtol=1e-14, atol=1e-14)

    # Check energy property
    xo.assert_allclose(particles.energy, energy, rtol=1e-14, atol=1e-14)


@for_all_test_contexts
def test_basics(test_context):
    particles = xp.Particles(_context=test_context, _capacity=10,
                             mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12,  # 7 TeV
                             x=[1e-3, 0], px=[1e-6, -1e-6], y=[0, 1e-3],
                             py=[2e-6, 0], zeta=[1e-2, 2e-2], delta=[0, 1e-4])

    dct = particles.to_dict() # transfers it to cpu
    assert dct['x'][0] == 1e-3
    assert dct['ptau'][0] == 0
    xo.assert_allclose(dct['ptau'][1], 1e-4, rtol=0, atol=1e-9)
    xo.assert_allclose(1/(dct['rpp'][1]) - 1, 1e-4, rtol=0, atol=1e-14)

    particles = xp.Particles(_context=test_context,
            mass0=xp.PROTON_MASS_EV, q0=1, p0c=3e9,
            x=[1e-3, 0], px=[1e-6, -1e-6], y=[0, 1e-3], py=[2e-6, 0],
            zeta=[1e-2, 2e-2], pzeta=[0, 1e-4])

    dct = particles.to_dict() # transfers it to cpu
    assert dct['x'][0] == 1e-3
    xo.assert_allclose(dct['ptau'][0], 0, atol=1e-14, rtol=0)
    xo.assert_allclose(dct['ptau'][1]/dct['beta0'][1], 1e-4, rtol=0, atol=1e-9)
    xo.assert_allclose(dct['delta'][1], 9.99995545e-05, rtol=0, atol=1e-13)

    particles.move(_context=xo.ContextCpu())
    _check_consistency_energy_variables(particles)


@for_all_test_contexts
def test_unallocated_particles(test_context):
    particles = xp.Particles(_context=test_context, _capacity=10,
                             mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12,  # 7 TeV
                             x=[1e-3, 0], px=[1e-6, -1e-6], y=[0, 1e-3],
                             py=[2e-6, 0], zeta=[1e-2, 2e-2], delta=[0, 1e-4])

    dct = particles.to_dict() # transfers it to cpu
    assert dct['x'][0] == 1e-3
    assert dct['ptau'][0] == 0
    xo.assert_allclose(dct['ptau'][1], 1e-4, rtol=0, atol=1e-9)
    xo.assert_allclose(1/(dct['rpp'][1]) - 1, 1e-4, rtol=0, atol=1e-14)

    particles2 = xp.Particles.from_dict(dct, _context=test_context)


@for_all_test_contexts(excluding='ContextPyopencl')
def test_linked_arrays(test_context):
    ctx2np = test_context.nparray_from_context_array
    np2ctx = test_context.nparray_to_context_array

    particles = xp.Particles(_context=test_context, p0c=26e9, delta=[1,2,3])

    assert ctx2np(particles.delta[2]) == 3
    xo.assert_allclose(ctx2np(particles.rvv[2]), 1.00061, rtol=0, atol=1e-5)
    xo.assert_allclose(ctx2np(particles.rpp[2]), 0.25, rtol=0, atol=1e-10)
    xo.assert_allclose(ctx2np(particles.ptau[2]), 2.9995115176, rtol=0, atol=1e-6)

    particles.delta[1] = particles.delta[2]

    assert particles.delta[2] == particles.delta[1]
    assert particles.ptau[2] == particles.ptau[1]
    assert particles.rpp[2] == particles.rpp[1]
    assert particles.rvv[2] == particles.rvv[1]

    particles.ptau[0] = particles.ptau[2]

    xo.assert_allclose(particles.delta[2], particles.delta[0], rtol=0, atol=1e-15)
    xo.assert_allclose(particles.ptau[2], particles.ptau[0], rtol=0, atol=1e-16)
    xo.assert_allclose(particles.rpp[2], particles.rpp[0], rtol=0, atol=1e-16)
    xo.assert_allclose(particles.rvv[2], particles.rvv[0], rtol=0, atol=1e-15)

    particles = xp.Particles(_context=test_context, p0c=26e9,
                             delta=[1,2,3,4,100,0])
    p0 = particles.copy()
    particles.state = np2ctx(np.array([1,1,1,1,0,1]))
    particles.delta[3:] = np2ctx([np.nan, 2, 3])

    assert particles.delta[5] == particles.delta[2]
    assert particles.ptau[5] == particles.ptau[2]
    assert particles.rvv[5] == particles.rvv[2]
    assert particles.rpp[5] == particles.rpp[2]

    assert particles.delta[4] == p0.delta[4]
    assert particles.ptau[4] == p0.ptau[4]
    assert particles.rvv[4] == p0.rvv[4]
    assert particles.rpp[4] == p0.rpp[4]

    assert particles.delta[3] == p0.delta[3]
    assert particles.ptau[3] == p0.ptau[3]
    assert particles.rvv[3] == p0.rvv[3]
    assert particles.rpp[3] == p0.rpp[3]


@for_all_test_contexts
@pytest.mark.parametrize(
    'varname,values',
    [
        ('p0c', [4e9, 5e11, 6e13]),
        ('gamma0', [3., 4., 5.]),
        ('beta0', [0.8, 0.9, 0.99999999]),
    ]
)
def test_particles_update_ref_vars(test_context, varname, values):
    p = xp.Particles(_context=test_context,
                     delta=[1, 2, 3],
                     **{varname: values})

    p_ref = p.copy()

    getattr(p, varname)[1] = getattr(p, varname)[0]

    assert p.p0c[0] == p.p0c[1]
    assert p.gamma0[0] == p.gamma0[1]
    assert p.beta0[0] == p.beta0[1]

    assert p.p0c[0] == p_ref.p0c[0]
    assert p.gamma0[0] == p_ref.gamma0[0]
    assert p.beta0[0] == p_ref.beta0[0]

@for_all_test_contexts
def test_particles_update_p0c_and_energy_deviations(test_context):

    part = xp.Particles(_context=test_context,
                     p0c=[1e12, 3e12, 2e12],
                     delta=[0,  0.1,    0],
                     state=[1,  0.,     1.])

    part.update_p0c_and_energy_deviations(p0c=2e12)

    part.move(_context=xo.ContextCpu())
    part.sort(interleave_lost_particles = True)
    xo.assert_allclose(part.p0c, [2e12, 3e12, 2e12], rtol=0, atol=1e-14)
    xo.assert_allclose(part.delta, [-0.5, 0.1, 0], rtol=0, atol=1e-14)


def test_sort():
    # Sorting available only on CPU for now

    p = xp.Particles(x=[0, 1, 2, 3, 4, 5, 6], _capacity=10)
    p.state[[0, 3, 4]] = 0

    line = xt.Line(elements=[xt.Cavity()])
    line.build_tracker()
    line.track(p)

    assert np.all(p.particle_id == np.array([6, 1, 2, 5, 4, 3, 0,
                                             -999999999, -999999999,
                                             -999999999]))
    assert np.all(p.x == np.array([6, 1, 2, 5, 4, 3, 0,
                                   -999999999, -999999999, -999999999]))
    assert np.all(p.state == np.array([1, 1, 1, 1, 0, 0, 0,
                                       -999999999, -999999999, -999999999]))
    assert p._num_active_particles == 4
    assert p._num_lost_particles == 3

    p.sort()

    assert np.all(p.particle_id == np.array([1, 2, 5, 6, 0, 3, 4,
                                             -999999999, -999999999,
                                             -999999999]))
    assert np.all(p.particle_id == np.array([1, 2, 5, 6, 0, 3, 4,
                                             -999999999, -999999999,
                                             -999999999]))
    assert np.all(p.state == np.array([1, 1, 1, 1, 0, 0, 0,
                                       -999999999, -999999999, -999999999]))
    assert p._num_active_particles == 4
    assert p._num_lost_particles == 3

    p.sort(interleave_lost_particles=True)

    assert np.all(p.particle_id == np.array([0, 1, 2, 3, 4, 5, 6,
                                             -999999999, -999999999,
                                             -999999999]))
    assert np.all(p.particle_id == np.array([0, 1, 2, 3, 4, 5, 6,
                                             -999999999, -999999999,
                                             -999999999]))
    assert np.all(p.state == np.array([0, 1, 1, 0, 0, 1, 1,
                                       -999999999, -999999999, -999999999]))
    assert p._num_active_particles == -2
    assert p._num_lost_particles == -2

    p = xp.Particles(x=[6, 5, 4, 3, 2, 1, 0], _capacity=10)
    p.state[[0,3,4]] = 0

    line.track(p)

    assert np.all(p.particle_id == np.array([6, 1, 2, 5, 4, 3, 0,
                                             -999999999, -999999999,
                                             -999999999]))
    assert np.all(p.x == np.array([0, 5, 4, 1, 2, 3, 6,
                                   -999999999, -999999999, -999999999]))
    assert np.all(p.state == np.array([1, 1, 1, 1, 0, 0, 0,
                                       -999999999, -999999999, -999999999]))
    assert p._num_active_particles == 4
    assert p._num_lost_particles == 3

    p.sort(by='x')

    assert np.all(p.particle_id == np.array([6, 5, 2, 1, 4, 3, 0,
                                             -999999999, -999999999,
                                             -999999999]))
    assert np.all(p.x == np.array([0, 1, 4, 5, 2, 3, 6,
                                   -999999999, -999999999, -999999999]))
    assert np.all(p.state == np.array([1, 1, 1, 1, 0, 0, 0,
                                       -999999999, -999999999, -999999999]))
    assert p._num_active_particles == 4
    assert p._num_lost_particles == 3

    p.sort(by='x', interleave_lost_particles=True)

    assert np.all(p.particle_id == np.array([6, 5, 4, 3, 2, 1, 0,
                                             -999999999, -999999999,
                                             -999999999]))
    assert np.all(p.x == np.array([0, 1, 2, 3, 4, 5, 6,
                                   -999999999, -999999999, -999999999]))
    assert np.all(p.state == np.array([1, 1, 0, 0, 1, 1, 0,
                                       -999999999, -999999999, -999999999]))
    assert p._num_active_particles == -2
    assert p._num_lost_particles == -2


@for_all_test_contexts
def test_python_add_to_energy(test_context):
    particles = xp.Particles(_context=test_context, mass0=xp.PROTON_MASS_EV,
                             q0=1, p0c=1.4e9, x=[1e-3, 0], px=[1e-6, -1e-6],
                             y=[0, 1e-3], py=[2e-6, 0], zeta=[1e-2, 2e-2],
                             delta=[0, 1e-4])

    energy_before = particles.copy(_context=xo.ContextCpu()).energy
    zeta_before = particles.copy(_context=xo.ContextCpu()).zeta

    particles.add_to_energy(3e6)

    expected_energy = energy_before + 3e6
    particles.move(_context=xo.ContextCpu())
    xo.assert_allclose(particles.energy, expected_energy,
                       atol=1e-14, rtol=1e-14)

    _check_consistency_energy_variables(particles)

    assert np.all(particles.zeta == zeta_before)

    particles = xp.Particles(_context=test_context, mass0=xp.PROTON_MASS_EV,
                             q0=1, p0c=1.4e9, x=[1e-3, 0], px=[1e-6, -1e-6],
                             y=[0, 1e-3], py=[2e-6, 0], zeta=[1e-2, 2e-2],
                             delta=[0, 1e-4], mass_ratio=[0.2, 1.7], charge_ratio=[0.5, 2])

    energy_before = particles.copy(_context=xo.ContextCpu()).energy
    zeta_before = particles.copy(_context=xo.ContextCpu()).zeta

    particles.add_to_energy(3e6)

    expected_energy = energy_before + 3e6
    particles.move(_context=xo.ContextCpu())
    xo.assert_allclose(particles.energy, expected_energy,
                       atol=1e-14, rtol=1e-14)

    _check_consistency_energy_variables(particles)

    assert np.all(particles.zeta == zeta_before)


@for_all_test_contexts
def test_python_delta_setter(test_context):
    particles = xp.Particles(_context=test_context, p0c=1.4e9, delta=[0, 1e-3],
                             px=[1e-6, -1e-6], py=[2e-6, 0], zeta=0.1)
    _check_consistency_energy_variables(
                                particles.copy(_context=xo.ContextCpu()))
    px_before = particles.copy(_context=xo.ContextCpu()).px
    py_before = particles.copy(_context=xo.ContextCpu()).py
    zeta_before = particles.copy(_context=xo.ContextCpu()).zeta
    gamma0_before = particles.copy(_context=xo.ContextCpu()).gamma0

    particles.delta = -2e-3

    particles.move(_context=xo.ContextCpu())
    xo.assert_allclose(particles.delta, -2e-3, atol=1e-14, rtol=1e-14)

    _check_consistency_energy_variables(particles)

    assert np.all(particles.gamma0 == gamma0_before)
    assert np.all(particles.zeta == zeta_before)
    assert np.all(particles.px == px_before)
    assert np.all(particles.py == py_before)


@for_all_test_contexts
def test_LocalParticle_add_to_energy(test_context):
    class TestElement(xt.BeamElement):
        _xofields={
            'value': xo.Float64,
            'pz_only': xo.Int64,
            }
        _extra_c_sources = ['''
            /*gpufun*/
            void TestElement_track_local_particle(
                    TestElementData el, LocalParticle* part0){
                double const value = TestElementData_get_value(el);
                int const pz_only = (int) TestElementData_get_pz_only(el);
                //start_per_particle_block (part0->part)
                    LocalParticle_add_to_energy(part, value, pz_only);
                //end_per_particle_block
            }
            ''']

    # pz_only = 1
    telem = TestElement(_context=test_context, value=1e6, pz_only=1)

    particles = xp.Particles(_context=test_context, p0c=1.4e9, delta=[0, 1e-3],
                             px=[1e-6, -1e-6], py=[2e-6, 0], zeta=0.1)
    _check_consistency_energy_variables(
                                particles.copy(_context=xo.ContextCpu()))
    energy_before = particles.copy(_context=xo.ContextCpu()).energy
    px_before = particles.copy(_context=xo.ContextCpu()).px
    py_before = particles.copy(_context=xo.ContextCpu()).py
    zeta_before = particles.copy(_context=xo.ContextCpu()).zeta
    gamma0_before = particles.copy(_context=xo.ContextCpu()).gamma0
    telem.track(particles)

    particles.move(_context=xo.ContextCpu())
    xo.assert_allclose(particles.energy, energy_before + 1e6,
                       atol=1e-14, rtol=1e-14)

    _check_consistency_energy_variables(particles)

    assert np.all(particles.gamma0 == gamma0_before)
    assert np.all(particles.zeta == zeta_before)
    assert np.all(particles.px == px_before)
    assert np.all(particles.py == py_before)

    # pz_only = 0
    telem = TestElement(_context=test_context, value=1e6, pz_only=0)

    particles = xp.Particles(_context=test_context, p0c=1.4e9, delta=[0, 1e-3],
                             px=[1e-6, -1e-6], py=[2e-6, 0], zeta=0.1)
    _check_consistency_energy_variables(
                                particles.copy(_context=xo.ContextCpu()))
    energy_before = particles.copy(_context=xo.ContextCpu()).energy
    px_before = particles.copy(_context=xo.ContextCpu()).px
    py_before = particles.copy(_context=xo.ContextCpu()).py
    rpp_before = particles.copy(_context=xo.ContextCpu()).rpp
    zeta_before = particles.copy(_context=xo.ContextCpu()).zeta
    gamma0_before = particles.copy(_context=xo.ContextCpu()).gamma0
    telem.track(particles)

    particles.move(_context=xo.ContextCpu())
    xo.assert_allclose(particles.energy, energy_before + 1e6,
                       atol=1e-14, rtol=1e-14)

    _check_consistency_energy_variables(particles)

    rpp_after = particles.copy(_context=xo.ContextCpu()).rpp
    assert np.all(particles.gamma0 == gamma0_before)
    assert np.all(particles.zeta == zeta_before)
    xo.assert_allclose(particles.px, px_before*rpp_before/rpp_after,
                       atol=1e-14, rtol=1e-14)
    xo.assert_allclose(particles.py, py_before*rpp_before/rpp_after,
                       atol=1e-14, rtol=1e-14)


@for_all_test_contexts
def test_LocalParticle_update_delta(test_context):
    class TestElement(xt.BeamElement):
        _xofields={
            'value': xo.Float64,
            }

        _extra_c_sources =['''
            /*gpufun*/
            void TestElement_track_local_particle(
                    TestElementData el, LocalParticle* part0){
                double const value = TestElementData_get_value(el);
                //start_per_particle_block (part0->part)
                    LocalParticle_update_delta(part, value);
                //end_per_particle_block
            }
            ''']

    telem = TestElement(_context=test_context, value=-2e-3)

    particles = xp.Particles(_context=test_context, p0c=1.4e9, delta=[0, 1e-3],
                             px=[1e-6, -1e-6], py=[2e-6, 0], zeta=0.1)
    _check_consistency_energy_variables(
                                particles.copy(_context=xo.ContextCpu()))
    px_before = particles.copy(_context=xo.ContextCpu()).px
    py_before = particles.copy(_context=xo.ContextCpu()).py
    zeta_before = particles.copy(_context=xo.ContextCpu()).zeta
    gamma0_before = particles.copy(_context=xo.ContextCpu()).gamma0
    telem.track(particles)

    particles.move(_context=xo.ContextCpu())
    xo.assert_allclose(particles.delta, -2e-3, atol=1e-14, rtol=1e-14)

    _check_consistency_energy_variables(particles)

    assert np.all(particles.gamma0 == gamma0_before)
    assert np.all(particles.zeta == zeta_before)
    assert np.all(particles.px == px_before)
    assert np.all(particles.py == py_before)


@for_all_test_contexts
def test_LocalParticle_update_ptau(test_context):
    class TestElement(xt.BeamElement):
        _xofields={
            'value': xo.Float64,
            }

        _extra_c_sources = ['''
            /*gpufun*/
            void TestElement_track_local_particle(
                    TestElementData el, LocalParticle* part0){
                double const value = TestElementData_get_value(el);
                //start_per_particle_block (part0->part)
                    LocalParticle_update_ptau(part, value);
                //end_per_particle_block
            }
            ''']

    telem = TestElement(_context=test_context, value=-2e-3)

    particles = xp.Particles(_context=test_context, p0c=1.4e9, delta=[0, 1e-3],
                             px=[1e-6, -1e-6], py=[2e-6, 0], zeta=0.1)
    _check_consistency_energy_variables(
                                particles.copy(_context=xo.ContextCpu()))
    px_before = particles.copy(_context=xo.ContextCpu()).px
    py_before = particles.copy(_context=xo.ContextCpu()).py
    zeta_before = particles.copy(_context=xo.ContextCpu()).zeta
    gamma0_before = particles.copy(_context=xo.ContextCpu()).gamma0
    telem.track(particles)

    particles.move(_context=xo.ContextCpu())
    xo.assert_allclose(particles.ptau, -2e-3, atol=1e-14, rtol=1e-14)

    _check_consistency_energy_variables(particles)

    assert np.all(particles.gamma0 == gamma0_before)
    assert np.all(particles.zeta == zeta_before)
    assert np.all(particles.px == px_before)
    assert np.all(particles.py == py_before)


@for_all_test_contexts
def test_LocalParticle_update_pzeta(test_context):
    class TestElement(xt.BeamElement):
        _xofields={
            'value': xo.Float64,
            }
        _extra_c_sources = ['''
            /*gpufun*/
            void TestElement_track_local_particle(
                    TestElementData el, LocalParticle* part0){
                double const value = TestElementData_get_value(el);
                //start_per_particle_block (part0->part)
                    double const pzeta = LocalParticle_get_pzeta(part);
                    LocalParticle_update_pzeta(part, pzeta+value);
                //end_per_particle_block
            }
            ''']

    telem = TestElement(_context=test_context, value=-2e-3)

    particles = xp.Particles(_context=test_context, p0c=1.4e9, delta=[0, 1e-3],
                             px=[1e-6, -1e-6], py=[2e-6, 0], zeta=0.1)
    _check_consistency_energy_variables(
                                particles.copy(_context=xo.ContextCpu()))
    px_before = particles.copy(_context=xo.ContextCpu()).px
    py_before = particles.copy(_context=xo.ContextCpu()).py
    ptau_before  = particles.copy(_context=xo.ContextCpu()).ptau
    zeta_before = particles.copy(_context=xo.ContextCpu()).zeta
    gamma0_before = particles.copy(_context=xo.ContextCpu()).gamma0
    telem.track(particles)

    particles.move(_context=xo.ContextCpu())
    xo.assert_allclose((particles.ptau - ptau_before)/particles.beta0,
                       -2e-3, atol=1e-14, rtol=1e-14)

    _check_consistency_energy_variables(particles)

    assert np.all(particles.gamma0 == gamma0_before)
    assert np.all(particles.zeta == zeta_before)
    assert np.all(particles.px == px_before)
    assert np.all(particles.py == py_before)


@for_all_test_contexts
def test_LocalParticle_update_p0c(test_context):
    class TestElement(xt.BeamElement):
        _xofields={
            'value': xo.Float64,
            }
        _extra_c_sources = ['''
            /*gpufun*/
            void TestElement_track_local_particle(
                    TestElementData el, LocalParticle* part0){
                double const value = TestElementData_get_value(el);
                //start_per_particle_block (part0->part)
                    LocalParticle_update_p0c(part, value);
                //end_per_particle_block
            }
            ''']

    telem = TestElement(_context=test_context, value=1.5e9)

    particles = xp.Particles(_context=test_context, p0c=1.4e9, delta=[0, 1e-3],
                             px=[1e-6, -1e-6], py=[2e-6, 0], zeta=0.1)

    _check_consistency_energy_variables(
                                particles.copy(_context=xo.ContextCpu()))
    px_before = particles.copy(_context=xo.ContextCpu()).px
    py_before = particles.copy(_context=xo.ContextCpu()).py
    energy_before  = particles.copy(_context=xo.ContextCpu()).energy
    beta0_before = particles.copy(_context=xo.ContextCpu()).beta0
    p0c_before = particles.copy(_context=xo.ContextCpu()).p0c
    zeta_before = particles.copy(_context=xo.ContextCpu()).zeta
    telem.track(particles)

    particles.move(_context=xo.ContextCpu())
    xo.assert_allclose(particles.p0c, 1.5e9, atol=1e-14, rtol=1e-14)
    xo.assert_allclose(particles.energy, energy_before, atol=1e-14, rtol=1e-14)

    _check_consistency_energy_variables(particles)

    xo.assert_allclose(particles.zeta, zeta_before*particles.beta0/beta0_before, atol=1e-14)
    xo.assert_allclose(particles.px, px_before*p0c_before/particles.p0c, atol=1e-14)
    xo.assert_allclose(particles.py, py_before*p0c_before/particles.p0c, atol=1e-14)



@for_all_test_contexts
def test_LocalParticle_angles(test_context):
    class ScaleAng(xt.BeamElement):
        _xofields={
            'scale_x':   xo.Float64,
            'scale_y':   xo.Float64,
            'scale_x2':  xo.Float64,
            'scale_y2':  xo.Float64,
            }
        _extra_c_sources = ['''
            /*gpufun*/
            void ScaleAng_track_local_particle(
                    ScaleAngData el, LocalParticle* part0){
                double const scale_x = ScaleAngData_get_scale_x(el);
                double const scale_y = ScaleAngData_get_scale_y(el);
                double const scale_x2 = ScaleAngData_get_scale_x2(el);
                double const scale_y2 = ScaleAngData_get_scale_y2(el);
                //start_per_particle_block (part0->part)
                    LocalParticle_scale_xp(part, scale_x);
                    LocalParticle_scale_yp(part, scale_y);
                    LocalParticle_scale_xp_yp(part, scale_x2, scale_y2);
                //end_per_particle_block
            }
            ''']

    class KickAng(xt.BeamElement):
        _xofields={
            'kick_x':  xo.Float64,
            'kick_y':  xo.Float64,
            'kick_x2': xo.Float64,
            'kick_y2': xo.Float64,
            }
        _extra_c_sources = ['''
            /*gpufun*/
            void KickAng_track_local_particle(
                    KickAngData el, LocalParticle* part0){
                double const kick_x = KickAngData_get_kick_x(el);
                double const kick_y = KickAngData_get_kick_y(el);
                double const kick_x2 = KickAngData_get_kick_x2(el);
                double const kick_y2 = KickAngData_get_kick_y2(el);
                //start_per_particle_block (part0->part)
                    LocalParticle_add_to_xp(part, kick_x);
                    LocalParticle_add_to_yp(part, kick_y);
                    LocalParticle_add_to_xp_yp(part, kick_x2, kick_y2);
                //end_per_particle_block
            }
            ''']

    telem1 = KickAng(_context=test_context, kick_x=-5.0e-3, kick_y=2.0e-3, kick_x2=12.0e-3, kick_y2=-6.0e-3)
    telem2 = ScaleAng(_context=test_context, scale_x=2.5, scale_y=1.3, scale_x2=1.2, scale_y2=0.7)
    line = xt.Line(elements=[xt.Drift(length=1.2), telem1, xt.Drift(length=0.3),
                             telem2, xt.Drift(length=2.8)])
    line.build_tracker(_context=test_context)
    particles = xp.Particles(_context=test_context, p0c=1.4e9, delta=[0, 1e-3],
                             px=[1.0e-3, -1.0e-3], py=[2.0e-3, -1.2e-3], zeta=0.1)
    line.track(particles)
    particles.move(_context=xo.ContextCpu())
    xo.assert_allclose(particles.px, [24.0e-3, 18.021e-3], atol=1e-14, rtol=1e-14)
    xo.assert_allclose(particles.py, [-1.82e-3, -4.73564e-3], atol=1e-14, rtol=1e-14)

    class ScaleAngExact(xt.BeamElement):
        _xofields={
            'scale_x':   xo.Float64,
            'scale_y':   xo.Float64,
            'scale_x2':  xo.Float64,
            'scale_y2':  xo.Float64,
            }
        _extra_c_sources = ['''
            /*gpufun*/
            void ScaleAngExact_track_local_particle(
                    ScaleAngExactData el, LocalParticle* part0){
                double const scale_x = ScaleAngExactData_get_scale_x(el);
                double const scale_y = ScaleAngExactData_get_scale_y(el);
                double const scale_x2 = ScaleAngExactData_get_scale_x2(el);
                double const scale_y2 = ScaleAngExactData_get_scale_y2(el);
                //start_per_particle_block (part0->part)
                    LocalParticle_scale_exact_xp(part, scale_x);
                    LocalParticle_scale_exact_yp(part, scale_y);
                    LocalParticle_scale_exact_xp_yp(part, scale_x2, scale_y2);
                //end_per_particle_block
            }
            ''']

    class KickAngExact(xt.BeamElement):
        _xofields={
            'kick_x':  xo.Float64,
            'kick_y':  xo.Float64,
            'kick_x2': xo.Float64,
            'kick_y2': xo.Float64,
            }
        _extra_c_sources = ['''
            /*gpufun*/
            void KickAngExact_track_local_particle(
                    KickAngExactData el, LocalParticle* part0){
                double const kick_x = KickAngExactData_get_kick_x(el);
                double const kick_y = KickAngExactData_get_kick_y(el);
                double const kick_x2 = KickAngExactData_get_kick_x2(el);
                double const kick_y2 = KickAngExactData_get_kick_y2(el);
                //start_per_particle_block (part0->part)
                    LocalParticle_add_to_exact_xp(part, kick_x);
                    LocalParticle_add_to_exact_yp(part, kick_y);
                    LocalParticle_add_to_exact_xp_yp(part, kick_x2, kick_y2);
                //end_per_particle_block
            }
            ''']

    telem1 = KickAngExact(_context=test_context, kick_x=-5.0e-3, kick_y=2.0e-3, kick_x2=12.0e-3, kick_y2=-6.0e-3)
    telem2 = ScaleAngExact(_context=test_context, scale_x=2.5, scale_y=1.3, scale_x2=1.2, scale_y2=0.7)
    line = xt.Line(elements=[xt.Drift(length=1.2), telem1, xt.Drift(length=0.3),
                             telem2, xt.Drift(length=2.8)])
    line.build_tracker(_context=test_context)
    particles = xp.Particles(_context=test_context, p0c=1.4e9, delta=[0, 1e-3],
                             px=[1.0e-3, -1.0e-3], py=[2.0e-3, -1.2e-3], zeta=0.1)
    line.track(particles)
    particles.move(_context=xo.ContextCpu())
    xo.assert_allclose(particles.px, [23.99302e-3, 18.01805e-3], atol=1e-14, rtol=5e-7)
    xo.assert_allclose(particles.py, [-1.81976e-3, -4.73529e-3], atol=1e-14, rtol=5e-7)
