# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from pathlib import Path

import numpy as np
import copy

import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts
import xtrack as xt
import xpart as xp

def ruth_PDF(t, A, B):
    return (A/(t**2))*(np.exp(-B*t))

t0 = 0.001
t1 = 0.02
rA = 0.0012306225579197868
rB = 53.50625
iterations = 7

@for_all_test_contexts(excluding=('ContextCupy', 'ContextPyopencl'))
def test_random_generation(test_context):

    part = xp.Particles(_context=test_context, p0c=6.5e12, x=[1,2,3])
    part._init_random_number_generator()

    class TestElement(xt.BeamElement):
        _xofields={
            'dummy': xo.Float64,
            'rng':   xt.RandomRutherford
            }

        _extra_c_sources = [
            '''
                /*gpufun*/
                void TestElement_track_local_particle(
                        TestElementData el, LocalParticle* part0){
                    RandomRutherfordData rng = TestElementData_getp_rng(el);
                    //start_per_particle_block (part0->part)
                        double rr = RandomRutherford_generate(rng, part);
                        LocalParticle_set_x(part, rr);
                    //end_per_particle_block
                }
            '''
        ]

        def __init__(self, **kwargs):
            if '_xobject' not in kwargs:
                kwargs.setdefault('rng', xt.RandomRutherford())
            super().__init__(**kwargs)

    telem = TestElement(_context=test_context)
    telem.rng.A = rA
    telem.rng.B = rB
    telem.rng.lower_val = t0
    telem.rng.upper_val = t1
    telem.rng.Newton_iterations = iterations

    telem.track(part)

    # Use turn-by-turn monitor to acquire some statistics
    line = xt.Line(elements=[telem])
    line.build_tracker(_buffer=telem._buffer)

    line.track(part, num_turns=1e6, turn_by_turn_monitor=True)

    for i_part in range(part._capacity):
        x = line.record_last_track.x[i_part, :]
        assert np.all(x[i_part]>=t0)
        assert np.all(x[i_part]<=t1)
        hstgm, bin_edges = np.histogram(x[0],  bins=100, range=(t0, t1), density=True)
        bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
        ruth = np.array([ruth_PDF(t, rA, rB) for t in bin_centers ])
        np.allclose(hstgm[:-10], ruth[:-10], rtol=5e-2, atol=1)


@for_all_test_contexts(excluding=('ContextCupy', 'ContextPyopencl'))
def test_direct_sampling(test_context):
    n_seeds = 3
    n_samples = 3e6
    ran = xt.RandomRutherford(_context=test_context)
    ran.A = rA
    ran.B = rB
    ran.lower_val = t0
    ran.upper_val = t1
    ran.Newton_iterations = iterations
    samples = ran.generate(n_samples=n_samples, n_seeds=n_seeds)
    samples = test_context.nparray_from_context_array(samples)

    for i_part in range(n_seeds):
        assert np.all(samples[i_part]>=t0)
        assert np.all(samples[i_part]<=t1)
        hstgm, bin_edges = np.histogram(samples[0],  bins=100, range=(t0, t1), density=True)
        bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
        ruth = np.array([ruth_PDF(t, rA, rB) for t in bin_centers ])
        np.allclose(hstgm[:-10], ruth[:-10], rtol=5e-2, atol=1)


@for_all_test_contexts(excluding=('ContextCupy', 'ContextPyopencl'))
def test_reproducibility(test_context):
    # 1e8 samples in total
    n_seeds = int(1e5)
    n_samples_per_seed = int(1e3)
    x_init = np.random.uniform(0.001, 0.003, n_seeds)
    part_init = xp.Particles(x=x_init, p0c=4e11, _context=test_context)
    part_init._init_random_number_generator(seeds=np.arange(n_seeds, dtype=int))
    ran = xt.RandomRutherford(_context=test_context)
    ran.A = rA
    ran.B = rB
    ran.lower_val = t0
    ran.upper_val = t1
    ran.Newton_iterations = iterations
    part1 = part_init.copy(_context=test_context)
    part2 = part_init.copy(_context=test_context)
    # Instead of having more particles - which would lead to memory issues -
    # we repeatedly sample and compare
    for i in range(2):  # 2 is enough; this is slow
        results1 = ran.generate(n_samples=n_samples_per_seed*n_seeds, particles=part1)
        results1 = test_context.nparray_from_context_array(results1)
        results1 = copy.deepcopy(results1)
        results2 = ran.generate(n_samples=n_samples_per_seed*n_seeds, particles=part2)
        results2 = test_context.nparray_from_context_array(results2)
        assert np.all(results1 == results2)

