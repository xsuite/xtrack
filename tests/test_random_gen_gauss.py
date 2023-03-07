# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from pathlib import Path

import numpy as np

import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts
import xtrack as xt
import xpart as xp

@for_all_test_contexts
def test_random_generation(test_context):

    part = xp.Particles(_context=test_context, p0c=6.5e12, x=[1,2,3])
    part._init_random_number_generator()

    class TestElement(xt.BeamElement):
        _xofields={
            'dummy': xo.Float64,
            }

        _depends_on = [xt.RandomNormal]

        _extra_c_sources = [
            '''
                /*gpufun*/
                void TestElement_track_local_particle(
                        TestElementData el, LocalParticle* part0){
                    //start_per_particle_block (part0->part)
                        double rr = RandomNormal_generate(part);
                        LocalParticle_set_x(part, rr);
                    //end_per_particle_block
                }
            '''
        ]

    telem = TestElement(_context=test_context)

    telem.track(part)

    # Use turn-by turin monitor to acquire some statistics
    line = xt.Line(elements=[telem])
    line.build_tracker(_buffer=telem._buffer)

    line.track(part, num_turns=1e6, turn_by_turn_monitor=True)

    for i_part in range(part._capacity):
        x = line.record_last_track.x[i_part, :]
        hstgm, bin_edges = np.histogram(x,  bins=50, range=(-3, 3), density=True)
        bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
        gauss = np.exp(-bin_centers**2/2)/np.sqrt(2.0*np.pi)
        assert np.allclose(hstgm, gauss, rtol=1e-10, atol=1E-2)


@for_all_test_contexts
def test_direct_sampling(test_context):
    n_seeds = 3
    n_samples = 3e6
    ran = xt.RandomNormal(_context=test_context)
    samples, _ = ran.generate(n_samples=n_samples, n_seeds=n_seeds)
    samples = test_context.nparray_from_context_array(samples)

    for i_part in range(n_seeds):
        hstgm, bin_edges = np.histogram(samples[i_part],  bins=50, range=(-3, 3), density=True)
        bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
        gauss = np.exp(-bin_centers**2/2)/np.sqrt(2.0*np.pi)
        assert np.allclose(hstgm, gauss, rtol=1e-10, atol=1E-2)

