# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

import cffi
import pathlib

import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_purely_longitudinal(test_context):
    p_fixed = xp.ParticlesPurelyLongitudinal(p0c=[1, 2, 3], delta=[3, 2, 1], _context=test_context)
    p = xp.Particles(p0c=[1, 2, 3], delta=[3, 2, 1], x=[1, 2, 3], _context=test_context)

    l = xt.Line(elements=[xt.Cavity(voltage=1e6, frequency=1e6)])
    t = xt.Tracker(line=l, compile=False, _context=test_context)

    t.track(p_fixed)
    t.track(p)

    d_fixed = p_fixed.to_dict()
    d = {k: v for k, v in p.to_dict().items() if k in d_fixed}

    assert d.keys() == d_fixed.keys()
    for k in d.keys():
        assert np.allclose(d[k], d_fixed[k])


@for_all_test_contexts
def test_per_particle_kernel(test_context, mocker):
    class TestElement(xt.BeamElement):
        _xofields = {
            'new_state': xo.Float64
        }

        _extra_c_sources = ["""
            /*gpufun*/
            void test_function(
                TestElementData el,
                LocalParticle* part0
            ) {
                double const new_state = TestElementData_get_new_state(el);

                //start_per_particle_block (part0->part)

                    // This function has different implementations for
                    // different particle types. It will change x for
                    // Particles, but not ParticlesPurelyLongitudinal since
                    // it does not have x.
                    LocalParticle_kill_particle(part, new_state);
            
                //end_per_particle_block
            }
            
            /*gpufun*/
            void TestElement_track_local_particle(
                TestElementData el,
                LocalParticle* part0
            ) {
                // irrelevant
            }
        """]

        _per_particle_kernels = {
            'test_kernel': xo.Kernel(
                c_name='test_function',
                args=[]),
        }

    el = TestElement(_context=test_context, new_state=42)

    # The per particle kernel should work for both particle types transparently.
    p1 = xp.ParticlesPurelyLongitudinal(p0c=1e9, zeta=[1, 2, 3], _context=test_context)
    p2 = xp.Particles(p0c=1e9, x=[1, 2, 3], zeta=[1, 2, 3], _context=test_context)

    el.test_kernel(p1)
    p1.move(_context=xo.ContextCpu())
    assert np.all(p1.zeta == 1e30)
    assert np.all(p1.state == 42)

    # Now, the kernel for `p1` should not be used, instead a new one must be
    # compiled. We check that by asserting a change in `x`: the kernel for a
    # purely longitudinal particles cannot change `x`.
    el.test_kernel(p2)
    p2.move(_context=xo.ContextCpu())
    assert np.all(p2.zeta == 1e30)
    assert np.all(p2.state == 42)
    assert np.all(p2.x == 1e30)

    # Now do it again, but verify that the kernels are reused, not recompiled.
    el.new_state = 64

    cffi_compile = mocker.patch.object(cffi.FFI, 'compile')

    el.test_kernel(p1)
    assert np.all(p1.state == 64)

    el.test_kernel(p2)
    assert np.all(p1.state == 64)

    cffi_compile.assert_not_called()
