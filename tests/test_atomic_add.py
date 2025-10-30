# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

import pytest
import numpy as np

import xtrack as xt
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
@pytest.mark.parametrize("overload", [True, False], ids=['overload', 'no_overload'])
@pytest.mark.parametrize("ctype", [xo.Int8, xo.Int16, xo.Int32, xo.Int64,
                                   xo.UInt8, xo.UInt16, xo.UInt32,
                                   xo.UInt64, xo.Float32, xo.Float64])
def test_atomic(overload, ctype, test_context):
    if overload:
        func_name = 'atomicAdd'
    else:
        func_name = f'atomicAdd_{ctype.__name__.lower()[0]}{ctype.__name__.split("t")[1]}'
    class TestAtomic(xt.BeamElement):
        _xofields = {f'val':  ctype}
        allow_track = False
        _extra_c_sources = [f'''
        #include <headers/track.h>
        #include <headers/atomicadd.h>

        GPUKERN
        void run_atomic_test(TestAtomicData el, GPUGLMEM {ctype._c_type}* increments,
                             GPUGLMEM {ctype._c_type}* retvals, int length) {{
            VECTORIZE_OVER(ii, length);
                GPUGLMEM {ctype._c_type}* val = TestAtomicData_getp_val(el);
                {ctype._c_type} ret = {func_name}(val, increments[ii]);
                retvals[ii] = ret;
            END_VECTORIZE;
        }}
        ''']

        _kernels = {
            'run_atomic_test': xo.Kernel(
                    c_name='run_atomic_test',
                    args=[xo.Arg(xo.ThisClass, name='el'),
                          xo.Arg(ctype, pointer=True, name='increments'),
                          xo.Arg(ctype, pointer=True, name='retvals'),
                          xo.Arg(xo.Int32, name='length')],
                    n_threads='length')
        }

    atomic = TestAtomic(_context=test_context, val=0)
    assert atomic.val == 0

    # Test with all increments = 1, so we can check the return values easily.
    num_steps = 10000
    if ctype.__name__.startswith('Int') or ctype.__name__.startswith('Uint'):
        # Less steps to avoid overflow
        num_steps = min(num_steps, 2**(8*ctype._size - 1) - 1)
    increments = test_context.zeros(shape=(num_steps,), dtype=ctype._dtype) + 1
    retvals = test_context.zeros(shape=(num_steps,), dtype=ctype._dtype)
    atomic.run_atomic_test(increments=increments, retvals=retvals, length=num_steps)
    assert atomic.val == num_steps
    retvals = np.sort(test_context.nparray_from_context_array(retvals))
    assert np.allclose(retvals, np.arange(num_steps, dtype=ctype._dtype), atol=1.e-15, rtol=1.e-15)

    # Test with random increments, where we now only can check the total sum
    # (retvals can be anything). Watch out: overflow is undefined behaviour,
    # except for unsigned integers, so we skip this test for signed integers.
    atomic.val = 0
    retvals = test_context.zeros(shape=(num_steps,), dtype=ctype._dtype)
    if ctype.__name__.startswith('Uint'):
        low = 0
        high = 2**(8*ctype._size) - 1
        increments = np.random.randint(low, high+1, size=num_steps, dtype=ctype._dtype)
        increments = test_context.nparray_to_context_array(increments)
        atomic.run_atomic_test(increments=increments, retvals=retvals, length=num_steps)
        increments = test_context.nparray_from_context_array(increments)
        assert atomic.val == (np.sum(increments).item() % (2**(8*ctype._size)))

    elif ctype.__name__.startswith('Float'):
        increments  = np.zeros(shape=(num_steps,), dtype=ctype._dtype)
        increments += np.random.uniform(0, 10, size=num_steps)
        increments  = test_context.nparray_to_context_array(increments)
        atomic.run_atomic_test(increments=increments, retvals=retvals, length=num_steps)
        increments = test_context.nparray_from_context_array(increments)
        if ctype == xo.Float32:
            assert np.isclose(atomic.val, np.sum(increments), atol=10., rtol=1.e-4)
        else:
            assert np.isclose(atomic.val, np.sum(increments), atol=1.e-6, rtol=1.e-12)
