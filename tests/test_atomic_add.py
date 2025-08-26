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
@pytest.mark.parametrize("ctype", [xo.Int8, xo.Int16, xo.Int32, xo.Int64,
                                   xo.UInt8, xo.UInt16, xo.UInt32,
                                   xo.UInt64, xo.Float32, xo.Float64])
def test_atomic_types(ctype, test_context):
    class TestAtomic(xt.BeamElement):
        _xofields = {f'val':  ctype}
        allow_track = False
        _extra_c_sources = [f'''
        #include <headers/track.h>
        #include <headers/atomicadd.h>

        GPUFUN
        {ctype._c_type} run_atomic_test(TestAtomicData el, {ctype._c_type} increment) {{
            {ctype._c_type}* val = TestAtomicData_getp_val(el);
            return atomicAdd(val, increment);
        }}
        ''']

        _kernels = {
            'run_atomic_test': xo.Kernel(
                    c_name='run_atomic_test',
                    args=[xo.Arg(xo.ThisClass, name='el'), xo.Arg(ctype, name='increment')],
                    ret=xo.Arg(ctype)
            )
        }

    num_steps = 10000
    atomic = TestAtomic(_context=test_context, val=0)
    assert atomic.val == 0

    if ctype.__name__.startswith('Uint'):
        low = 0
        high = 2**(8*ctype._size) - 1
        increments = [low, high, *np.random.randint(low, high+1, size=num_steps,
                                                    dtype=ctype._dtype)]
    elif ctype.__name__.startswith('Int'):
        low = -2**(8*ctype._size - 1)
        high = 2**(8*ctype._size - 1) - 1
        increments = [low, high, *np.random.randint(low, high+1, size=num_steps,
                                                    dtype=ctype._dtype)]
    elif ctype.__name__.startswith('Float'):
        low  = np.finfo(ctype._dtype).min
        high = np.finfo(ctype._dtype).max
        eps  = np.finfo(ctype._dtype).eps
        neps = np.finfo(ctype._dtype).epsneg
        rans = np.random.uniform(0, high, size=num_steps).astype(ctype._dtype)
        rans *= (-1)**np.random.randint(0, 2, size=num_steps)
        increments = [low, high, eps, neps, *rans, *(rans*(1+neps)), *(rans*(1-neps))]

    prev_val = atomic.val
    for inc in increments:
        ret = atomic.run_atomic_test(increment=inc)
        assert ret == prev_val
        assert atomic.val == prev_val + inc or (  # Possibly with overflow
            (ctype.__name__.startswith('Uint') and
            (atomic.val == low + (prev_val + inc - high - 1) if (prev_val + inc) > high else False)) or
            (ctype.__name__.startswith('Int') and
            ((prev_val + inc) > high and atomic.val == low + (prev_val + inc - high - 1) or
            (prev_val + inc) < low and atomic.val == high - (low - (prev_val + inc) - 1)))
        )
        prev_val = atomic.val


@for_all_test_contexts
@pytest.mark.parametrize("ctype", [xo.Int8, xo.Int16, xo.Int32, xo.Int64,
                                   xo.UInt8, xo.UInt16, xo.UInt32,
                                   xo.UInt64, xo.Float32, xo.Float64])
def test_atomic_concurrency(ctype, test_context):
    class TestAtomic(xt.BeamElement):
        _xofields = {f'val':  ctype}
        allow_track = False
        _extra_c_sources = [f'''
        #include <headers/track.h>
        #include <headers/atomicadd.h>

        #if defined(XO_CONTEXT_CUDA)
        #define X_HAS_GPU     1
        #define X_GID0        (blockIdx.x * blockDim.x + threadIdx.x)
        #define X_GSIZE0      (gridDim.x  * blockDim.x)
        #elif defined(XO_CONTEXT_CL)
        #define X_HAS_GPU     1
        #define X_GID0        get_global_id(0)
        #define X_GSIZE0      get_global_size(0)
        #else
        #define X_HAS_GPU     0
        #define X_GID0        0
        #define X_GSIZE0      1
        #endif

        GPUFUN
        {ctype._c_type} run_atomic_stress(TestAtomicData el, {ctype._c_type} increment,
                                          int nworkers, int iters) {{
            {ctype._c_type}* val = TestAtomicData_getp_val(el);
            *val = ({ctype._c_type})0;
        #if X_HAS_GPU
            // On GPU backends, concurrency comes from the launch configuration.
            // We still accept nworkers to allow early-return for extra threads.
            const size_t gid = (size_t)X_GID0;
            if (gid >= (size_t)nworkers) return *val;

            for (int i = 0; i < iters; ++i)
                (void)atomicAdd(val, increment);

            return *val;

        #else
            // CPU path
        #ifdef XO_CONTEXT_CPU_OPENMP
            *val = ({ctype._c_type})0;
            #pragma omp parallel num_threads(nworkers)
            {{
                for (int i = 0; i < iters; ++i)
                    (void)atomicAdd(val, increment);
            }}
        #else
            *val = ({ctype._c_type})0;
            for (int k = 0; k < nworkers * iters; ++k)
                (void)atomicAdd(val, increment);
        #endif
            return *val;
        #endif
        }}
        ''']

        _kernels = {
            'run_atomic_stress': xo.Kernel(
                c_name='run_atomic_stress',
                args=[xo.Arg(xo.ThisClass, name='el'),
                    xo.Arg(ctype, name='increment'),
                    xo.Arg(xo.Int32, name='nworkers'),
                    xo.Arg(xo.Int32, name='iters')],
                ret=xo.Arg(ctype),
                n_threads='nworkers',
            ),
        }

    atomic = TestAtomic(_context=test_context, val=0)
    assert atomic.val == 0

    if isinstance(test_context, xo.ContextCpu):
        nworkers = 8
        iters    = 10000
    else:
        nworkers = 32768
        iters    = 4
    inc = ctype._dtype.type(1 if ctype.__name__.startswith(('Int','Uint')) else 1.0)
    total = atomic.run_atomic_stress(increment=inc, nworkers=nworkers, iters=iters)

    # Expected result, with wrap-around for integers:
    def wrap_int(sum_, ctype):
        bits = 8 * ctype._size
        width = 1 << bits
        if ctype.__name__.startswith('Int'):
            lo = -(width // 2)
            # map s into [lo, lo+width-1]
            return ((sum_ - lo) % width) + lo
        else:
            return sum_ % width

    expect = wrap_int(int(inc) * nworkers * iters, ctype)
    assert int(total) == expect
