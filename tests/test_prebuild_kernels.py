# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2022.                 #
# ######################################### #

import os
from importlib.util import find_spec
from itertools import chain

import cffi
import pytest
import xpart as xp

import xtrack as xt
from xtrack.prebuild_kernels import PREBUILT_KERNELS as XT_KERNELS
from xtrack.prebuild_kernels import precompile_kernels as xt_precompile

try:
    from xfields.prebuild_kernels import PREBUILT_KERNELS as XF_KERNELS
    from xfields.prebuild_kernels import precompile_kernels as xf_precompile
except ModuleNotFoundError:
    def xf_precompile():
        pass

    XF_KERNELS = {}


@pytest.fixture(scope='session')
def with_precompiled_kernels():
    xt_precompile()
    xf_precompile()

    yield

    for precompiled_module in chain(XT_KERNELS.keys(), XF_KERNELS.keys()):
        spec = find_spec(precompiled_module)
        os.remove(spec.origin)


def test_precompiled_kernels_avoid_compiling(with_precompiled_kernels, mocker):
    cffi_compile = mocker.patch.object(cffi.FFI, 'compile')

    line = xt.Line(elements=[xt.Drift()])
    tracker = line.build_tracker()
    particles = xp.Particles()
    tracker.track(particles)

    cffi_compile.assert_not_called()
