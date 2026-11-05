import numpy as np
import pytest
import xobjects as xo
import xtrack as xt
from scipy.constants import c as clight
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_exciter_gives_no_kick_before_its_first_sample(test_context):
    ctx2np = test_context.nparray_from_context_array
    sampling_frequency = 1e9
    speed = xt.Particles(p0c=1e10).beta0[0] * clight
    exciter = xt.Exciter(_context=test_context, knl=[1e-6], samples=np.ones(10), frev=1e6,
                         sampling_frequency=sampling_frequency)
    before_first_sample = xt.Particles(_context=test_context, p0c=1e10, zeta=.8 / sampling_frequency * speed)
    after_first_sample = xt.Particles(_context=test_context, p0c=1e10, zeta=-.2 / sampling_frequency * speed)

    exciter.track(before_first_sample)
    exciter.track(after_first_sample)

    assert ctx2np(before_first_sample.px)[0] == 0.
    assert ctx2np(after_first_sample.px)[0] != 0.


def test_exciter_with_samples_refuses_no_revolution_frequency():
    with pytest.raises(ValueError, match='frev'):
        xt.Exciter(knl=[1e-6], samples=np.ones(10), sampling_frequency=1e6)


def test_exciter_refuses_revolution_frequency_set_to_zero():
    exciter = xt.Exciter(knl=[1e-6], samples=np.ones(10), sampling_frequency=1e6, frev=1e6)
    with pytest.raises(ValueError, match='frev'):
        exciter.frev = 0.
