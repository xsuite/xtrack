# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

import pathlib

import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_purely_longitudinal(test_context):
    p_fixed = xp.ParticlesFixed(p0c=[1, 2, 3], delta=[3, 2, 1], _context=test_context)
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
