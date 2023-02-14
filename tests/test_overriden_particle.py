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
def test_overriden_particle(test_context):
    p = xp.ParticlesFixed()
    l = xt.Line(elements=[xt.Cavity(voltage=1e6, frequency=1e6)])
    t = xt.Tracker(line=l, compile=False, _context=test_context)
    t.track(p)
