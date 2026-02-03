import xtrack as xt
import xobjects as xo
from pathlib import Path

# TODO:
# - Add new monitor to pre-compiled kernels
# - Forbid backtrack for now
# - Forbid collective mode for now
# - Forbid GPU for now

import numpy as np

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tt = line.get_table()
p = xt.Particles(p0c=7e12, x=1e-6*np.arange(20),
                           delta=0
)
line.track(p, num_turns=10)