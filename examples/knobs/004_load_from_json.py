import json
import math
import numpy as np

import xtrack as xt
import xdeps as xd

with open('status.json', 'r') as fid:
    dct = json.load(fid)

line = xt.Line.from_dict(dct)

tracker = xt.Tracker(line=line)

line.vars['on_x1'] = 250
assert np.isclose(tracker.twiss(at_elements=['ip1'])['px'][0], 250e-6,
                  atol=1e-6, rtol=0)

line.vars['on_x1'] = -300
assert np.isclose(tracker.twiss(at_elements=['ip1'])['px'][0], -300e-6,
                  atol=1e-6, rtol=0)