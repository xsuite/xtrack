import json

import numpy as np

import xtrack as xt
import xobjects as xo

# Build a beam line
line = xt.Line(
    elements=[
        xt.Multipole(knl=np.array([1.,2.,3.])),
        xt.Drift(length=2.),
        xt.Cavity(frequency=400e9, voltage=1e6),
        xt.Multipole(knl=np.array([1.,2.,3.])),
        xt.Drift(length=2.),
    ],
    element_names=['m1', 'd1', 'c1', 'm2', 'c2']
)

# Save to json
with open('line.json', 'w') as fid:
    json.dump(line.to_dict(), fid, cls=xo.JEncoder)

# Load from json
with open('line.json', 'r') as fid:
    loaded_dct = json.load(fid)
line_2 = xt.Line.from_dict(loaded_dct)