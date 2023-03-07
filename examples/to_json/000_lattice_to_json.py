# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

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
line.to_json('line.json')

# Load from json
line_2 = xt.Line.from_json('line.json')

# Alternatively the to dict method can be used, which is more flexible for
# example to save additional information in the json file

#Save
dct = line.to_dict()
dct['my_additional_info'] = 'Important information'
with open('line.json', 'w') as fid:
    json.dump(dct, fid, cls=xo.JEncoder)

# Load
with open('line.json', 'r') as fid:
    loaded_dct = json.load(fid)
line_2 = xt.Line.from_dict(loaded_dct)
# loaded_dct['my_additional_info'] contains "Important information"