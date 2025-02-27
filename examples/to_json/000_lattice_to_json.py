# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np

import xtrack as xt
import xobjects as xo

env = xt.Environment()

# Build a beam line
line = env.new_line(components=[
    env.new('m1', xt.Multipole, knl=np.array([1.,2.,3.])),
    env.new('d1', xt.Drift, length=2.),
    env.new('c1', xt.Cavity, frequency=400e9, voltage=1e6),
    env.new('m2', xt.Multipole, knl=np.array([1.,2.,3.])),
    env.new('c2', xt.Drift, length=2.),
    ],
)

# Save to json
line.to_json('line.json')

# Load from json
line_2 = xt.Line.from_json('line.json')

# Alternatively the to_dict method can be used, which is more flexible for
# example to save additional information in the json file

#Save
dct = line.to_dict()
dct['my_additional_info'] = 'Important information'
xt.json.dump(dct, 'line.json')

# Load
loaded_dct = xt.json.load('line.json')
line_2 = xt.Line.from_dict(loaded_dct)
# loaded_dct['my_additional_info'] contains "Important information"