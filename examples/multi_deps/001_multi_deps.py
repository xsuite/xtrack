import json
import time
import json

import numpy as np

import xtrack as xt
import xpart as xp
import xdeps


# load data
f1 = "../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json"
f4 = "../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b4.json"

line1 = xt.Line.from_dict(json.load(open(f1)))
line2 = xt.Line.from_dict(json.load(open(f4)))

mgr1 = line1._var_management["manager"]
mgr2 = line2._var_management["manager"]
vars1=mgr1.containers['vars']
vars2=mgr2.containers['vars']

# prepare manager with common vars and f
mgr = xdeps.Manager()
vref = mgr.ref({}, "vars")
lhcb1ref = mgr.ref(vars1, "lhcb1")
lhcb2ref = mgr.ref(vars2, "lhcb2")

vref._owner.update(mgr1.containers['vars']._owner)
for vv in vars1._owner:
    lhcb1ref[vv]=vref[vv]

vref._owner.update(mgr2.containers['vars']._owner)
for vv in vars2._owner:
    lhcb2ref[vv]=vref[vv]




