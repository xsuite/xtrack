import json
import time
import json

import numpy as np

import xtrack as xt
import xpart as xp
import xdeps

# This examples 

# load data
f1 = "../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json"
f4 = "../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b4.json"

line1 = xt.Line.from_dict(json.load(open(f1)))
line2 = xt.Line.from_dict(json.load(open(f4)))

vars1 = line1.vars
vars2 = line2.vars

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


print(vars1["on_x1"]._get_value())
print(vars2["on_x1"]._get_value())


vref["on_x1"] = 150

print(vars1["on_x1"]._get_value())
print(vars2["on_x1"]._get_value())


# test twiss
tracker1 = line1.build_tracker()
tracker2 = line2.build_tracker()

tw1 = tracker1.twiss()
print(tw1["px"][np.array(tw1["name"]) == "ip1"])

tw2 = tracker2.twiss(reverse=True)
print(tw2["px"][np.array(tw2["name"]) == "ip1"])
