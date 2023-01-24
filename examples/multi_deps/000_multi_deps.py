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

# prepare manager with common vars and f
mgr = xdeps.Manager()
newvref = mgr.ref({}, "vars")
newfref = mgr.ref(mgr1.containers["f"]._owner, "f")

# Load variables in common environment
vref = mgr1.containers["vars"]
newvref._owner.update(vref._owner) # copy data
mgr.copy_expr_from(mgr1, "vars") # copy expressions

# Load variables in common environment
vref = mgr2.containers["vars"]
newvref._owner.update(vref._owner) # copy data
mgr.copy_expr_from(mgr2, "vars") #copy expressions

# Prepare multi line
newe = {}
neweref = mgr.ref(newe, "eref")

# Load elements in specific environment
eref = mgr1.containers["element_refs"]
newe["lhcb1"] = eref._owner # bind data
mgr.copy_expr_from(mgr1, "element_refs", {"element_refs": neweref["lhcb1"]})

# Load elements in specific environment
eref = mgr2.containers["element_refs"]
newe["lhcb2"] = eref._owner # bind data
mgr.copy_expr_from(mgr2, "element_refs", {"element_refs": neweref["lhcb2"]})

# test results

mgr.find_deps([mgr.containers["vars"]["on_x1"]])
mgr.containers["vars"]["on_x1"] = 150

# test twiss
tracker1 = line1.build_tracker()
tracker2 = line2.build_tracker()

tw1 = tracker1.twiss()
tw2 = tracker2.twiss(reverse=True)

print(tw1["px"][np.array(tw1["name"]) == "ip1"])
print(tw2["px"][np.array(tw2["name"]) == "ip1"])
