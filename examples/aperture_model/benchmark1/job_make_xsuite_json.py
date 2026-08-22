import xtrack as xt
import numpy as np
from cpymad.madx import Madx
from xdeps import Table

madx = Madx()
madx.call("job_model.madx")

b1 = xt.Line.from_madx_sequence(madx.sequence.lhcb1, enable_layout_data=True)
b2 = xt.Line.from_madx_sequence(madx.sequence.lhcb2, enable_layout_data=True)

lhc = xt.Environment(lines={'b1': b1, 'b2': b2})
lhc.b1.set_particle_ref('proton', p0c=7e12)
lhc.b2.set_particle_ref('proton', p0c=7e12)
lhc.b2.twiss_default['reverse'] = True

for ll in lhc.lines.values():
    ll.twiss_default["method"] = "4d"
    ll.twiss_default["co_search_at"] = "ip7"
    ll.twiss_default["strengths"] = True

lhc.b1.metadata["aperture_offsets"] = {}
lhc.b2.metadata["aperture_offsets"] = {}

lhc["lagrf400.b1"] = 0.5
lhc["vrf400"] = 6.5

for ipn in range(1, 9):
    for beam in "14":
        tfs = Table.from_tfs(f"offsets/offset.ip{ipn}.b{beam}.tfs")
        tfs._data["name"] = np.array(tfs._data["name"], dtype=str)
        line = lhc.b1 if beam == "1" else lhc.b2
        line.metadata["aperture_offsets"][f"ip{ipn}"] = tfs._data.copy()

lhc.to_json("lhc_aperture.json")
