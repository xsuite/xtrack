import xtrack as xt
from cpymad.madx import Madx
from xdeps import Table

madx = Madx()
madx.call("job_model.madx")

lines = xt.Environment.from_madx(madx=madx, enable_layout_data=True, return_lines=True)
lines["b1"] = lines.pop("lhcb1")
lines["b2"] = lines.pop("lhcb2")
lines["b1"].particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7e12)
lines["b2"].particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7e12)
lhc = xt.Environment(lines=lines)


lhc.b1.metadata['aperture_offsets']={}
lhc.b2.metadata['aperture_offsets']={}
for ll in lhc.lines.values():
    ll.twiss_default["method"] = "4d"
    ll.twiss_default["co_search_at"] = "ip7"
    ll.twiss_default["strengths"] = True
    ll.metadata = lines[ll.name].metadata

lhc['lagrf400.b1']=0.5
lhc['vrf400']=6.5


for ipn in range(1,9):
    for beam in "14":
        tfs=Table.from_tfs(f"offsets/offset.ip{ipn}.b{beam}.tfs")
        tfs._data['name']=np.array(tfs._data['name'],dtype=str)
        line=lhc.b1 if beam=="1" else lhc.b2
        line.metadata['aperture_offsets'][f"ip{ipn}"]=tfs._data.copy()

lhc.to_json("lhc_aperture.json")
