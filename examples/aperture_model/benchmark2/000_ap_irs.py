import xtrack as xt
from lhcoptics import LHCOptics, LHC

base = "https://cern.ch/acc-models/lhc/hl19/"
lhc = xt.load(f"{base}/xsuite/lhc.json")
lhc.vars.load(f"{base}/strengths/cycle_round_v0/opt_6000.madx")
lhc.set_particle_ref(p0c=450e9)
opt = LHCOptics.from_xsuite(lhc)

mad = opt.make_madx_model()
ap = mad.get_ap_irs()
# mad.get_ap_arc('12', '1')

# all sort of params in ap, including ap.s which we should use for benchmark slicing