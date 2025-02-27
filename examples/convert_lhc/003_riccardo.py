import numpy as np
import xtrack as xt
from xtrack.mad_parser.loader import MadxLoader
from cpymad.madx import Madx
import matplotlib.pyplot as plt

import time


st=time.time()
mad = Madx(stdout=False)
mad.call('lhc.seq')
mad.call('squeeze_0.madx')
mad.input('beam, sequence=lhcb1, particle=proton, energy=6800;')
mad.input('beam, sequence=lhcb2, particle=proton, energy=6800,bv=-1;')
mad.use('lhcb1')
print("mad beam1 qx",mad.twiss().summary["q1"])
mad.use('lhcb2')
print("mad beam2 qx",mad.twiss().summary["q1"])
print(f"Time {time.time()-st:.3g} sec")

st=time.time()
mad = Madx(stdout=False)
mad.call('lhc.seq')
mad.call('squeeze_0.madx')
mad.input('beam, sequence=lhcb1, particle=proton, energy=6800;')
mad.input('beam, sequence=lhcb2, particle=proton, energy=6800,bv=-1;')
mad.use('lhcb1')
mad.use('lhcb2')
particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)
lhcb1_ref = xt.Line.from_madx_sequence(mad.sequence.lhcb1, deferred_expressions=True)
lhcb2_ref = xt.Line.from_madx_sequence(mad.sequence.lhcb2, deferred_expressions=True)
lhcb1_ref.particle_ref = particle_ref
lhcb2_ref.particle_ref = particle_ref
tw_refb1 = lhcb1_ref.twiss4d()
tw_refb2 = lhcb2_ref.twiss4d()
print("line beam1 qx",tw_refb1.qx)
print("line beam2 qx",tw_refb2.qx)
print(f"Time {time.time()-st:.3g} sec")

st=time.time()
loader = MadxLoader()
loader.load_file("lhc.seq")
loader.load_file("squeeze_0.madx")
env = loader.env
lhcb1 = env.lines['lhcb1']
lhcb2 = env.lines['lhcb2']
particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)
lhcb1.particle_ref = particle_ref
lhcb2.particle_ref = particle_ref
twb1 = lhcb1.twiss4d()
twb2 = lhcb2.twiss4d()
print("new loader beam1 qx",twb1.qx)
print("new loader beam2 qx",twb2.qx)
print(f"Time {time.time()-st:.3g} sec")

# lhcb1.merge_consecutive_drifts(inplace=True)
# lhcb1_ref.merge_consecutive_drifts(inplace=True)
#
# a = lhcb1.to_dict("lhcb1.json")
# b = lhcb1_ref.to_dict("lhcb1_ref.json")
#
# def convert(d):
#     return [[name, d['elements'][name]] for name in d['element_names']]
#
# import xtrack.json_utils as xjs
# xjs.to_json(convert(a), open("lhcb1.json", "w"), indent=2)
# xjs.to_json(convert(b), open("lhcb1_ref.json", "w"), indent=2)
