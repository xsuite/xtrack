import numpy as np
from cpymad.madx import Madx
from pymadng import MAD
import xtrack as xt
import os

# madx1 = Madx()
# madx1.call("../../test_data/hllhc15_thick/lhc.seq")
# madx1.call("../../test_data/hllhc15_thick/hllhc_sequence.madx")
# madx1.call("../../test_data/hllhc15_thick/opt_round_150_1500.madx")
# madx1.beam(particle='proton', energy=7000e9)
# madx1.use("lhcb1")
# madx1.input('save, sequence=lhcb1, file="formadn.seq";')

# mng1 = MAD()
# mng1.MADX.load('"formadn.seq"', f"'mad1.madng'")
# mng1["lhcb1"] = mng1.MADX.lhcb1
# mng1.lhcb1.beam = mng1.beam()
# mng1["mytwtable", 'mytwflow'] = mng1.twiss(
#     sequence=mng1.lhcb1, method=4, mapdef=2, implicit=True, nslice=3, save="'atbody'")


# line = xt.Line.from_madx_sequence(sequence=madx1.sequence.lhcb1,
#                                     deferred_expressions=True)
# line.particle_ref = xt.Particles(p0c=7000e9, mass0=xt.PROTON_MASS_EV)


# with open('xsuite_to_mad.madx', 'w') as fid:
#     fid.write(line.to_madx_sequence(sequence_name='lhcb1'))

# madx2 = Madx()
# madx2.call("xsuite_to_mad.madx")
# madx2.beam(particle='proton', energy=7000)
# madx2.use("lhcb1")
# tmx2 = madx2.twiss()

from pymadng import MAD
mng2 = MAD()
# mng2.MADX.load('"xsuite_to_mad.madx"', f"'mad2.madng'")
mng2.MADX.load('"manual.seq"', f"'mad2.madng'")
mng2["lhcb1"] = mng2.MADX.lhcb1
mng2.lhcb1.beam = mng2.beam(particle='proton', energy=7000)
mng2["mytwtable", 'mytwflow'] = mng2.twiss(
    sequence=mng2.lhcb1, method=4, mapdef=2, implicit=True, nslice=3, save="'atbody'")

print(mng2["mytwtable"].mu1[-1])

