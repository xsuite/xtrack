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

with open('formadn.seq', 'r') as fid:
    formadng = fid.readlines()

with open('xsuite_to_mad.madx', 'r') as fid:
    xsmad = fid.readlines()

for i_start_seq_forng, ln in enumerate(formadng):
    if 'sequence, ' in ln:
        break

for i_end_seq_forng, ln in enumerate(formadng):
    if 'endsequence' in ln:
        break

for i_start_seq_xsmad, ln in enumerate(xsmad):
    if 'sequence, ' in ln:
        break

for i_end_seq_xsmad, ln in enumerate(xsmad):
    if 'endsequence' in ln:
        break


replace_until = 'mq.13l2.b1' # OK
replace_until = 'e.arc.12.b1' # OK
replace_until = 'e.ds.l2.b1' # OK
replace_until = 'msia.exit.b1' # OK
replace_until = 'lhcinj.b1' # OK
replace_until = 'mcbyv.a4l2.b1' # OK
replace_until = 'dfbxc.3l2' # OK
replace_until = 'mcosx.3l2' # OK
replace_until = 'mqxa.1l2' # OK
replace_until = 'mbwmd.1l2' # OK
replace_until = 'mbls2.1l2' # OK
replace_until = 'ip2' # OK
replace_until = 'ip3' # OK
replace_until = 'e.ds.l4.b1' # OK
replace_until = 'bpmwa.a5l4.b1' # OK



for i_end_replace_forng in range(i_start_seq_forng, i_end_seq_forng):
    ln = formadng[i_end_replace_forng]
    if replace_until+':' in ln:
        break
assert i_end_replace_forng > i_start_seq_forng
assert i_end_replace_forng < i_end_seq_forng-1

for i_end_replace_xsmad in range(i_start_seq_xsmad, i_end_seq_xsmad):
    ln = xsmad[i_end_replace_xsmad]
    if replace_until+':' in ln or replace_until+'_exit' in ln:
        break
assert i_end_replace_xsmad > i_start_seq_xsmad
assert i_end_replace_xsmad < i_end_seq_xsmad-1

# replace
formadng[i_start_seq_forng:i_end_replace_forng+1] = xsmad[i_start_seq_xsmad:i_end_replace_xsmad+1]

out = ''.join(formadng)
out = out.replace(': kicker' , '_: kicker')

with open('testseq.seq', 'w') as fid:
    fid.write(out)

from pymadng import MAD
mng2 = MAD()
# mng2.MADX.load('"xsuite_to_mad.madx"', f"'mad2.madng'")
# mng2.MADX.load('"manual.seq"', f"'mad2.madng'")
mng2.MADX.load('"testseq.seq"', f"'mad2.madng'")
mng2["lhcb1"] = mng2.MADX.lhcb1
mng2.lhcb1.beam = mng2.beam(particle='proton', energy=7000)
mng2["mytwtable", 'mytwflow'] = mng2.twiss(
    sequence=mng2.lhcb1, method=4, mapdef=2, implicit=True, nslice=3, save="'atbody'")

print(mng2["mytwtable"].mu1[-1])
assert np.isclose(mng2["mytwtable"].mu1[-1][0], 62.31, atol=1e-6)
