import xtrack as xt
import xpart as xp
from cpymad.madx import Madx
from xtrack.mad_loader import MadLoader
import matplotlib.pyplot as plt
import numpy as np


# Make thin using madx
mad = Madx()
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence_thick.madx')
mad.input('beam, particle=proton, energy=7000;')
mad.input('seqedit,sequence=lhcb1;flatten;cycle,start=IP7;flatten;endedit;')
mad.sequence.lhcb1.use()
seq = mad.sequence.lhcb1
mad.globals.on_x1 = 0
mad.globals.on_x5 = 0
mad.globals.vrf400 = 0

mad.input('twiss, deltap=1e-7;')
plt.plot(mad.table.twiss.s, mad.table.twiss.x, '-')

plt.show()

tw0 = mad.twiss()
tw0df = tw0.dframe()
summ0 = tw0.summary.__dict__.copy()


# for ee in seq.elements:
#     if hasattr(ee, 'KILL_ENT_FRINGE'):
#         ee.KILL_ENT_FRINGE = True
#     if hasattr(ee, 'KILL_EXI_FRINGE'):
#         ee.KILL_EXI_FRINGE = True
    # kill sextupoles
    # if hasattr(ee, 'k2'):
    #     ee.k2 = 0

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1, allow_thick=True)
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, energy0=7e12)
line.twiss_default['method'] = '4d'
line.build_tracker()
line.freeze_longitudinal(True)

tw0xs = line.twiss()

for ee in line.elements:
    if isinstance(ee, xt.DipoleEdge):
        ee.r21 = 0
        ee.r43 = 0

twnf_mad = mad.twiss()
twnf_mad_df = tw0.dframe()
summnf = tw0.summary.__dict__.copy()

twxsnf = line.twiss()

delta1 = -1e-7
delta2 = 1e-7

tt_1 = line.twiss(method='4d', delta0=delta1)
mad.input(f'twiss, deltap={delta1}')
tt_mad_1 = mad.table.twiss.dframe()

tt_2 = line.twiss(method='4d', delta0=delta2)
mad.input(f'twiss, deltap={delta2}')
tt_mad_2 = mad.table.twiss.dframe()

dqx_mad = (tt_mad_2.mux[-1] - tt_mad_1.mux[-1]) / (delta2 - delta1)
dqx = (tt_2.mux[-1] - tt_1.mux[-1]) / (delta2 - delta1)


twmad = mad.twiss(chrom=True)
twmad_no_chrom = mad.twiss(chrom=False)


print(f'dqx xsuite diff: {dqx}')
print(f'dqx xsuite:      {twxsnf.dqx}')
print(f'dqx mad diff:    {dqx_mad}')
print(f'dqx mad chrom:   {twmad.summary.dq1}')
print(f'dqx mad nochrom: {twmad_no_chrom.summary.dq1}')