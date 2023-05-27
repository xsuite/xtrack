from cpymad.madx import Madx
import xtrack as xt
import xpart as xp

# hllhc15 can be found at git@github.com:lhcopt/hllhc15.git

mad = Madx()

mad.input("""
call,file="../../../hllhc15/util/lhc.seq";
call,file="../../../hllhc15/hllhc_sequence.madx";
call,file="../../../hllhc15/toolkit/macro.madx";
seqedit,sequence=lhcb1;flatten;cycle,start=IP7;flatten;endedit;
seqedit,sequence=lhcb2;flatten;cycle,start=IP7;flatten;endedit;
exec,mk_beam(7000);
call,file="../../../hllhc15/round/opt_round_150_1500.madx";
exec,check_ip(b1);
exec,check_ip(b2);
""")

mad.use(sequence="lhcb1")

seq = mad.sequence.lhcb1

tw_mad = mad.twiss().dframe()

for ee in seq.elements:
    if hasattr(ee, 'KILL_ENT_FRINGE'):
        ee.KILL_ENT_FRINGE = True
    if hasattr(ee, 'KILL_EXI_FRINGE'):
        ee.KILL_EXI_FRINGE = True

delta1 = -1e-6
delta2 = 1e-6

# mad.input(f"twiss, deltap={delta1};")
# tw_mad1 = mad.table.twiss.dframe()

# mad.input(f"twiss, deltap={delta2};")
# tw_mad2 = mad.table.twiss.dframe()

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1, allow_thick=True)
line.particle_ref = xp.Particles(mass0=seq.beam.mass*1e9, gamma0=seq.beam.gamma)
line.twiss_default['method'] = '4d'
line.config.XTRACK_USE_EXACT_DRIFTS = True
line.build_tracker()

# tw0 = line.twiss()
# tw1 = line.twiss(delta0=delta1)
# tw2 = line.twiss(delta0=delta2)

# dqx_diff_mad = (tw_mad2.mux[-1] - tw_mad1.mux[-1]) / (delta2 - delta1)
# dqx_diff_xs = (tw2.mux[-1] - tw1.mux[-1]) / (delta2 - delta1)
tw0 = line.twiss()

betx0 = tw0.betx[0]
bety0 = tw0.bety[0]
alfx0 = tw0.alfx[0]
alfy0 = tw0.alfy[0]

mad.input(f"twiss, betx={betx0}, bety={bety0}, alfx={alfx0}, alfy={alfy0};")
tw_mad0 = mad.table.twiss.dframe()

mad.input(f"twiss, betx={betx0}, bety={bety0}, alfx={alfx0}, alfy={alfy0}, deltap={delta1};")
tw_mad1 = mad.table.twiss.dframe()

mad.input(f"twiss, betx={betx0}, bety={bety0}, alfx={alfx0}, alfy={alfy0}, deltap={delta2};")
tw_mad2 = mad.table.twiss.dframe()

dqx_diff_mad = (tw_mad2.mux[-1] - tw_mad1.mux[-1]) / (delta2 - delta1)


twiss_init=tw0.get_twiss_init(at_element=0)
twiss_init.particle_on_co.delta = delta1
tw1 = line.twiss(twiss_init=twiss_init)
twiss_init.particle_on_co.delta = delta2
tw2 = line.twiss(delta0=delta2, twiss_init=twiss_init)

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(tw_mad0.s, tw_mad2.mux - tw_mad1.mux, label='madx')
plt.plot(tw0.s, tw2.mux - tw1.mux, label='xtrack')

def twiss_with_dp(line, dp):
    for ee in line.elements:
        if isinstance(ee, xt.CombinedFunctionMagnet):
            ee.k0 /= (1 + dp)
            ee.k1 /= (1 + dp)
        if isinstance(ee, xt.Multipole):
            ee.knl /= (1 + dp)
            ee.ksl /= (1 + dp)
        if isinstance(ee, xt.DipoleEdge):
            ee.r21 /= (1 + dp)
            ee.r43 /= (1 + dp)
    res = line.twiss()
    for ee in line.elements:
        if isinstance(ee, xt.CombinedFunctionMagnet):
            ee.k0 *= (1 + dp)
            ee.k1 *= (1 + dp)
        if isinstance(ee, xt.Multipole):
            ee.knl *= (1 + dp)
            ee.ksl *= (1 + dp)
        if isinstance(ee, xt.DipoleEdge):
            ee.r21 *= (1 + dp)
            ee.r43 *= (1 + dp)
    return res

tw1 = twiss_with_dp(line, dp=-1e-6)
tw2 = twiss_with_dp(line, dp=1e-6)

plt.show()