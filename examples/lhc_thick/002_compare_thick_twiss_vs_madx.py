from cpymad.madx import Madx
import xtrack as xt
import xdeps as xd

import numpy as np

# hllhc15 can be found at git@github.com:lhcopt/hllhc15.git

kill_fringes_and_edges = True

mad = Madx()

mad.input(f"""
call,file="../../test_data/hllhc15_thick/lhc.seq";
call,file="../../test_data/hllhc15_thick/hllhc_sequence.madx";
seqedit,sequence=lhcb1;flatten;cycle,start=IP7;flatten;endedit;
seqedit,sequence=lhcb2;flatten;cycle,start=IP7;flatten;endedit;
beam, sequence=lhcb1, particle=proton, pc=7000;
call,file="../../test_data/hllhc15_thick/opt_round_150_1500.madx";
""")

mad.use(sequence="lhcb1")
seq = mad.sequence.lhcb1
mad.twiss()

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1,
            allow_thick=True, deferred_expressions=True)
line.particle_ref = xt.Particles(mass0=seq.beam.mass*1e9, gamma0=seq.beam.gamma)
line.twiss_default['method'] = '4d'
line.twiss_default['matrix_stability_tol'] = 100


line.build_tracker()

line.to_json('lhc_thick_with_knobs.json', include_var_management=True)

if kill_fringes_and_edges:
    for ee in seq.elements:
        if hasattr(ee, 'KILL_ENT_FRINGE'):
            ee.KILL_ENT_FRINGE = True
        if hasattr(ee, 'KILL_EXI_FRINGE'):
            ee.KILL_EXI_FRINGE = True

    for ee in line.elements:
        if isinstance(ee, xt.DipoleEdge):
            ee.k = 0

tw_mad = mad.twiss().dframe()

delta1 = -1e-4
delta2 = 1e-4

mad.input(f"twiss, deltap={delta1};")
tw_mad1 = mad.table.twiss.dframe()

mad.input(f"twiss, deltap={delta2};")
tw_mad2 = mad.table.twiss.dframe()

tw0 = line.twiss()
tw1 = line.twiss(delta0=delta1)
tw2 = line.twiss(delta0=delta2)

dqx_diff_mad = (tw_mad2.mux[-1] - tw_mad1.mux[-1]) / (delta2 - delta1)
dqx_diff_xs = (tw2.mux[-1] - tw1.mux[-1]) / (delta2 - delta1)

dqy_diff_mad = (tw_mad2.muy[-1] - tw_mad1.muy[-1]) / (delta2 - delta1)
dqy_diff_xs = (tw2.muy[-1] - tw1.muy[-1]) / (delta2 - delta1)

tw0 = line.twiss()

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(tw_mad1.s, tw_mad2.mux - tw_mad1.mux, label='madx')
plt.plot(tw0.s, tw2.mux - tw1.mux, label='xtrack')

twmad = mad.twiss(table='twiss')
tmad = xd.Table(twmad)

print(f'Config: {kill_fringes_and_edges=} \n' )
print(f'dqx xsuite diff: {dqx_diff_xs}')
print(f'dqx xsuite:      {tw0.dqx}')
print(f'dqx mad diff:    {dqx_diff_mad}')
print(f'dqx mad nochrom: {twmad.summary.dq1}')
print(f'dqy xsuite diff: {dqy_diff_xs}')
print(f'dqy xsuite:      {tw0.dqy}')
print(f'dqy mad diff:    {dqy_diff_mad}')
print(f'dqy mad nochrom: {twmad.summary.dq2}')

assert np.isclose(tw0.dqx, twmad.summary.dq1, atol=0.2, rtol=0)
assert np.isclose(tw0.dqy, twmad.summary.dq2, atol=0.2, rtol=0)
assert np.isclose(tw0.qx, twmad.summary.q1, atol=1e-6, rtol=0)
assert np.isclose(tw0.qy, twmad.summary.q2, atol=1e-6, rtol=0)
assert np.isclose(tw0['betx', 'ip1'], tmad['betx', 'ip1:1'], atol=0, rtol=1e-5)
assert np.isclose(tw0['bety', 'ip1'], tmad['bety', 'ip1:1'], atol=0, rtol=1e-5)
assert np.isclose(tw0['betx', 'ip5'], tmad['betx', 'ip5:1'], atol=0, rtol=1e-5)
assert np.isclose(tw0['bety', 'ip5'], tmad['bety', 'ip5:1'], atol=0, rtol=1e-5)

plt.show()
