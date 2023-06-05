from cpymad.madx import Madx
import xtrack as xt
import xpart as xp

# hllhc15 can be found at git@github.com:lhcopt/hllhc15.git

thin = False
kill_fringes_and_edges = True

mad = Madx()

mad.input(f"""
call,file="../../../hllhc15/util/lhc.seq";
call,file="../../../hllhc15/hllhc_sequence.madx";
call,file="../../../hllhc15/toolkit/macro.madx";
seqedit,sequence=lhcb1;flatten;cycle,start=IP7;flatten;endedit;
seqedit,sequence=lhcb2;flatten;cycle,start=IP7;flatten;endedit;
exec,mk_beam(7000);
call,file="../../../hllhc15/round/opt_round_150_1500.madx";
{('exec,myslice;' if thin else '')}
exec,check_ip(b1);
exec,check_ip(b2);
// acbv11.r8b1 = 2e-6;
""")

mad.use(sequence="lhcb1")
seq = mad.sequence.lhcb1
mad.twiss()

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1, allow_thick=True, deferred_expressions=True)
line.particle_ref = xp.Particles(mass0=seq.beam.mass*1e9, gamma0=seq.beam.gamma)
line.twiss_default['method'] = '4d'
line.twiss_default['matrix_stability_tol'] = 100


line.config.XTRACK_USE_EXACT_DRIFTS = True
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
            ee.r21 = 0
            ee.r43 = 0

tw_mad = mad.twiss().dframe()

delta1 = -1e-5
delta2 = 1e-5

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

print(f'Config: {thin=} {kill_fringes_and_edges=} \n' )
print(f'dqx xsuite diff: {dqx_diff_xs}')
print(f'dqx xsuite:      {tw0.dqx}')
print(f'dqx mad diff:    {dqx_diff_mad}')
print(f'dqx mad nochrom: {twmad.summary.dq1}')
print(f'dqy xsuite diff: {dqy_diff_xs}')
print(f'dqy xsuite:      {tw0.dqy}')
print(f'dqy mad diff:    {dqy_diff_mad}')
print(f'dqy mad nochrom: {twmad.summary.dq2}')

plt.show()

# def twiss_with_dp(line, dp):
#     for ee in line.elements:
#         if isinstance(ee, xt.CombinedFunctionMagnet):
#             ee.k0 /= (1 + dp)
#             ee.k1 /= (1 + dp)
#         if isinstance(ee, xt.Multipole):
#             ee.knl /= (1 + dp)
#             ee.ksl /= (1 + dp)
#         if isinstance(ee, xt.DipoleEdge):
#             ee.r21 /= (1 + dp)
#             ee.r43 /= (1 + dp)
#     res = line.twiss()
#     for ee in line.elements:
#         if isinstance(ee, xt.CombinedFunctionMagnet):
#             ee.k0 *= (1 + dp)
#             ee.k1 *= (1 + dp)
#         if isinstance(ee, xt.Multipole):
#             ee.knl *= (1 + dp)
#             ee.ksl *= (1 + dp)
#         if isinstance(ee, xt.DipoleEdge):
#             ee.r21 *= (1 + dp)
#             ee.r43 *= (1 + dp)
#     return res

# tw1 = twiss_with_dp(line, dp=-1e-6)
# tw2 = twiss_with_dp(line, dp=1e-6)



# betx0 = tw0.betx[0]
# bety0 = tw0.bety[0]
# alfx0 = tw0.alfx[0]
# alfy0 = tw0.alfy[0]

# mad.input(f"twiss, betx={betx0}, bety={bety0}, alfx={alfx0}, alfy={alfy0};")
# tw_mad0 = mad.table.twiss.dframe()

# mad.input(f"twiss, betx={betx0}, bety={bety0}, alfx={alfx0}, alfy={alfy0}, deltap={delta1};")
# tw_mad1 = mad.table.twiss.dframe()

# mad.input(f"twiss, betx={betx0}, bety={bety0}, alfx={alfx0}, alfy={alfy0}, deltap={delta2};")
# tw_mad2 = mad.table.twiss.dframe()

# dqx_diff_mad = (tw_mad2.mux[-1] - tw_mad1.mux[-1]) / (delta2 - delta1)


# twiss_init=tw0.get_twiss_init(at_element=0)
# twiss_init.particle_on_co.delta = delta1
# tw1 = line.twiss(twiss_init=twiss_init)
# twiss_init.particle_on_co.delta = delta2
# tw2 = line.twiss(delta0=delta2, twiss_init=twiss_init)
