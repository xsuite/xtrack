import xtrack as xt
from cpymad.madx import Madx

mad = Madx()

mad.input("""
call,file="../../test_data/hllhc15_thick/lhc.seq";
call,file="../../test_data/hllhc15_thick/hllhc_sequence.madx";
call,file="../../test_data/hllhc15_thick/macro.madx";
call,file="../../test_data/hllhc15_thick/enable_crabcavities.madx";
call,file="../../test_data/hllhc15_thick/opt_round_150_1500.madx";

seqedit,sequence=lhcb1;flatten;cycle,start=IP3;flatten;endedit;
seqedit,sequence=lhcb2;flatten;cycle,start=IP3;flatten;endedit;

exec,mk_beam(7000);
call,file="../../test_data/hllhc15_thick/opt_round_150_1500.madx";
exec,check_ip(b1);
exec,check_ip(b2);
""")

mad.input('''
on_crab1 = -190;
on_crab5 = -190;
on_x1 = 0;
on_x5 = 0;
on_disp = 0;
vrf400 = 0;
lagrf400.b1' = 0.5;
lagrf400.b2' = 0.;

use, sequence=lhcb1;
use, sequence=lhcb2;

save, sequence=lhcb1,lhcb2, file="saved_b1b2.madx";

''')

mad.use('lhcb1')
tw_b1 = xt.Table(mad.twiss(), _copy_cols=True)
tw_b1_t_plus = xt.Table(mad.twiss(t=1e-4), _copy_cols=True)
tw_b1t_minus = xt.Table(mad.twiss(t=-1e-4), _copy_cols=True)

tw_b1['dx_dz'] = (tw_b1_t_plus['x'] - tw_b1t_minus['x'])/(tw_b1_t_plus['t'] - tw_b1t_minus['t'])
tw_b1['dy_dz'] = (tw_b1_t_plus['y'] - tw_b1t_minus['y'])/(tw_b1_t_plus['t'] - tw_b1t_minus['t'])

mad.use('lhcb2')
tw_b2 = xt.Table(mad.twiss(), _copy_cols=True)
tw_b2_t_plus = xt.Table(mad.twiss(t=1e-4), _copy_cols=True)
tw_b2t_minus = xt.Table(mad.twiss(t=-1e-4), _copy_cols=True)

tw_b2['dx_dz'] = (tw_b2_t_plus['x'] - tw_b2t_minus['x'])/(tw_b2_t_plus['t'] - tw_b2t_minus['t'])
tw_b2['dy_dz'] = (tw_b2_t_plus['y'] - tw_b2t_minus['y'])/(tw_b2_t_plus['t'] - tw_b2t_minus['t'])

print('IP1 - MADX')
print(f'B1 - dx_dz: {tw_b1["dx_dz", "ip1:1"]}')
print(f'B2 - dx_dz: {tw_b2["dx_dz", "ip1:1"]}')
print('IP5 - MADX')
print(f'B1 - dy_dz: {tw_b1["dy_dz", "ip5:1"]}')
print(f'B2 - dy_dz: {tw_b2["dy_dz", "ip5:1"]}')

line_b1 = xt.Line.from_madx_sequence(mad.sequence['lhcb1'], deferred_expressions=True)
line_b4 = xt.Line.from_madx_sequence(mad.sequence['lhcb2'], deferred_expressions=True)

line_b1.particle_ref = xt.Particles(p0c=7e12)
line_b4.particle_ref = xt.Particles(p0c=7e12)

tw_xs_b1 = line_b1.twiss4d()
tw_xs_b4 = line_b4.twiss4d()
tw_xs_b2 = tw_xs_b4.reverse()

print('IP1 - Xsuite')
print(f'B1 - dx_zeta: {tw_xs_b1["dx_zeta", "ip1"]}')
print(f'B2 - dx_zeta: {tw_xs_b2["dx_zeta", "ip1"]}')
print(f'B4 - dx_zeta: {tw_xs_b4["dx_zeta", "ip1"]}')

print('IP5 - Xsuite')
print(f'B1 - dy_zeta: {tw_xs_b1["dy_zeta", "ip5"]}')
print(f'B2 - dy_zeta: {tw_xs_b2["dy_zeta", "ip5"]}')
print(f'B4 - dy_zeta: {tw_xs_b4["dy_zeta", "ip5"]}')

mad_b4 = Madx()
mad_b4.input("""
mylhcbeam=4;
call,file="../../test_data/hllhc15_thick/lhcb4.seq";
call,file="../../test_data/hllhc15_thick/hllhc_sequence.madx";
call,file="../../test_data/hllhc15_thick/macro.madx";
call,file="../../test_data/hllhc15_thick/enable_crabcavities.madx";
call,file="../../test_data/hllhc15_thick/opt_round_150_1500.madx";

seqedit,sequence=lhcb2;flatten;cycle,start=IP3;flatten;endedit;

exec,mk_beam(7000);
call,file="../../test_data/hllhc15_thick/opt_round_150_1500.madx";
exec,check_ip(b2);
""")

mad_b4.input('''
on_crab1 = -190;
on_crab5 = -190;
on_x1 = 0;
on_x5 = 0;
on_disp = 0;
vrf400 = 0;
lagrf400.b1' = 0.5;
lagrf400.b2' = 0.;

use, sequence=lhcb2;

save, sequence=lhcb2, file="saved_b4.madx";

''')

line_b4_mad = xt.Line.from_madx_sequence(mad_b4.sequence['lhcb2'], deferred_expressions=True)
line_b4_mad.particle_ref = xt.Particles(p0c=7e12)

tw_b4_mad = line_b4_mad.twiss4d()
tw_b2_mad = tw_b4_mad.reverse()

env_b1b2 = xt.load("saved_b1b2.madx")
env_b1b2.lhcb1.particle_ref = xt.Particles(p0c=7e12)
env_b1b2.lhcb2.particle_ref = xt.Particles(p0c=7e12)
tw_b1_b1b2 = env_b1b2.lhcb1.twiss4d()
tw_b2_b1b2 = env_b1b2.lhcb2.twiss4d()
tw_b4_b1b2 = tw_b2_b1b2.reverse()

env_b4 = xt.load("saved_b4.madx")
env_b4.lhcb2.particle_ref = xt.Particles(p0c=7e12)
tw_b4_b4 = env_b4.lhcb2.twiss4d()
tw_b2_b4 = tw_b4_b4.reverse()

print('------- IP1 - Xsuite -------')
print(f'B1 (loaded as b1 cpymad) - dx_zeta: {tw_xs_b1["dx_zeta", "ip1"]}')
print(f'B1 (loaded as b1 parsed) - dx_zeta: {tw_b1_b1b2["dx_zeta", "ip1"]}')
print()
print(f'B2 (loaded as b2 cpymad) - dx_zeta: {tw_xs_b2["dx_zeta", "ip1"]}')
print(f'B2 (loaded as b4 cpymad) - dx_zeta: {tw_b2_mad["dx_zeta", "ip1"]}')
print(f'B2 (loaded as b2 parsed) - dx_zeta: {tw_b2_b1b2["dx_zeta", "ip1"]}')
print(f'B2 (loaded as b4 parsed) - dx_zeta: {tw_b2_b4["dx_zeta", "ip1"]}')
print()
print(f'B4 (loaded as b2 cpymad) - dx_zeta: {tw_xs_b4["dx_zeta", "ip1"]}')
print(f'B4 (loaded as b4 cpymad) - dx_zeta: {tw_b4_mad["dx_zeta", "ip1"]}')
print(f'B4 (loaded as b2 parsed) - dx_zeta: {tw_b4_b1b2["dx_zeta", "ip1"]}')
print(f'B4 (loaded as b4 parsed) - dx_zeta: {tw_b4_b4["dx_zeta", "ip1"]}')
print()
print('------- IP5 - Xsuite -------')
print(f'B1 (loaded as b1 cpymad) - dy_zeta: {tw_xs_b1["dy_zeta", "ip5"]}')
print(f'B1 (loaded as b1 parsed) - dy_zeta: {tw_b1_b1b2["dy_zeta", "ip5"]}')
print()
print(f'B2 (loaded as b2 cpymad) - dy_zeta: {tw_xs_b2["dy_zeta", "ip5"]}')
print(f'B2 (loaded as b4 cpymad) - dy_zeta: {tw_b2_mad["dy_zeta", "ip5"]}')
print(f'B2 (loaded as b2 parsed) - dy_zeta: {tw_b2_b1b2["dy_zeta", "ip5"]}')
print(f'B2 (loaded as b4 parsed) - dy_zeta: {tw_b2_b4["dy_zeta", "ip5"]}')
print()
print(f'B4 (loaded as b2 cpymad) - dy_zeta: {tw_xs_b4["dy_zeta", "ip5"]}')
print(f'B4 (loaded as b4 cpymad) - dy_zeta: {tw_b4_mad["dy_zeta", "ip5"]}')
print(f'B4 (loaded as b2 parsed) - dy_zeta: {tw_b4_b1b2["dy_zeta", "ip5"]}')
print(f'B4 (loaded as b4 parsed) - dy_zeta: {tw_b4_b4["dy_zeta", "ip5"]}')