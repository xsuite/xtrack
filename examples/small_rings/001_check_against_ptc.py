from cpymad.madx import Madx
import numpy as np

import xtrack as xt

# machine = 'psb'
machine = 'ps'
# machine = 'elena'
machine = 'leir'
# machine = 'pimms'

test_data_folder = '../../test_data/'
mad = Madx()

if machine == 'leir':
    mad.call(test_data_folder + 'leir/leir_inj_nominal.beam')
    mad.input('BRHO = BEAM->PC * 3.3356 * (208./54.)')
    mad.call(test_data_folder + 'leir/leir.seq')
    mad.call(test_data_folder + 'leir/leir_inj_nominal.str')
    mad.use('leir')
    seq = mad.sequence.leir
    def_expr = False
elif machine == 'elena':
    mad.call(test_data_folder + 'elena/elena.seq')
    mad.call(test_data_folder + 'elena/highenergy.str')
    mad.call(test_data_folder + 'elena/highenergy.beam')
    mad.use('elena')
    seq = mad.sequence.elena
    def_expr = False
elif machine == 'psb':
    mad.call(test_data_folder + 'psb_chicane/psb.seq')
    mad.call(test_data_folder + 'psb_chicane/psb_fb_lhc.str')
    mad.input('beam, sequence=psb1, particle=proton, energy=1.098')
    mad.use('psb1')
    seq = mad.sequence.psb1
    def_expr = False
elif machine == 'ps':
    mad.call(test_data_folder + 'ps_sftpro/ps.seq')
    mad.call(test_data_folder + 'ps_sftpro/ps_hs_sftpro.str')
    mad.input('beam, particle=proton, pc = 14.0; BRHO = BEAM->PC * 3.3356;')
    mad.use('ps')
    seq = mad.sequence.ps
    def_expr = False
elif machine == 'pimms':
    mad.call(test_data_folder + 'pimms/PIMM.seq')
    mad.call(test_data_folder + 'pimms/betatron.str')
    mad.beam(particle='proton', gamma=1.05328945) # 50 MeV
    mad.use('pimms')
    seq = mad.sequence.pimms
    def_expr = True
else:
    raise ValueError(f'Unknown machine `{machine}`')


line = xt.Line.from_madx_sequence(seq, deferred_expressions=def_expr)
line.particle_ref = xt.Particles(gamma0=seq.beam.gamma,
                                 mass0=seq.beam.mass * 1e9,
                                 q0=seq.beam.charge)

# To see functions inside thick elements, we need to slice them
# line.slice_thick_elements(
#     slicing_strategies=[
#         xt.Strategy(slicing=None), # don't touch other elements
#         xt.Strategy(slicing=xt.Uniform(10, mode='thick'), element_type=xt.Bend),
#         xt.Strategy(slicing=xt.Uniform(5, mode='thick'), element_type=xt.Quadrupole),
#         xt.Strategy(slicing=xt.Uniform(5, mode='thick'), element_type=xt.CombinedFunctionMagnet),
#     ])

tt = line.get_table()
line.configure_bend_model(core='adaptive', edge='full')
# line.configure_bend_model(core='bend-kick-bend', edge='full')
# line.configure_bend_model(core='rot-kick-rot', edge='full', num_multipole_kicks=10)

tw = line.twiss(method='4d')

mad.input('twiss, chrom, table=twchr;')
twmad_ch = mad.table.twchr
mad.input('twiss, chrom=false, table=twnochr;')
twmad_nc = mad.table.twnochr

delta_chrom = 1e-4
mad.input(f'''
  ptc_create_universe;
  !ptc_create_layout, time=false, model=2, exact=true, method=6, nst=10;
  ptc_create_layout, time=false, model=1, exact=true, method=6, nst=10;
    select, flag=ptc_twiss, clear;
    select, flag=ptc_twiss, column=name,keyword,s,l,x,px,y,py,beta11,beta22,disp1,k1l;
    ptc_twiss, closed_orbit, icase=56, no=2, deltap=0, table=ptc_twiss,
               summary_table=ptc_twiss_summary, slice_magnets=true;
    ptc_twiss, closed_orbit, icase=56, no=2, deltap={delta_chrom:e}, table=ptc_twiss_pdp,
               summary_table=ptc_twiss_summary_pdp, slice_magnets=true;
    ptc_twiss, closed_orbit, icase=56, no=2, deltap={-delta_chrom:e}, table=ptc_twiss_mdp,
               summary_table=ptc_twiss_summary_mdp, slice_magnets=true;
  ptc_end;
''')

qx_ptc = mad.table.ptc_twiss.mu1[-1]
qy_ptc = mad.table.ptc_twiss.mu2[-1]
dq1_ptc = (mad.table.ptc_twiss_pdp.mu1[-1] - mad.table.ptc_twiss_mdp.mu1[-1]) / (2 * delta_chrom)
dq2_ptc = (mad.table.ptc_twiss_pdp.mu2[-1] - mad.table.ptc_twiss_mdp.mu2[-1]) / (2 * delta_chrom)

ddq1_ptc = (mad.table.ptc_twiss_pdp.mu1[-1] + mad.table.ptc_twiss_mdp.mu1[-1]
            - 2 * mad.table.ptc_twiss.mu1[-1]) / delta_chrom**2
ddq2_ptc = (mad.table.ptc_twiss_pdp.mu2[-1] + mad.table.ptc_twiss_mdp.mu2[-1]
            - 2 * mad.table.ptc_twiss.mu2[-1]) / delta_chrom**2

tptc = mad.table.ptc_twiss
tptc_p = mad.table.ptc_twiss_pdp
tptc_m = mad.table.ptc_twiss_mdp

fp = 1 + delta_chrom
fm = 1 - delta_chrom

# The MAD-X PTC interface rescales the beta functions with (1 + deltap)
# see: https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/eb495b4f926db53f3cd05133638860f910f42fe2/src/madx_ptc_twiss.f90#L1982
# We need to undo that
beta11_p = tptc_p.beta11 / fp
beta11_m = tptc_m.beta11 / fm
beta22_p = tptc_p.beta22 / fp
beta22_m = tptc_m.beta22 / fm
alfa11_p = tptc_p.alfa11
alfa11_m = tptc_m.alfa11
alfa22_p = tptc_p.alfa22
alfa22_m = tptc_m.alfa22

dx_ptc = (tptc_p.x - tptc_m.x) / (2 * delta_chrom)
dy_ptc = (tptc_p.y - tptc_m.y) / (2 * delta_chrom)

betx = 0.5 * (beta11_p + beta11_m)
bety = 0.5 * (beta22_p + beta22_m)
alfx = 0.5 * (alfa11_p + alfa11_m)
alfy = 0.5 * (alfa22_p + alfa22_m)
d_betx = (beta11_p - beta11_m) / (2 * delta_chrom)
d_bety = (beta22_p - beta22_m) / (2 * delta_chrom)
d_alfx = (alfa11_p - alfa11_m) / (2 * delta_chrom)
d_alfy = (alfa22_p - alfa22_m) / (2 * delta_chrom)

bx_ptc = d_betx / betx
by_ptc = d_bety / bety
ax_ptc = d_alfx - d_betx * alfx / betx
ay_ptc = d_alfy - d_bety * alfy / bety
wx_ptc = np.sqrt(ax_ptc**2 + bx_ptc**2)
wy_ptc = np.sqrt(ay_ptc**2 + by_ptc**2)

mad.input('''
  ptc_create_universe;
  ptc_create_layout, time=false, model=1, exact=true, method=6, nst=10;
  ptc_twiss, closed_orbit, icase=56, no=2, deltap=0, table=ptc_twiss,
              summary_table=ptc_twiss_summary, slice_magnets=true;
''')
delta_arr = np.linspace(-0.7e-4, 0.7e-4, 5)
tw_arr_ptc = []
for delta_chrom in delta_arr:
    mad.input(f'''

        select, flag=ptc_twiss, clear;
        select, flag=ptc_twiss, column=name,keyword,s,l,x,px,y,py,beta11,beta22,disp1,k1l;

        ptc_twiss, closed_orbit, icase=56, no=2, deltap={delta_chrom:e}, table=ptc_twiss_pdp,
                summary_table=ptc_twiss_summary_pdp, slice_magnets=true;
    ''')
    tw_arr_ptc.append(mad.table.ptc_twiss_pdp.dframe())

mad.input('''
  ptc_end;
''')

# Same for xsuite
tw_arr_xs = []
for delta_chrom in delta_arr:
    tw_arr_xs.append(line.twiss(method='4d', delta0=delta_chrom))

qx_ptc = mad.table.ptc_twiss.mu1[-1]
qy_ptc = mad.table.ptc_twiss.mu2[-1]

qx_vs_delta_ptc = np.array([tt.mu1[-1] for tt in tw_arr_ptc])
qy_vs_delta_ptc = np.array([tt.mu2[-1] for tt in tw_arr_ptc])

qx_vs_delta_xs = np.array([tt.mux[-1] for tt in tw_arr_xs])
qy_vs_delta_xs = np.array([tt.muy[-1] for tt in tw_arr_xs])

print(f'qx xsuite:          {tw.qx}')
print(f'qx ptc:             {qx_ptc}')
print(f'qx mad (chrom=F):   {twmad_nc.summary.q1}')
print()
print(f'qy xsuite:          {tw.qy}')
print(f'qy ptc:             {qy_ptc}')
print(f'qy mad (chrom=F):   {twmad_nc.summary.q2}')
print()
print(f'dqx xsuite:         {tw.dqx}')
print(f'dqx ptc:            {dq1_ptc}')
print(f'dqx mad (chrom=F):  {twmad_nc.summary.dq1 * seq.beam.beta}')
print()
print(f'dqy xsuite:         {tw.dqy}')
print(f'dqy ptc:            {dq2_ptc}')
print(f'dqy mad (chrom=F):  {twmad_nc.summary.dq2 * seq.beam.beta}')


import matplotlib.pyplot as plt
plt.close('all')
figsize = (6.4*1.3, 4.8*1.3)

fig_abx = plt.figure(101, figsize=figsize)

ax1 = plt.subplot(2,1,1)
plt.plot(tw.s, tw.ax_chrom, label='xsuite')
plt.plot(tptc.s, ax_ptc, '--', label='ptc')
plt.ylabel(r'$A_x$')
plt.legend(loc='best')

ax2 = plt.subplot(2,1,2, sharex=ax1)
plt.plot(tw.s, tw.bx_chrom)
plt.plot(tptc.s, bx_ptc, '--')
plt.ylabel(r'$B_x$')

fig_aby = plt.figure(111, figsize=figsize)
ax3 = plt.subplot(2,1,1, sharex=ax1)
plt.plot(tw.s, tw.ay_chrom)
plt.plot(tptc.s, ay_ptc, '--')
plt.ylabel(r'$A_y$')
plt.xlabel('s [m]')

ax4 = plt.subplot(2,1,2, sharex=ax1)
plt.plot(tw.s, tw.by_chrom)
plt.plot(tptc.s, by_ptc, '--')
plt.ylabel(r'$B_y$')
plt.xlabel('s [m]')

# Same for beta and Wxy
fig_bet = plt.figure(102, figsize=figsize)

ax21 = plt.subplot(2,1,1, sharex=ax1)
plt.plot(tw.s, tw.betx, label='xsuite')
plt.plot(tptc.s, tptc.beta11, '--', label='ptc')
# plt.plot(twmad_ch.s, twmad_ch.betx, 'r--', label='mad')
plt.ylabel(r'$\beta_x$')
plt.legend(loc='best')

ax22 = plt.subplot(2,1,2, sharex=ax1)
plt.plot(tw.s, tw.bety)
plt.plot(tptc.s, tptc.beta22, '--')
# plt.plot(twmad_ch.s, twmad_ch.bety, 'r--')
plt.ylabel(r'$\beta_y$')
plt.xlabel('s [m]')

fig_w = plt.figure(112, figsize=figsize)
ax23 = plt.subplot(2,1,1, sharex=ax1)
plt.plot(tw.s, tw.wx_chrom)
plt.plot(tptc.s, wx_ptc, '--')
# plt.plot(twmad_ch.s, twmad_ch.wx * tw.beta0, 'r--')
plt.ylabel(r'$W_x$')

ax24 = plt.subplot(2,1,2, sharex=ax1)
plt.plot(tw.s, tw.wy_chrom)
plt.plot(tptc.s, wy_ptc, '--')
# plt.plot(twmad_ch.s, twmad_ch.wy * tw.beta0, 'r--')
plt.ylabel(r'$W_y$')
plt.xlabel('s [m]')

# Same for orbit
fig_co = plt.figure(103, figsize=figsize)
ax31 = plt.subplot(2,1,1, sharex=ax1)
plt.plot(tw.s, tw.x * 1e3, label='xsuite')
plt.plot(tptc.s, tptc.x * 1e3, '--', label='ptc')
plt.ylabel('x [mm]')

ax32 = plt.subplot(2,1,2, sharex=ax1)
plt.plot(tw.s, tw.y * 1e3)
plt.plot(tptc.s, tptc.y * 1e3, '--')
plt.ylabel('y [mm]')
plt.xlabel('s [m]')

# Same for dispersion
fig_disp = plt.figure(104, figsize=figsize)
ax41 = plt.subplot(2,1,1, sharex=ax1)
plt.plot(tw.s, tw.dx, label='xsuite')
plt.plot(tptc.s, dx_ptc, '--', label='ptc')
plt.ylabel(r'$D_x$ [m]')

ax42 = plt.subplot(2,1,2, sharex=ax1)
plt.plot(tw.s, tw.dy)
plt.plot(tptc.s, dy_ptc, '--')
plt.ylabel(r'$D_y$ [m]')
plt.xlabel('s [m]')

tt = line.get_table()
tbends = tt.rows[tt.element_type == 'Bend']
tquads = tt.rows[tt.element_type == 'Quadrupole']
tcombined = tt.rows[tt.element_type == 'CombinedFunctionMagnet']
for ax in [ax1, ax2, ax3, ax4, ax21, ax22, ax23, ax24, ax31, ax32, ax41, ax42]:
    for nn in tbends.name:
        ax.axvspan(tbends['s', nn], tbends['s', nn] + line[nn].length, color='b',
                    alpha=0.2, lw=0)
    for nn in tquads.name:
        ax.axvspan(tquads['s', nn], tquads['s', nn] + line[nn].length, color='r',
                    alpha=0.2, lw=0)
    for nn in tcombined.name:
        ax.axvspan(tcombined['s', nn], tcombined['s', nn] + line[nn].length, color='g',
                    alpha=0.2, lw=0)


plt.figure(2, figsize=(6.4*1.4, 4.8*1.4))
ax1 = plt.subplot(211)
plt.plot(delta_arr*1e3, qx_vs_delta_ptc - dq1_ptc * delta_arr - qx_ptc, label='ptc')
plt.plot(delta_arr*1e3, qx_vs_delta_xs - tw.dqx * delta_arr - tw.qx, label='xsuite')
plt.ylabel(r'$\Delta Q_x$')
plt.legend()

ax2 = plt.subplot(212, sharex=ax1)
plt.plot(delta_arr*1e3, qy_vs_delta_ptc - dq2_ptc * delta_arr - qy_ptc)
plt.plot(delta_arr*1e3, qy_vs_delta_xs - tw.dqy * delta_arr - tw.qy)
plt.xlabel(r'$\delta$ [1e-3]')
plt.ylabel(r'$\Delta Q_y$')

plt.suptitle(f"PTC:    Qx = {qx_ptc:.4f}, Qy = {qy_ptc:.4f} Q'x={dq1_ptc:.4f}, Q'y={dq2_ptc:.4f}\n"
             f"Xsuite: Qx = {tw.qx:.4f}, Qy = {tw.qy:.4f} Q'x={tw.dqx:.4f}, Q'y={tw.dqy:.4f}\n"
             "Linear chromaticity removed from the plot")

title = f'{machine.upper()}' + r' - $\beta_0$' + f' = {tw.beta0:.4f}'
for fig in [fig_abx, fig_aby, fig_bet, fig_w, fig_co, fig_disp]:
    fig.suptitle(title)

plt.show()