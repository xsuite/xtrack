import xtrack as xt
import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

line = xt.Line.from_json('fccee_z_thick.json')
line.cycle('ip.4', inplace=True)

tt = line.get_table()
bz_data_file = './z_fieldmaps/Koratsinos_Bz_closed_before_quads.dat'

line.vars['voltca1'] = 0

import pandas as pd
bz_df = pd.read_csv(bz_data_file, sep='\s+', skiprows=1, names=['z', 'Bz'])

l_solenoid = 4.4
ds_sol_start = -2.2
ds_sol_end = 2.2
ip_sol = 'ip.1'

theta_tilt = 15e-3 # rad

s_sol_slices = np.linspace(-2.2, 2.2, 1001)
bz_sol_slices = np.interp(s_sol_slices, bz_df.z, bz_df.Bz)
bz_sol_slices[0] = 0

P0_J = line.particle_ref.p0c[0] * qe / clight
brho = P0_J / qe / line.particle_ref.q0
ks = 0.5 * (bz_sol_slices[:-1] + bz_sol_slices[1:]) / brho
l_sol_slices = np.diff(s_sol_slices)
s_sol_slices_entry = s_sol_slices[:-1]

sol_slices = []
for ii in range(len(s_sol_slices_entry)):
    sol_slices.append(xt.Solenoid(length=l_sol_slices[ii], ks=0)) # Off for now

s_ip = tt['s', ip_sol]

line.discard_tracker()
line.insert_element(name='sol_start_'+ip_sol, element=xt.Marker(),
                    at_s=s_ip + ds_sol_start)
line.insert_element(name='sol_end_'+ip_sol, element=xt.Marker(),
                    at_s=s_ip + ds_sol_end)

sol_start_tilt = xt.YRotation(angle=-theta_tilt * 180 / np.pi)
sol_end_tilt = xt.YRotation(angle=+theta_tilt * 180 / np.pi)
sol_start_shift = xt.XYShift(dx=l_solenoid/2 * np.sin(theta_tilt))
sol_end_shift = xt.XYShift(dx=l_solenoid/2 * np.sin(theta_tilt))

line.element_dict['sol_start_tilt_'+ip_sol] = sol_start_tilt
line.element_dict['sol_end_tilt_'+ip_sol] = sol_end_tilt
line.element_dict['sol_start_shift_'+ip_sol] = sol_start_shift
line.element_dict['sol_end_shift_'+ip_sol] = sol_end_shift


sol_slice_names = []
for ii in range(len(s_sol_slices_entry)):
    nn = f'sol_slice_{ii}_{ip_sol}'
    line.element_dict[nn] = sol_slices[ii]
    sol_slice_names.append(nn)

tt = line.get_table()
names_upstream = list(tt.rows[:'sol_start_'+ip_sol].name)
names_downstream = list(tt.rows['sol_end_'+ip_sol:].name[:-1]) # -1 to exclude '_end_point' added by the table

element_names = (names_upstream
                 + ['sol_start_tilt_'+ip_sol, 'sol_start_shift_'+ip_sol]
                 + sol_slice_names
                 + ['sol_end_shift_'+ip_sol, 'sol_end_tilt_'+ip_sol]
                 + names_downstream)

line.element_names = element_names

# re-insert the ip
line.element_dict.pop(ip_sol)
line.insert_element(name=ip_sol, element=xt.Marker(), at_s=s_ip)

# Add dipole correctors
line.insert_element(name='mcb1.r1', element=xt.Multipole(knl=[0]),
                    at='qc1r1.1_entry')
line.insert_element(name='mcb2.r1', element=xt.Multipole(knl=[0]),
                    at='qc1r1.1_exit')
line.insert_element(name='mcb1.l1', element=xt.Multipole(knl=[0]),
                    at='qc1l1.4_exit')
line.insert_element(name='mcb2.l1', element=xt.Multipole(knl=[0]),
                    at='qc1l1.4_entry')

line.vars['acb1h.r1'] = 0
line.vars['acb1v.r1'] = 0
line.vars['acb2h.r1'] = 0
line.vars['acb2v.r1'] = 0
line.vars['acb1h.l1'] = 0
line.vars['acb1v.l1'] = 0
line.vars['acb2h.l1'] = 0
line.vars['acb2v.l1'] = 0

line.element_refs['mcb1.r1'].knl[0] = line.vars['acb1h.r1']
line.element_refs['mcb2.r1'].knl[0] = line.vars['acb2h.r1']
line.element_refs['mcb1.r1'].ksl[0] = line.vars['acb1v.r1']
line.element_refs['mcb2.r1'].ksl[0] = line.vars['acb2v.r1']
line.element_refs['mcb1.l1'].knl[0] = line.vars['acb1h.l1']
line.element_refs['mcb2.l1'].knl[0] = line.vars['acb2h.l1']
line.element_refs['mcb1.l1'].ksl[0] = line.vars['acb1v.l1']
line.element_refs['mcb2.l1'].ksl[0] = line.vars['acb2v.l1']

line.build_tracker()

# Set strength
line.vars['on_sol_'+ip_sol] = 0
for ii in range(len(s_sol_slices_entry)):
    nn = f'sol_slice_{ii}_{ip_sol}'
    line.element_refs[nn].ks = ks[ii] * line.vars['on_sol_'+ip_sol]

tt = line.get_table()

tt.rows['sol_start_ip.1':'sol_end_ip.1'].show()

tw_sol_off = line.twiss(method='4d')
line.vars['on_sol_ip.1'] = 1
tw_sol_on = line.twiss(method='4d')

tw_local = line.twiss(start='ip.7', end='ip.2', init_at='ip.1',
                      init=tw_sol_off)



line.vars['ks0.r1'] = 0
line.vars['ks1.r1'] = 0
line.vars['ks2.r1'] = 0
line.vars['ks3.r1'] = 0
line.vars['ks4.r1'] = 0
line.vars['ks0.l1'] = 0
line.vars['ks1.l1'] = 0
line.vars['ks2.l1'] = 0
line.vars['ks3.l1'] = 0
line.vars['ks4.l1'] = 0

line.element_refs['qc1r1.1'].k1s = line.vars['ks0.r1']
line.element_refs['qc2r1.1'].k1s = line.vars['ks1.r1']
line.element_refs['qc2r2.1'].k1s = line.vars['ks2.r1']
line.element_refs['qc1r2.1'].k1s = line.vars['ks3.r1']

line.element_refs['qc1l1.4'].k1s = line.vars['ks0.l1']
line.element_refs['qc2l1.4'].k1s = line.vars['ks1.l1']
line.element_refs['qc2l2.4'].k1s = line.vars['ks2.l1']
line.element_refs['qc1l2.4'].k1s = line.vars['ks3.l1']

line.vars['corr_k0.r1'] = 0
line.vars['corr_k1.r1'] = 0
line.vars['corr_k2.r1'] = 0
line.vars['corr_k3.r1'] = 0
line.vars['corr_k4.r1'] = 0
line.vars['corr_k0.l1'] = 0
line.vars['corr_k1.l1'] = 0
line.vars['corr_k2.l1'] = 0
line.vars['corr_k3.l1'] = 0
line.vars['corr_k4.l1'] = 0

line.element_refs['qc1r1.1'].k1 += line.vars['corr_k0.r1']
line.element_refs['qc2r1.1'].k1 += line.vars['corr_k1.r1']
line.element_refs['qc2r2.1'].k1 += line.vars['corr_k2.r1']
line.element_refs['qc1r2.1'].k1 += line.vars['corr_k3.r1']
line.element_refs['qc1l1.4'].k1 += line.vars['corr_k0.l1']
line.element_refs['qc2l1.4'].k1 += line.vars['corr_k1.l1']
line.element_refs['qc2l2.4'].k1 += line.vars['corr_k2.l1']
line.element_refs['qc1l2.4'].k1 += line.vars['corr_k3.l1']

opt_l = line.match(
    solve=False,
    method='4d',
    start='pqc2le.4',
    end='ip.1',
    init=tw_sol_off,
    init_at=xt.START,
    vary=[
        xt.VaryList(['acb1h.l1', 'acb2h.l1','acb1v.l1', 'acb2v.l1'], step=1e-8, tag='corr_l'),
        xt.VaryList(['ks1.l1', 'ks2.l1', 'ks3.l1', 'ks0.l1'], step=1e-7, tag='skew_l'),
        xt.VaryList(['corr_k1.l1', 'corr_k2.l1', 'corr_k3.l1', 'corr_k0.l1'], step=1e-6, tag='normal_l'),
    ],
    targets=[

        xt.TargetSet(['x', 'px', 'y', 'py'], value=tw_sol_off, at='ip.1', tag='orbit'),

        xt.TargetRmatrix(
                    r13=0, r14=0, r23=0, r24=0, # Y-X block
                    r31=0, r32=0, r41=0, r42=0, # X-Y block,
                    start='pqc2le.4', end='ip.1', tol=1e-8, tag='coupl'),

        xt.Target('mux', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
        xt.Target('muy', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
        xt.Target('betx', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=1, tol=1e-5),
        xt.Target('bety', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=10, tol=1e-7),
        xt.Target('alfx', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),
        xt.Target('alfy', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),

    ]
)

# Orbit alone
opt_l.disable_all_targets(); opt_l.disable_all_vary()
opt_l.enable_targets(tag='orbit'); opt_l.enable_vary(tag='corr_l')
opt_l.solve()

# Coupling alone
opt_l.disable_all_targets(); opt_l.disable_all_vary()
opt_l.enable_targets(tag='coupl'); opt_l.enable_vary(tag='skew_l')
opt_l.solve()

# Orbit and coupling
opt_l.enable_targets(tag='orbit'); opt_l.enable_vary(tag='corr_l')
opt_l.solve()

# Phase advance alone
opt_l.disable_all_targets(); opt_l.disable_all_vary()
opt_l.enable_targets(tag='mu_ip'); opt_l.enable_vary(tag='normal_l')
opt_l.solve()

# Coupling alone
opt_l.disable_all_targets(); opt_l.disable_all_vary()
opt_l.enable_targets(tag='coupl'); opt_l.enable_vary(tag='skew_l')
opt_l.step(25)

# Combine phase, coupling and orbit
opt_l.enable_targets(tag='coupl'); opt_l.enable_vary(tag='skew_l')
opt_l.enable_targets(tag='orbit'); opt_l.enable_vary(tag='corr_l')
opt_l.solve()

# Add targets on ip beta
opt_l.enable_targets(tag='bet_ip')
opt_l.solve()

# Add targets on ip alfa (--> all targets active)
opt_l.enable_targets(tag='alf_ip')
opt_l.solve()

opt_r = line.match(
    solve=False,
    method='4d',
    start='ip.1',
    end='pqc2re.1',
    init=tw_sol_off,
    init_at=xt.END,
    vary=[
        xt.VaryList(['acb1h.r1', 'acb2h.r1','acb1v.r1', 'acb2v.r1'], step=1e-8, tag='corr_r'),
        xt.VaryList(['ks1.r1', 'ks2.r1', 'ks3.r1', 'ks0.r1'], step=1e-7, tag='skew_r'),
        xt.VaryList(['corr_k1.r1', 'corr_k2.r1', 'corr_k3.r1', 'corr_k0.r1'], step=1e-6, tag='normal_r'),
    ],
    targets=[

        xt.TargetSet(['x', 'px', 'y', 'py'], value=tw_sol_off, at='ip.1', tag='orbit'),

        xt.TargetRmatrix(r13=0, r14=0, r23=0, r24=0, # Y-X block
                         r31=0, r32=0, r41=0, r42=0, # X-Y block,
                         start='ip.1', end='pqc2re.1', tol=1e-8, tag='coupl'),

        xt.Target('mux', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
        xt.Target('muy', value=tw_sol_off, at='ip.1', tag='mu_ip', weight=0.1, tol=1e-6),
        xt.Target('betx', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=1, tol=1e-5),
        xt.Target('bety', value=tw_sol_off, at='ip.1', tag='bet_ip', weight=10, tol=1e-6),
        xt.Target('alfx', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),
        xt.Target('alfy', value=tw_sol_off, at='ip.1', tag='alf_ip', weight=0.1, tol=1e-4),

    ]
)

# Orbit alone
opt_r.disable_all_targets(); opt_r.disable_all_vary()
opt_r.enable_targets(tag='orbit'); opt_r.enable_vary(tag='corr_r')
opt_r.solve()

# Coupling alone
opt_r.disable_all_targets(); opt_r.disable_all_vary()
opt_r.enable_targets(tag='coupl'); opt_r.enable_vary(tag='skew_r')
opt_r.step(25)

# Orbit and coupling
opt_r.enable_targets(tag='orbit'); opt_r.enable_vary(tag='corr_r')
opt_r.solve()

# Phase advance alone
opt_r.disable_all_targets(); opt_r.disable_all_vary()
opt_r.enable_targets(tag='mu_ip'); opt_r.enable_vary(tag='normal_r')
opt_r.solve()

# Coupling alone
opt_r.disable_all_targets(); opt_r.disable_all_vary()
opt_r.enable_targets(tag='coupl'); opt_r.enable_vary(tag='skew_r')
opt_r.step(25)

# Combine phase, coupling and orbit
opt_r.enable_targets(tag='coupl'); opt_r.enable_vary(tag='skew_r')
opt_r.enable_targets(tag='orbit'); opt_r.enable_vary(tag='corr_r')
opt_r.solve()

# Add targets on ip beta
opt_r.enable_targets(tag='bet_ip')
opt_r.solve()

# Add targets on ip alfa (--> all targets active)
opt_r.enable_targets(tag='alf_ip')
opt_r.solve()

tw_local_corr = line.twiss(start='ip.4', end='_end_point', init_at='ip.1',
                            init=tw_sol_off)

tw_local_corr_back = line.twiss(start='ip.4', end='_end_point', init_at='ip.4',
                                init=tw_local_corr)


line.to_json('fccee_z_thick_with_sol_corrected.json')


# plot
import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(bz_df.z, bz_df.Bz, label='Bz')
plt.plot(s_sol_slices, bz_sol_slices, '.-', label='slices')
plt.xlabel('z [m]')
plt.ylabel('Bz [T]')
plt.grid()

plt.figure(2)
ax1 = plt.subplot(2, 1, 1)
plt.plot(tw_local.s, tw_local.x*1e3, label='x')
plt.plot(tw_local_corr.s, tw_local_corr.x*1e3, label='x corr')
plt.ylabel('x [mm]')

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw_local.s, tw_local.y*1e3, label='y')
plt.plot(tw_local_corr.s, tw_local_corr.y*1e3, label='y corr')

plt.xlabel('s [m]')
plt.ylabel('y [mm]')

plt.figure(3)
ax1 = plt.subplot(2, 1, 1)
plt.plot(tw_local.s, tw_local.betx2)
plt.plot(tw_local_corr.s, tw_local_corr.betx2)
plt.plot(tw_local_corr_back.s, tw_local_corr_back.betx2)
plt.ylabel(r'$\beta_{x,2}$ [m]')

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw_local.s, tw_local.bety1)
plt.plot(tw_local_corr.s, tw_local_corr.bety1)
plt.plot(tw_local_corr_back.s, tw_local_corr_back.bety1)
plt.ylabel(r'$\beta_{y,1}$ [m]')

plt.xlabel('s [m]')


tw_sol_on_corrected = line.twiss(method='4d')

plt.figure(5)
ax1 = plt.subplot(2, 1, 1)
plt.plot(tw_sol_off.s, tw_sol_on.betx2, label='betx2 off')
plt.plot(tw_sol_off.s, tw_sol_on_corrected.betx2, label='betx2 on')
plt.ylabel(r'$\beta_{x,2}$ [m]')
plt.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw_sol_off.s, tw_sol_on.bety1, label='bety1 off')
plt.plot(tw_sol_off.s, tw_sol_on_corrected.bety1, label='bety1 on')
plt.ylabel(r'$\beta_{y,1}$ [m]')



plt.show()