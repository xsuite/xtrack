import numpy as np
import xtrack as xt
import xobjects as xo
from scipy.constants import c as clight
from scipy.constants import e as qe

from cpymad.madx import Madx

fname = 'fccee_z'; pc_gev = 45.6

env = xt.load_madx_lattice('../../test_data/fcc_ee/' + fname + '.seq')
line = env.fccee_p_ring

line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV,
                                 p0c=pc_gev*1e9)
line.cycle('ip.4', inplace=True)
line.append_element(element=xt.Marker(), name='ip.4.l')
line.build_tracker()

# Get table with the elements
tt = line.get_table()

# Save voltage values
line.vars['voltca1_ref'] = line.vv['voltca1']
if 'voltca2' in line.vars.keys():
    line.vars['voltca2_ref'] = line.vv['voltca2']
else:
    line.vars['voltca2_ref'] = 0

# Switch off the cavities
line.vars['voltca1'] = 0
line.vars['voltca2'] = 0

# Load solenoid Bz data
bz_data_file = '../../test_data/fcc_ee/Bz_closed_before_quads.dat'
import pandas as pd
bz_df = pd.read_csv(bz_data_file, sep='\s+', skiprows=1, names=['z', 'Bz'])

# Solenoid parameters
theta_tilt = 15e-3 # rad
l_beam = 4.4       # Length of the solenoid along the beam trajectory
l_solenoid = l_beam * np.cos(theta_tilt)
ds_sol_start = -l_beam / 2 # from ip
ds_sol_end = +l_beam / 2   # from ip
ip_sol = 'ip.1'

# Prepare solenoid slices
s_sol_slices = np.linspace(-l_solenoid/2, l_solenoid/2, 1001)
bz_sol_slices = np.interp(s_sol_slices, bz_df.z, bz_df.Bz)
bz_sol_slices[0] = 0
bz_sol_slices[-1] = 0

# Compute strength of solenoid slices
P0_J = line.particle_ref.p0c[0] * qe / clight
brho = P0_J / qe / line.particle_ref.q0
ks = 0.5 * (bz_sol_slices[:-1] + bz_sol_slices[1:]) / brho

# Length and entry position of solenoid slices
l_sol_slices = np.diff(s_sol_slices)
s_sol_slices_entry = s_sol_slices[:-1]

# Build the slices
sol_slices = []
for ii in range(len(s_sol_slices_entry)):
    sol_slices.append(xt.Solenoid(length=l_sol_slices[ii], ks=0)) # Off for now

s_ip = tt['s', ip_sol]

# Install markers at solenoid entry and exit
line.discard_tracker()
line.insert('sol_start_'+ip_sol, xt.Marker(), at=s_ip + ds_sol_start)
line.insert('sol_end_'+ip_sol, xt.Marker(), at=s_ip + ds_sol_end)

# Build reference frame transformations at ebtry and exit of solenoid
sol_start_tilt = xt.YRotation(angle=-theta_tilt * 180 / np.pi)
sol_end_tilt = xt.YRotation(angle=+theta_tilt * 180 / np.pi)
sol_start_shift = xt.XYShift(dx=l_solenoid/2 * np.tan(theta_tilt))
sol_end_shift = xt.XYShift(dx=l_solenoid/2 * np.tan(theta_tilt))

line.element_dict['sol_start_tilt_'+ip_sol] = sol_start_tilt
line.element_dict['sol_end_tilt_'+ip_sol] = sol_end_tilt
line.element_dict['sol_start_shift_'+ip_sol] = sol_start_shift
line.element_dict['sol_end_shift_'+ip_sol] = sol_end_shift

line.element_dict['sol_entry_'+ip_sol] = xt.Solenoid(length=0, ks=0)
line.element_dict['sol_exit_'+ip_sol] = xt.Solenoid(length=0, ks=0)
line.element_dict['sol_zeta_shift_'+ip_sol] = xt.ZetaShift(dzeta=-(l_beam - l_solenoid))

# Add slices to the elements pot
sol_slice_names = []
sol_slice_names.append('sol_entry_'+ip_sol)
for ii in range(len(s_sol_slices_entry)):
    nn = f'sol_slice_{ii}_{ip_sol}'
    line.element_dict[nn] = sol_slices[ii]
    sol_slice_names.append(nn)
sol_slice_names.append('sol_exit_'+ip_sol)

# Insert new elements in the names list
tt = line.get_table()
names_upstream = list(tt.rows[:'sol_start_'+ip_sol].name)
names_downstream = list(tt.rows['sol_end_'+ip_sol:].name[:-1]) # -1 to exclude '_end_point' added by the table
element_names = (names_upstream
                 + ['sol_start_tilt_'+ip_sol, 'sol_start_shift_'+ip_sol]
                 + sol_slice_names
                 + ['sol_end_shift_'+ip_sol, 'sol_end_tilt_'+ip_sol]
                 + ['sol_zeta_shift_'+ip_sol]
                 + names_downstream)
line.element_names = element_names

# re-insert the ip
line.element_dict.pop(ip_sol)
tt = line.get_table()
line.insert(ip_sol, xt.Marker(),
        at = 0.5 * (tt['s', 'sol_start_'+ip_sol] + tt['s', 'sol_end_'+ip_sol]))

line.build_tracker()

# Set strength
line.vars['on_sol_'+ip_sol] = 0
for ii in range(len(s_sol_slices_entry)):
    nn = f'sol_slice_{ii}_{ip_sol}'
    line.element_refs[nn].ks = ks[ii] * line.vars['on_sol_'+ip_sol]

tt = line.get_table()

# Inspect
tt.rows['sol_start_ip.1':'sol_end_ip.1'].show()

# Add skew quadrupole knobs to the final focus quadrupoles
line.vars['on_corr_ip.1'] = 1
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

line.element_refs['qc1r1.1'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks0.r1']
line.element_refs['qc2r1.1'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks1.r1']
line.element_refs['qc2r2.1'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks2.r1']
line.element_refs['qc1r2.1'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks3.r1']
line.element_refs['qc1l1.4'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks0.l1']
line.element_refs['qc2l1.4'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks1.l1']
line.element_refs['qc2l2.4'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks2.l1']
line.element_refs['qc1l2.4'].k1s = line.vars['on_corr_ip.1'] * line.vars['ks3.l1']

# Add correction knobs normal strength of the final focus quadrupoles
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

line.element_refs['qc1r1.1'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k0.r1']
line.element_refs['qc2r1.1'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k1.r1']
line.element_refs['qc2r2.1'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k2.r1']
line.element_refs['qc1r2.1'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k3.r1']
line.element_refs['qc1l1.4'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k0.l1']
line.element_refs['qc2l1.4'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k1.l1']
line.element_refs['qc2l2.4'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k2.l1']
line.element_refs['qc1l2.4'].k1 += line.vars['on_corr_ip.1'] * line.vars['corr_k3.l1']

tw_thick_no_rad = line.twiss(method='4d')


# Add dipole correctors
line.insert('mcb1.r1', xt.Multipole(knl=[0]), at='qc1r1.1@end')
line.insert('mcb2.r1', xt.Multipole(knl=[0]), at='qc1r2.1@end')
line.insert('mcb1.l1', xt.Multipole(knl=[0]), at='qc1l1.4@start')
line.insert('mcb2.l1', xt.Multipole(knl=[0]), at='qc1l2.4@start')

line.vars['acb1h.r1'] = 0
line.vars['acb1v.r1'] = 0
line.vars['acb2h.r1'] = 0
line.vars['acb2v.r1'] = 0
line.vars['acb1h.l1'] = 0
line.vars['acb1v.l1'] = 0
line.vars['acb2h.l1'] = 0
line.vars['acb2v.l1'] = 0

line.element_refs['mcb1.r1'].knl[0] = line.vars['on_corr_ip.1']*line.vars['acb1h.r1']
line.element_refs['mcb2.r1'].knl[0] = line.vars['on_corr_ip.1']*line.vars['acb2h.r1']
line.element_refs['mcb1.r1'].ksl[0] = line.vars['on_corr_ip.1']*line.vars['acb1v.r1']
line.element_refs['mcb2.r1'].ksl[0] = line.vars['on_corr_ip.1']*line.vars['acb2v.r1']
line.element_refs['mcb1.l1'].knl[0] = line.vars['on_corr_ip.1']*line.vars['acb1h.l1']
line.element_refs['mcb2.l1'].knl[0] = line.vars['on_corr_ip.1']*line.vars['acb2h.l1']
line.element_refs['mcb1.l1'].ksl[0] = line.vars['on_corr_ip.1']*line.vars['acb1v.l1']
line.element_refs['mcb2.l1'].ksl[0] = line.vars['on_corr_ip.1']*line.vars['acb2v.l1']

assert line.element_names[-1] == 'ip.4.l'
assert line.element_names[0] == 'ip.4'

opt = line.match(
    method='4d',
    start='ip.4', end='ip.4.l',
    init=tw_thick_no_rad,
    vary=xt.VaryList(['k1qf4', 'k1qf2', 'k1qd3', 'k1qd1',], step=1e-8,
    ),
    targets=[
        xt.TargetSet(at=xt.END, mux=tw_thick_no_rad.qx, muy=tw_thick_no_rad.qy, tol=1e-5),
    ]
)
opt.solve()
tw_thin_no_rad = line.twiss(method='4d')

# Check partition numbers
line.to_json(fname + '_thick_with_sol.json')

line.vars['voltca1'] = line.vars['voltca1_ref']
line.vars['voltca2'] = line.vars['voltca2_ref']

line.build_tracker()

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

two = line.twiss(eneloss_and_damping=True, particle_on_co=line.particle_ref.copy())
tw = line.twiss(eneloss_and_damping=True)

print(tw.partition_numbers)