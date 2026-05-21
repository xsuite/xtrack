import xtrack as xt

from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
import numpy as np

env = xt.load('fccee_z_lcc.json')
line = env.fccee_p_ring

tw0 = line.twiss4d(strengths=True)

ip_name = 'ipg'

sf = SolenoidField(L=1.23*2, a=0.13, B0=2., z0=0)

# Tilt with respect to the beam axis
theta = -0.015

# s coordinate along the beam axis
s = np.linspace(-2.399, 2.399, 201)

# Corresponding coordinates of the beam reference trajectory in the solenoid frame
s_sol = s * np.cos(theta)
x_sol = s * np.sin(theta)
y_sol = 0 * x_sol

# Compute field on the beam reference trajectory in the solenoid frame
bx_sol, by_sol, bz_sol = sf.get_field(x_sol, y_sol, s_sol)

# Transform field to the beam frame
bx = bx_sol * np.cos(theta) - bz_sol * np.sin(theta)
bz = bx_sol * np.sin(theta) + bz_sol * np.cos(theta)
by = by_sol

# Normalized strengths
rigidity0 = line.particle_ref.rigidity0[0]
ks = bz / rigidity0
k0s = bx / rigidity0
k0 = by / rigidity0

# Build solenoid line
env['on_sol'] = 1
ele_names = []
for ii in range(len(s)-1):
    ks_entry = ks[ii]
    ks_exit = ks[ii+1]
    k0s_entry = k0s[ii]
    k0s_exit = k0s[ii+1]
    k0_entry = k0[ii]
    k0_exit = k0[ii+1]
    s_entry = s[ii]
    s_exit = s[ii+1]

    length = s_exit - s_entry
    s_mid = 0.5 * (s_entry + s_exit)

    env.new(f'sol_slice_{ii}', xt.VariableSolenoid,
        length=length,
        ks_profile=[ks_entry * env.ref['on_sol'], ks_exit * env.ref['on_sol']],
        knl=[0.5 * (k0_exit + k0_entry) * length * env.ref['on_sol']],
        ksl=[0.5 * (k0s_exit + k0s_entry) * length * env.ref['on_sol']],
    )
    ele_names.append(f'sol_slice_{ii}')

line_solenoid = env.new_line(components=ele_names)
ksol_l_main_solenoid = 0
for nn in line_solenoid.element_names:
    ee = env.get(nn)
    if isinstance(ee, xt.VariableSolenoid):
        ksol_l_main_solenoid += ee.ks_profile.mean() * ee.length

# Measure rotation angle from solenoid
line_solenoid.particle_ref = line.particle_ref.copy()
tw_tst = line_solenoid.twiss(betx=1, bety=1, px=1e-6)

# Make compensation solenoid

sfc = SolenoidField(L=1.5, a=0.03, B0=1., z0=0)
s_comp = np.linspace(-1, 1., 51)
_, _, bzc = sfc.get_field(0*s_comp, 0*s_comp, s_comp)
ks_comp = bzc / rigidity0
env['on_comp_sol'] = 1
env['field_comp_sol'] = 1.
ele_names_comp = []
for ii in range(len(s_comp)-1):

    s_entry = s_comp[ii]
    s_exit = s_comp[ii+1]

    length = s_exit - s_entry

    env.new(f'comp_sol_slice_{ii}', xt.VariableSolenoid,
        length=length,
        ks_profile=[ks_comp[ii] * env.ref['on_comp_sol'] * env.ref['field_comp_sol'],
                    ks_comp[ii+1] * env.ref['on_comp_sol'] * env.ref['field_comp_sol']],
    )
    ele_names_comp.append(f'comp_sol_slice_{ii}')

line_comp_solenoid = env.new_line(components=ele_names_comp)

ksol_l_comp_solenoid = 0
for nn in line_comp_solenoid.element_names:
    ee = env.get(nn)
    if isinstance(ee, xt.VariableSolenoid):
        ksol_l_comp_solenoid += ee.ks_profile.mean() * ee.length

# Scale to have zero integrated field
env['field_comp_sol'] = -ksol_l_main_solenoid / ksol_l_comp_solenoid / 2

line_comp_solenoid_left = line_comp_solenoid.clone(suffix='_left')
line_comp_solenoid_right = line_comp_solenoid.clone(suffix='_right')

# Put the solenoids in the fcc lattice
s_ip = tw0['s', ip_name]
line.insert(line_solenoid, anchor='center', at=s_ip)
line.insert(ip_name, at=s_ip, s_tol=1e-9) # Put back the ip
line.insert(line_comp_solenoid_left, anchor='end', at=-12, from_=ip_name)
line.insert(line_comp_solenoid_right, anchor='start', at=12, from_=ip_name)

# Tilt the doublets
doublet_quad_left = [
       'qd0al.1', 'qd0bl.1', 'qd0cl.1', 'qf1al.1', 'qf1bl.1', 'qf1cl.1', 'qf1dl.1']
doublet_quad_right = [
       'qd0ar.2', 'qd0br.2', 'qd0cr.2', 'qf1ar.2', 'qf1br.2', 'qf1cr.2', 'qf1dr.2']

env['phi_rot_doublet'] = (ksol_l_main_solenoid / 2) / 2 # in parentheses is the full solenoid rotation, we want half of it for each doublet
env['on_rot_doublet_left'] = 1
env['on_rot_doublet_right'] = 1
for nn in doublet_quad_left:
    env[nn].rot_s_rad = +env.ref['phi_rot_doublet'] * env.ref['on_rot_doublet_left']
for nn in doublet_quad_right:
    env[nn].rot_s_rad = -env.ref['phi_rot_doublet'] * env.ref['on_rot_doublet_right']


tt = line.get_table()
tt_left = tt.rows['end_ds_start_straight_ipg':'ipg']
tt_right = tt.rows['ipg':'end_straight_start_ds_ipg']

line['on_sol'] = 0
line['on_comp_sol'] = 0
line['on_rot_doublet_right'] = 0
line['on_rot_doublet_left'] = 0
tw_off = line.twiss4d()

line['on_sol'] = 1
line['on_comp_sol'] = 0
line['on_rot_doublet_right'] = 0
line['on_rot_doublet_left'] = 0
tw_sol_on_comp_sol_off = line.twiss4d()
two_sol_on_comp_sol_off = line.twiss(
    start='end_ds_start_straight_ipg',
    end='end_straight_start_ds_ipg',
    init_at=ip_name,
    init=tw_off)

line['on_sol'] = 1
line['on_comp_sol'] = 1
line['on_rot_doublet_right'] = 1
line['on_rot_doublet_left'] = 1
two_sol_on_comp_sol_on = line.twiss(
    start='end_ds_start_straight_ipg',
    end='end_straight_start_ds_ipg',
    init_at=ip_name,
    init=tw_off)

