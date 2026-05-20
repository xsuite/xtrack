import xtrack as xt

from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
import numpy as np

env = xt.load('fccee_z_lcc.json')
line = env.fccee_p_ring

tw0 = line.twiss4d()

ip_name = 'ipg'

sf = SolenoidField(L=1.23*2, a=0.13, B0=2., z0=0)

# Tilt with respect to the beam axis
theta = -0.015

# s coordinate along the beam axis
s = np.linspace(-2.4, 2.4, 201)

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

# Put the solenoid in the fcc lattice
s_ip = tw0['s', ip_name]
line.insert(line_solenoid, anchor='center', at=s_ip)
line.insert(ip_name, at=s_ip, s_tol=1e-9) # Put back the ip

tt = line.get_table()

line['on_sol'] = 0
tw_off = line.twiss4d()

line['on_sol'] = 1
tw_on = line.twiss4d()


tt_left = tt.rows['end_ds_start_straight_ipg':'ipg']
tt_right = tt.rows['ipg':'end_straight_start_ds_ipg']

# Attach dipole correction knobs
elements_for_orbit_correction_left = [
       'qf1dl.1', 'qf1cl.1', 'qf1bl.1', 'qf1al.1', 'qd0cl.1', 'qd0bl.1']
h_correction_knobs_left = []
v_correction_knobs_left = []
for nn in elements_for_orbit_correction_left:
    nn_h = 'acdh.' + nn
    nn_v = 'acv.' + nn
    env[nn_h] = 0
    env[nn_v] = 0
    env[nn].knl[0] = nn_h
    env[nn].ksl[0] = nn_v
    h_correction_knobs_left.append(nn_h)
    v_correction_knobs_left.append(nn_v)

elements_for_orbit_correction_right = [
       'qd0ar.2', 'qd0br.2', 'qd0cr.2', 'qf1ar.2', 'qf1br.2', 'qf1cr.2']
h_correction_knobs_right = []
v_correction_knobs_right = []
for nn in elements_for_orbit_correction_right:
    nn_h = 'acdh.' + nn
    nn_v = 'acv.' + nn
    env[nn_h] = 0
    env[nn_v] = 0
    env[nn].knl[0] = nn_h
    env[nn].ksl[0] = nn_v
    h_correction_knobs_right.append(nn_h)
    v_correction_knobs_right.append(nn_v)

opt_orbit_left = line.match(
    solve=False,
    start=tt_left.name[0],
    end=ip_name,
    init_at=ip_name,
    init=tw_off,
    targets=xt.TargetSet(x=0, px=0, y=0, py=0, at=tt_left.name[0]),
    vary=xt.VaryList(h_correction_knobs_left + v_correction_knobs_left)
)
opt_orbit_left.solve()

opt_orbit_right = line.match(
    solve=False,
    start='ipg',
    end=tt_right.name[-1],
    init_at=ip_name,
    init=tw_off,
    targets=xt.TargetSet(x=0, px=0, y=0, py=0, at=tt_right.name[-1]),
    vary=xt.VaryList(h_correction_knobs_right + v_correction_knobs_right)
)
opt_orbit_right.solve()

tw_corrected_orbit = line.twiss4d()