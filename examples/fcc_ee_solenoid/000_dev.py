import xtrack as xt

from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
import numpy as np

env = xt.load('fccee_z_lcc.json')
line = env.fccee_p_ring

tw0 = line.twiss4d(strengths=True)

# ip_name = 'ipg'
# quad_for_optics_correction = [
#        # right
#        'qd0ar.2', 'qd0br.2', 'qd0cr.2', 'qf1ar.2', 'qf1br.2', 'qf1cr.2',
#        'qf1dr.2', 'qf2r.2', 'qd3r.2', 'qd4r.2', 'qf5r.2', 'qd6r.2',
#        #left
#        'qd6l.1', 'qf5l.1', 'qd4l.1', 'qd3l.1', 'qf2l.1', 'qf1dl.1', 'qf1cl.1',
#        'qf1bl.1', 'qf1al.1', 'qd0cl.1', 'qd0bl.1', 'qd0al.1'
# ]
# doublet_quad_left = [
#        'qd0al.1', 'qd0bl.1', 'qd0cl.1', 'qf1al.1', 'qf1bl.1', 'qf1cl.1', 'qf1dl.1']
# doublet_quad_right = [
#        'qd0ar.2', 'qd0br.2', 'qd0cr.2', 'qf1ar.2', 'qf1br.2', 'qf1cr.2', 'qf1dr.2']
# corr_1_right_on_quad = 'qd0ar.2'
# corr_2_right_on_quad = 'qd0br.2'
# corr_3_right_on_quad = 'qf1ar.2'
# corr_4_right_on_quad = 'qf1br.2'
# corr_1_left_on_quad = 'qd0al.1'
# corr_2_left_on_quad = 'qd0bl.1'
# corr_3_left_on_quad = 'qf1al.1'
# corr_4_left_on_quad = 'qf1bl.1'

ip_name = 'ipj'
quad_for_optics_correction = [
       # right
       'qd0ar.3', 'qd0br.3', 'qd0cr.3', 'qf1ar.3', 'qf1br.3', 'qf1cr.3',
       'qf1dr.3', 'qf2r.3', 'qd3r.3', 'qd4r.3', 'qf5r.3', 'qd6r.3',
       #left
       'qd6l.2', 'qf5l.2', 'qd4l.2', 'qd3l.2', 'qf2l.2', 'qf1dl.2', 'qf1cl.3',
       'qf1bl.2', 'qf1al.2', 'qd0cl.2', 'qd0bl.2', 'qd0al.2'
]
doublet_quad_left = [
       'qd0al.2', 'qd0bl.2', 'qd0cl.2', 'qf1al.2', 'qf1bl.2', 'qf1cl.2', 'qf1dl.2']
doublet_quad_right = [
       'qd0ar.3', 'qd0br.3', 'qd0cr.3', 'qf1ar.3', 'qf1br.3', 'qf1cr.3', 'qf1dr.3']
corr_1_right_on_quad = 'qd0ar.3'
corr_2_right_on_quad = 'qd0br.3'
corr_3_right_on_quad = 'qf1ar.3'
corr_4_right_on_quad = 'qf1br.3'
corr_1_left_on_quad = 'qd0al.2'
corr_2_left_on_quad = 'qd0bl.2'
corr_3_left_on_quad = 'qf1al.2'
corr_4_left_on_quad = 'qf1bl.2'

line.insert('dy_match_r_'+ip_name, xt.Marker(), at=11.95, from_=ip_name)
line.insert('dy_match_l_'+ip_name, xt.Marker(), at=-11.95, from_=ip_name)

line.insert([
    env.new(f'corr_sol_right_{ip_name}', xt.Multipole, at=0, from_=f'dy_match_r_{ip_name}@start'),
    env.new(f'corr_sol_left_{ip_name}', xt.Multipole, at=0, from_=f'dy_match_l_{ip_name}@end'),
])

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
env[f'on_sol_{ip_name}'] = 1
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

    env.new(f'sol_slice_{ii}_{ip_name}', xt.VariableSolenoid,
        length=length,
        ks_profile=[ks_entry * env.ref[f'on_sol_{ip_name}'], ks_exit * env.ref[f'on_sol_{ip_name}']],
        knl=[0.5 * (k0_exit + k0_entry) * length * env.ref[f'on_sol_{ip_name}']],
        ksl=[0.5 * (k0s_exit + k0s_entry) * length * env.ref[f'on_sol_{ip_name}']],
    )
    ele_names.append(f'sol_slice_{ii}_{ip_name}')

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
env[f'on_comp_sol_{ip_name}'] = 1
env[f'field_comp_sol_{ip_name}'] = 1.
ele_names_comp = []
for ii in range(len(s_comp)-1):

    s_entry = s_comp[ii]
    s_exit = s_comp[ii+1]

    length = s_exit - s_entry

    env.new(f'comp_sol_slice_{ii}_{ip_name}', xt.VariableSolenoid,
        length=length,
        ks_profile=[ks_comp[ii] * env.ref[f'on_comp_sol_{ip_name}'] * env.ref[f'field_comp_sol_{ip_name}'],
                    ks_comp[ii+1] * env.ref[f'on_comp_sol_{ip_name}'] * env.ref[f'field_comp_sol_{ip_name}']],
    )
    ele_names_comp.append(f'comp_sol_slice_{ii}_{ip_name}')

line_comp_solenoid = env.new_line(components=ele_names_comp)

ksol_l_comp_solenoid = 0
for nn in line_comp_solenoid.element_names:
    ee = env.get(nn)
    if isinstance(ee, xt.VariableSolenoid):
        ksol_l_comp_solenoid += ee.ks_profile.mean() * ee.length

# Scale to have zero integrated field
env[f'field_comp_sol_{ip_name}'] = -ksol_l_main_solenoid / ksol_l_comp_solenoid / 2

line_comp_solenoid_left = line_comp_solenoid.clone(suffix=f'left_{ip_name}')
line_comp_solenoid_right = line_comp_solenoid.clone(suffix=f'right_{ip_name}')

# Put the solenoids in the fcc lattice``
s_ip = tw0['s', ip_name]
line.insert(line_solenoid, anchor='center', at=s_ip)
line.insert(ip_name, at=s_ip, s_tol=1e-9) # Put back the ip
line.insert(line_comp_solenoid_left, anchor='end', at=-12, from_=ip_name)
line.insert(line_comp_solenoid_right, anchor='start', at=12, from_=ip_name)

# Tilt the doublets


env[f'phi_rot_doublet_{ip_name}'] = (ksol_l_main_solenoid / 2) / 2 # in parentheses is the full solenoid rotation, we want half of it for each doublet
env[f'on_rot_doublet_left_{ip_name}'] = 1
env[f'on_rot_doublet_right_{ip_name}'] = 1
for nn in doublet_quad_left:
    env[nn].rot_s_rad = +env.ref[f'phi_rot_doublet_{ip_name}'] * env.ref[f'on_rot_doublet_left_{ip_name}']
for nn in doublet_quad_right:
    env[nn].rot_s_rad = -env.ref[f'phi_rot_doublet_{ip_name}'] * env.ref[f'on_rot_doublet_right_{ip_name}']


# Overlay dipole corrector in between 1.23m and 2.29
tt = line.get_table()
tt_region = tt.rows[f'end_ds_start_straight_{ip_name}':f'end_straight_start_ds_{ip_name}']
s_ip = tt_region['s', ip_name]

# Inner kicker right
ds_start = 1.23
ds_end = 2.29
tt_kicker_right= tt_region.rows[s_ip + ds_start: s_ip + ds_end:'s']

assert np.all(tt_kicker_right.element_type == 'VariableSolenoid')
l_tot = tt_kicker_right['s_end'][-1] - tt_kicker_right['s_start'][0]

env[f'acbh1_sol_right_{ip_name}'] = 0
env[f'acbv1_sol_right_{ip_name}'] = 0
for nn in tt_kicker_right.name:
    ee = env.get(nn)
    env.ref[nn].knl[0] += env.ref[f'acbh1_sol_right_{ip_name}']/l_tot * ee.length
    env.ref[nn].ksl[0] += env.ref[f'acbv1_sol_right_{ip_name}']/l_tot * ee.length

# Inner kicker left
ds_start = -2.29
ds_end = -1.23
tt_kicker_left = tt_region.rows[s_ip + ds_start: s_ip + ds_end:'s']
assert np.all(tt_kicker_left.element_type == 'VariableSolenoid')
l_tot = tt_kicker_left['s_end'][-1] - tt_kicker_left['s_start'][0]

env[f'acbh1_sol_left_{ip_name}'] = 0
env[f'acbv1_sol_left_{ip_name}'] = 0
for nn in tt_kicker_left.name:
    ee = env.get(nn)
    env.ref[nn].knl[0] += env.ref[f'acbh1_sol_left_{ip_name}']/l_tot * ee.length
    env.ref[nn].ksl[0] += env.ref[f'acbv1_sol_left_{ip_name}']/l_tot * ee.length

# Outer kickers
env[f'acbh2_sol_right_{ip_name}'] = 0
env[f'acbh3_sol_right_{ip_name}'] = 0
env[f'acbh4_sol_right_{ip_name}'] = 0
env[f'acbh5_sol_right_{ip_name}'] = 0
env[f'acbh6_sol_right_{ip_name}'] = 0
env[f'acbv2_sol_right_{ip_name}'] = 0
env[f'acbv3_sol_right_{ip_name}'] = 0
env[f'acbv4_sol_right_{ip_name}'] = 0
env[f'acbv5_sol_right_{ip_name}'] = 0
env[f'acbv6_sol_right_{ip_name}'] = 0
env[f'acbh2_sol_left_{ip_name}'] = 0
env[f'acbh3_sol_left_{ip_name}'] = 0
env[f'acbh4_sol_left_{ip_name}'] = 0
env[f'acbh5_sol_left_{ip_name}'] = 0
env[f'acbh6_sol_left_{ip_name}'] = 0
env[f'acbv2_sol_left_{ip_name}'] = 0
env[f'acbv3_sol_left_{ip_name}'] = 0
env[f'acbv4_sol_left_{ip_name}'] = 0
env[f'acbv5_sol_left_{ip_name}'] = 0
env[f'acbv6_sol_left_{ip_name}'] = 0

env[corr_1_right_on_quad].knl[0] += env.ref[f'acbh2_sol_right_{ip_name}']
env[corr_2_right_on_quad].knl[0] += env.ref[f'acbh3_sol_right_{ip_name}']
env[corr_3_right_on_quad].knl[0] += env.ref[f'acbh4_sol_right_{ip_name}']
env[corr_4_right_on_quad].knl[0] += env.ref[f'acbh5_sol_right_{ip_name}']
env['corr_sol_right_'+ip_name].knl[0] += env.ref[f'acbh6_sol_right_{ip_name}']

env[corr_1_left_on_quad].knl[0] += env.ref[f'acbh2_sol_left_{ip_name}']
env[corr_2_left_on_quad].knl[0] += env.ref[f'acbh3_sol_left_{ip_name}']
env[corr_3_left_on_quad].knl[0] += env.ref[f'acbh4_sol_left_{ip_name}']
env[corr_4_left_on_quad].knl[0] += env.ref[f'acbh5_sol_left_{ip_name}']
env['corr_sol_left_'+ip_name].knl[0] += env.ref[f'acbh6_sol_left_{ip_name}']

env[corr_1_right_on_quad].ksl[0] += env.ref[f'acbv2_sol_right_{ip_name}']
env[corr_2_right_on_quad].ksl[0] += env.ref[f'acbv3_sol_right_{ip_name}']
env[corr_3_right_on_quad].ksl[0] += env.ref[f'acbv4_sol_right_{ip_name}']
env[corr_4_right_on_quad].ksl[0] += env.ref[f'acbv5_sol_right_{ip_name}']
env['corr_sol_right_'+ip_name].ksl[0] += env.ref[f'acbv6_sol_right_{ip_name}']

env[corr_1_left_on_quad].ksl[0] += env.ref[f'acbv2_sol_left_{ip_name}']
env[corr_2_left_on_quad].ksl[0] += env.ref[f'acbv3_sol_left_{ip_name}']
env[corr_3_left_on_quad].ksl[0] += env.ref[f'acbv4_sol_left_{ip_name}']
env[corr_4_left_on_quad].ksl[0] += env.ref[f'acbv5_sol_left_{ip_name}']
env['corr_sol_left_'+ip_name].ksl[0] += env.ref[f'acbv6_sol_left_{ip_name}']

# Match orbit and vertical dispersion right
opt_orbit = line.match_knob(
    knob_name=f'on_sol_orbit_corr_{ip_name}',
    run=False,
    init=tw0,
    start='dy_match_l_'+ip_name,
    end='dy_match_r_'+ip_name,
    init_at=ip_name,
    vary=xt.VaryList([
        f'acbh1_sol_right_{ip_name}', f'acbv1_sol_right_{ip_name}',
        f'acbh2_sol_right_{ip_name}', f'acbh3_sol_right_{ip_name}',
        f'acbh4_sol_right_{ip_name}', f'acbh5_sol_right_{ip_name}',
        f'acbh6_sol_right_{ip_name}', f'acbv2_sol_right_{ip_name}',
        f'acbv3_sol_right_{ip_name}', f'acbv4_sol_right_{ip_name}',
        f'acbv5_sol_right_{ip_name}', f'acbv6_sol_right_{ip_name}',
        f'acbh1_sol_left_{ip_name}', f'acbv1_sol_left_{ip_name}',
        f'acbh2_sol_left_{ip_name}', f'acbh3_sol_left_{ip_name}',
        f'acbh4_sol_left_{ip_name}', f'acbh5_sol_left_{ip_name}',
        f'acbh6_sol_left_{ip_name}', f'acbv2_sol_left_{ip_name}',
        f'acbv3_sol_left_{ip_name}', f'acbv4_sol_left_{ip_name}',
        f'acbv5_sol_left_{ip_name}', f'acbv6_sol_left_{ip_name}',
        ], step=1e-6),
    targets=[
        xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.END),
        xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.START)
    ])
opt_orbit.solve()


k1_knobs = []
for nn in quad_for_optics_correction:
    nn_knob = 'k1_' + nn + '_sol_corr'
    env[nn_knob] = 0
    env[nn].k1 += env.ref[nn_knob]
    k1_knobs.append(nn_knob)

name_start = tt_region.name[0]
name_end = tt_region.name[-1]
opt_optics = line.match_knob(
    knob_name=f'on_sol_optics_corr_{ip_name}',
    run=False,
    init=tw0,
    init_at=ip_name,
    start=name_start,
    end=name_end,
    vary=xt.VaryList(k1_knobs, step=1e-6),
    targets=[
        xt.TargetSet(betx=tw0['betx', name_start], bety=tw0['bety', name_start], tol=1e-5, at=xt.START),
        xt.TargetSet(alfx=tw0['alfx', name_start], alfy=tw0['alfy', name_start], tol=1e-8, at=xt.START),
        xt.TargetSet(dx=tw0['dx', name_start], dpx=tw0['dpx', name_start], tol=1e-8, at=xt.START),
        xt.TargetSet(betx=tw0['betx', name_end], bety=tw0['bety', name_end], tol=1e-5, at=xt.END),
        xt.TargetSet(alfx=tw0['alfx', name_end], alfy=tw0['alfy', name_end], tol=1e-8, at=xt.END),
        xt.TargetSet(dx=tw0['dx', name_end], dpx=tw0['dpx', name_end], tol=1e-8, at=xt.END)
    ])
opt_optics.solve()

# Iterate to improve orbit and optics correction
opt_orbit.solve()
opt_optics.solve()
opt_orbit.solve()
opt_optics.solve()

# Generate the knobs
opt_orbit.generate_knob()
opt_optics.generate_knob()

line[f'on_sol_{ip_name}'] = 0
line[f'on_comp_sol_{ip_name}'] = 0
line[f'on_rot_doublet_right_{ip_name}'] = 0
line[f'on_rot_doublet_left_{ip_name}'] = 0
line[f'on_sol_orbit_corr_{ip_name}'] = 0
line[f'on_sol_optics_corr_{ip_name}'] = 0
tw_off = line.twiss4d(strengths=True, zero_at=ip_name)
nl_chrom_off = line.get_non_linear_chromaticity(delta0_range=(-1e-2, 1e-2))

line[f'on_sol_{ip_name}'] = 1
line[f'on_comp_sol_{ip_name}'] = 1
line[f'on_rot_doublet_right_{ip_name}'] = 1
line[f'on_rot_doublet_left_{ip_name}'] = 1
line[f'on_sol_orbit_corr_{ip_name}'] = 1
line[f'on_sol_optics_corr_{ip_name}'] = 1
tw_on_corr = line.twiss4d(strengths=True, zero_at=ip_name)
nl_chrom_on_corr = line.get_non_linear_chromaticity(delta0_range=(-1e-2, 1e-2))
two_on_corr = line.twiss(
    strengths=True,
    start=f'end_ds_start_straight_{ip_name}',
    end=f'end_straight_start_ds_{ip_name}',
    init_at=ip_name,
    init=tw_off,
    zero_at=ip_name)

import matplotlib.pyplot as plt

plt.close('all')
fig1 = plt.figure(1)
tw_on_corr.rows[-20:20:'s'].plot('betx2 bety1', figure=fig1)

fig2 = plt.figure(2)
tw_on_corr.rows[-20:20:'s'].plot('x y', figure=fig2)

# Plot phase error
fig3 = plt.figure(3)
ax = fig3.add_subplot(3,1,1)
tw_off.plot(ax=ax)

ax2 = fig3.add_subplot(3,1,2, sharex=ax)
ax2.plot(tw_off.s, tw_off.muy - tw_on_corr.muy, label='muy error')

ax3 = fig3.add_subplot(3,1,3, sharex=ax)
ax3.plot(tw_off.s, tw_off.muy)
ax3.plot(tw_on_corr.s, tw_on_corr.muy, label='muy with solenoid')

plt.show()
