import xtrack as xt

from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
import numpy as np

env = xt.load('temp_fcc_ee_lcc_solenoid.json')
line = env.fccee_p_ring

# Reference twiss with solenoids off
line['on_sol_ipa'] = 0
line['on_sol_ipd'] = 0
line['on_sol_ipg'] = 0
line['on_sol_ipj'] = 0
line['on_comp_sol_ipa'] = 0
line['on_comp_sol_ipd'] = 0
line['on_comp_sol_ipg'] = 0
line['on_comp_sol_ipj'] = 0
tw0 = line.twiss4d(strengths=True)
line['on_sol_ipa'] = 1
line['on_sol_ipd'] = 1
line['on_sol_ipg'] = 1
line['on_sol_ipj'] = 1
line['on_comp_sol_ipa'] = 1
line['on_comp_sol_ipd'] = 1
line['on_comp_sol_ipg'] = 1
line['on_comp_sol_ipj'] = 1

# Configuration (which elements are used for the correction at each ip)
config = {}
config['ipa'] = {
    'quad_for_optics_correction': [
        'qd0ar.0', 'qd0br.0', 'qd0cr.0', 'qf1ar.0', 'qf1br.0', 'qf1cr.0',
        'qf1dr.0', 'qf2r.0', 'qd3r.0', 'qd4r.0', 'qf5r.0', 'qd6r.0',
        'qd6l.3', 'qf5l.3', 'qd4l.3', 'qd3l.3', 'qf2l.3', 'qf1dl.3', 'qf1cl.3',
        'qf1bl.3', 'qf1al.3', 'qd0cl.3', 'qd0bl.3', 'qd0al.3'
        ],
    'doublet_quad_left': [
        'qd0al.3', 'qd0bl.3', 'qd0cl.3', 'qf1al.3', 'qf1bl.3', 'qf1cl.3', 'qf1dl.3'],
    'doublet_quad_right': [
        'qd0ar.0', 'qd0br.0', 'qd0cr.0', 'qf1ar.0', 'qf1br.0', 'qf1cr.0', 'qf1dr.0'],
    'corr_1_right_on_quad': 'qd0ar.0',
    'corr_2_right_on_quad': 'qd0br.0',
    'corr_3_right_on_quad': 'qf1ar.0',
    'corr_4_right_on_quad': 'qf1br.0',
    'corr_1_left_on_quad': 'qd0al.3',
    'corr_2_left_on_quad': 'qd0bl.3',
    'corr_3_left_on_quad': 'qf1al.3',
    'corr_4_left_on_quad': 'qf1bl.3',
}
config['ipd'] = {
    'quad_for_optics_correction': [
        'qd0ar.1', 'qd0br.1', 'qd0cr.1', 'qf1ar.1', 'qf1br.1', 'qf1cr.1',
        'qf1dr.1', 'qf2r.1', 'qd3r.1', 'qd4r.1', 'qf5r.1', 'qd6r.1',
        'qd6l.0', 'qf5l.0', 'qd4l.0', 'qd3l.0', 'qf2l.0', 'qf1dl.0', 'qf1cl.0',
        'qf1bl.0', 'qf1al.0', 'qd0cl.0', 'qd0bl.0', 'qd0al.0'
        ],
    'doublet_quad_left': [
        'qd0al.0', 'qd0bl.0', 'qd0cl.0', 'qf1al.0', 'qf1bl.0', 'qf1cl.0', 'qf1dl.0'],
    'doublet_quad_right': [
        'qd0ar.1', 'qd0br.1', 'qd0cr.1', 'qf1ar.1', 'qf1br.1', 'qf1cr.1', 'qf1dr.1'],
    'corr_1_right_on_quad': 'qd0ar.1',
    'corr_2_right_on_quad': 'qd0br.1',
    'corr_3_right_on_quad': 'qf1ar.1',
    'corr_4_right_on_quad': 'qf1br.1',
    'corr_1_left_on_quad': 'qd0al.0',
    'corr_2_left_on_quad': 'qd0bl.0',
    'corr_3_left_on_quad': 'qf1al.0',
    'corr_4_left_on_quad': 'qf1bl.0',
}
config['ipg'] = {
    'quad_for_optics_correction': [
        'qd0ar.2', 'qd0br.2', 'qd0cr.2', 'qf1ar.2', 'qf1br.2', 'qf1cr.2',
        'qf1dr.2', 'qf2r.2', 'qd3r.2', 'qd4r.2', 'qf5r.2', 'qd6r.2',
        'qd6l.1', 'qf5l.1', 'qd4l.1', 'qd3l.1', 'qf2l.1', 'qf1dl.1', 'qf1cl.1',
        'qf1bl.1', 'qf1al.1', 'qd0cl.1', 'qd0bl.1', 'qd0al.1'
        ],
    'doublet_quad_left': [
        'qd0al.1', 'qd0bl.1', 'qd0cl.1', 'qf1al.1', 'qf1bl.1', 'qf1cl.1', 'qf1dl.1'],
    'doublet_quad_right': [
        'qd0ar.2', 'qd0br.2', 'qd0cr.2', 'qf1ar.2', 'qf1br.2', 'qf1cr.2', 'qf1dr.2'],
    'corr_1_right_on_quad': 'qd0ar.2',
    'corr_2_right_on_quad': 'qd0br.2',
    'corr_3_right_on_quad': 'qf1ar.2',
    'corr_4_right_on_quad': 'qf1br.2',
    'corr_1_left_on_quad': 'qd0al.1',
    'corr_2_left_on_quad': 'qd0bl.1',
    'corr_3_left_on_quad': 'qf1al.1',
    'corr_4_left_on_quad': 'qf1bl.1',
}

config['ipj'] = {
    'quad_for_optics_correction': [
        'qd0ar.3', 'qd0br.3', 'qd0cr.3', 'qf1ar.3', 'qf1br.3', 'qf1cr.3',
        'qf1dr.3', 'qf2r.3', 'qd3r.3', 'qd4r.3', 'qf5r.3', 'qd6r.3',
        'qd6l.2', 'qf5l.2', 'qd4l.2', 'qd3l.2', 'qf2l.2', 'qf1dl.2', 'qf1cl.3',
        'qf1bl.2', 'qf1al.2', 'qd0cl.2', 'qd0bl.2', 'qd0al.2'
    ],
    'doublet_quad_left': [
        'qd0al.2', 'qd0bl.2', 'qd0cl.2', 'qf1al.2', 'qf1bl.2', 'qf1cl.2', 'qf1dl.2'],
    'doublet_quad_right': [
        'qd0ar.3', 'qd0br.3', 'qd0cr.3', 'qf1ar.3', 'qf1br.3', 'qf1cr.3', 'qf1dr.3'],
    'corr_1_right_on_quad': 'qd0ar.3',
    'corr_2_right_on_quad': 'qd0br.3',
    'corr_3_right_on_quad': 'qf1ar.3',
    'corr_4_right_on_quad': 'qf1br.3',
    'corr_1_left_on_quad': 'qd0al.2',
    'corr_2_left_on_quad': 'qd0bl.2',
    'corr_3_left_on_quad': 'qf1al.2',
    'corr_4_left_on_quad': 'qf1bl.2',
}

for ip_name in config.keys():

    print(f'IP {ip_name}:')
    line.cycle(f'end_ds_start_straight_{ip_name}')

    quad_for_optics_correction = config[ip_name]['quad_for_optics_correction']
    doublet_quad_left = config[ip_name]['doublet_quad_left']
    doublet_quad_right = config[ip_name]['doublet_quad_right']
    corr_1_right_on_quad = config[ip_name]['corr_1_right_on_quad']
    corr_2_right_on_quad = config[ip_name]['corr_2_right_on_quad']
    corr_3_right_on_quad = config[ip_name]['corr_3_right_on_quad']
    corr_4_right_on_quad = config[ip_name]['corr_4_right_on_quad']
    corr_1_left_on_quad = config[ip_name]['corr_1_left_on_quad']
    corr_2_left_on_quad = config[ip_name]['corr_2_left_on_quad']
    corr_3_left_on_quad = config[ip_name]['corr_3_left_on_quad']
    corr_4_left_on_quad = config[ip_name]['corr_4_left_on_quad']

    # Measure integrated field of the main solenoid
    ksol_l_main_solenoid = 0
    tt_sol_doublet = line.get_table().rows['dy_match_l_'+ip_name : 'dy_match_r_'+ip_name]
    for nn in tt_sol_doublet.name:

        if tt_sol_doublet['element_type', nn] == 'VariableSolenoid':
            ee = env.get(nn)
            ksol_l_main_solenoid += ee.ks_profile.mean() * ee.length

    # Tilt the doublets
    env[f'phi_rot_doublet_{ip_name}'] = (ksol_l_main_solenoid / 2) / 2 # in parentheses is the full solenoid rotation, we want half of it for each doublet
    env[f'on_rot_doublet_left_{ip_name}'] = 1
    env[f'on_rot_doublet_right_{ip_name}'] = 1
    for nn in doublet_quad_left:
        env[nn].rot_s_rad = +env.ref[f'phi_rot_doublet_{ip_name}'] * env.ref[f'on_rot_doublet_left_{ip_name}']
    for nn in doublet_quad_right:
        env[nn].rot_s_rad = -env.ref[f'phi_rot_doublet_{ip_name}'] * env.ref[f'on_rot_doublet_right_{ip_name}']

    # Define orbit corrector knobs (the first is already embedded in the solenoid region)
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

    # Attach knobs to correctors
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

    # Match orbit and vertical dispersion
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

    # Match optics and horizontal dispersion
    k1_knobs = []
    for nn in quad_for_optics_correction:
        nn_knob = 'k1_' + nn + '_sol_corr'
        env[nn_knob] = 0
        env[nn].k1 += env.ref[nn_knob]
        k1_knobs.append(nn_knob)

    name_start = f'end_ds_start_straight_{ip_name}'
    name_end = f'end_straight_start_ds_{ip_name}'
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

    # Iterate to improve consistency of optics + orbit correction
    opt_orbit.solve()
    opt_optics.solve()
    opt_orbit.solve()
    opt_optics.solve()

    # Generate the knobs
    opt_orbit.generate_knob()
    opt_optics.generate_knob()

    # Control all correction with a single knob
    line[f'on_sol_corr_{ip_name}'] = 1

    line[f'on_comp_sol_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_rot_doublet_right_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_rot_doublet_left_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_sol_orbit_corr_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_sol_optics_corr_{ip_name}'] = f'on_sol_corr_{ip_name}'

# Cycle to ipa before saving
line.cycle('ipa')

line['on_sol_ipa'] = 0
line['on_sol_ipd'] = 0
line['on_sol_ipg'] = 0
line['on_sol_ipj'] = 0
line['on_sol_corr_ipa'] = 0
line['on_sol_corr_ipd'] = 0
line['on_sol_corr_ipg'] = 0
line['on_sol_corr_ipj'] = 0
tw_off = line.twiss4d(strengths=True, zero_at=ip_name)
nl_chrom_off = line.get_non_linear_chromaticity(delta0_range=(-1e-2, 1e-2))

line['on_sol_ipa'] = 1
line['on_sol_ipd'] = 1
line['on_sol_ipg'] = 1
line['on_sol_ipj'] = 1
line['on_sol_corr_ipa'] = 1
line['on_sol_corr_ipd'] = 1
line['on_sol_corr_ipg'] = 1
line['on_sol_corr_ipj'] = 1
tw_on_corr = line.twiss4d(strengths=True, zero_at=ip_name)
# nl_chrom_on_corr = line.get_non_linear_chromaticity(delta0_range=(-1e-2, 1e-2))
two_on_corr = line.twiss(
    strengths=True,
    start=f'end_ds_start_straight_{ip_name}',
    end=f'end_straight_start_ds_{ip_name}',
    init_at=ip_name,
    init=tw_off,
    zero_at=ip_name)

env.to_json('fcc_z_lcc_solenoid.json')

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
