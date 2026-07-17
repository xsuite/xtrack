# Based on E. Boscolo, A. Ciarma, E. Burkhardt, https://cds.cern.ch/record/2948247
# Nuclear Instruments and Methods in Physics Research A 1083 (2026) 171135

import xtrack as xt
import time

from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
import numpy as np
from tilted_solenoid import TiltedSolenoid

t1 = time.time()

env = xt.load('fccee_z_lcc.json')
line = env.fccee_p_ring

ip_names = ['ipa'] #, 'ipd', 'ipg', 'ipj']

# Tilt with respect to the beam axis
theta = -0.015

sol_half_length = 1.3

# Location of first dipole corrector (overlaid with solenoid)
ds_start = 1.4
ds_end = 2.29

B0 = 3. # T

r0 = 0.13

for ip_name in ip_names:

    line.cycle(f'end_ds_start_straight_{ip_name}')
    tt = line.get_table()

    print(f'IP {ip_name}:')

    # Analytic field map
    sf = TiltedSolenoid(L=sol_half_length*2, a=r0, B0=B0, theta=theta)

    # s coordinate along the beam axis
    s = np.linspace(-2.399, 2.399, 201)

    # Compute field on the beam reference trajectory in the beam frame
    bx, by, bz = sf.get_field(0 * s, 0 * s, s)

    # Normalized strengths
    rigidity0 = line.particle_ref.rigidity0[0]
    ks = bz / rigidity0
    k0s = bx / rigidity0
    k0 = by / rigidity0

    # Build solenoid slices
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

    # Force solenoid field at edges to be zero (ax, ay zero at entry and exit)
    env[ele_names[0]].ks_profile[0] = 0
    env[ele_names[-1]].ks_profile[-1] = 0

    # Assemble the solenoid
    line_solenoid = env.new_line(components=ele_names)

    # Measure integrated field of the main solenoid
    ksol_l_main_solenoid = 0
    for nn in line_solenoid.element_names:
        ee = env.get(nn)
        if isinstance(ee, xt.VariableSolenoid):
            ksol_l_main_solenoid += ee.ks_profile.mean() * ee.length

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

    # Force compensation solenoid field at edges to be zero
    env[ele_names_comp[0]].ks_profile[0] = 0
    env[ele_names_comp[-1]].ks_profile[-1] = 0
    line_comp_solenoid = env.new_line(components=ele_names_comp)

    # Measure integrated field compensation solenoid
    ksol_l_comp_solenoid = 0
    for nn in line_comp_solenoid.element_names:
        ee = env.get(nn)
        if isinstance(ee, xt.VariableSolenoid):
            ksol_l_comp_solenoid += ee.ks_profile.mean() * ee.length

    # Scale to have zero integrated field (main + compensation)
    env[f'field_comp_sol_{ip_name}'] = -ksol_l_main_solenoid / ksol_l_comp_solenoid / 2

    # Put the solenoids in the fcc lattice
    line_comp_solenoid_left = line_comp_solenoid.clone(suffix=f'left_{ip_name}')
    line_comp_solenoid_right = line_comp_solenoid.clone(suffix=f'right_{ip_name}')
    s_ip = tt['s', ip_name]
    line.remove(ip_name)
    line.insert([
        env.place(line_solenoid, anchor='center', at=s_ip),
        env.place(ip_name, at=s_ip), # Put back the ip
        env.place(line_comp_solenoid_left, anchor='end', at=-12, from_=ip_name),
        env.place(line_comp_solenoid_right, anchor='start', at=12, from_=ip_name)
    ], s_tol=1e-9)

    # Overlay dipole corrector with solenoid in between 1.23m and 2.29
    tt_region = line.get_table().rows[f'end_ds_start_straight_{ip_name}':f'end_straight_start_ds_{ip_name}']
    s_ip = tt_region['s', ip_name]


    tt_kicker_right= tt_region.rows[s_ip + ds_start: s_ip + ds_end:'s']
    assert np.all(tt_kicker_right.element_type == 'VariableSolenoid')
    l_tot = tt_kicker_right['s_end'][-1] - tt_kicker_right['s_start'][0]

    env[f'acbh1_sol_right_{ip_name}'] = 0
    env[f'acbv1_sol_right_{ip_name}'] = 0
    for nn in tt_kicker_right.name:
        ee = env.get(nn)
        env.ref[nn].knl[0] += env.ref[f'acbh1_sol_right_{ip_name}']/l_tot * ee.length
        env.ref[nn].ksl[0] += env.ref[f'acbv1_sol_right_{ip_name}']/l_tot * ee.length

    tt_kicker_left = tt_region.rows[s_ip - ds_end: s_ip - ds_start:'s']
    assert np.all(tt_kicker_left.element_type == 'VariableSolenoid')
    l_tot = tt_kicker_left['s_end'][-1] - tt_kicker_left['s_start'][0]

    env[f'acbh1_sol_left_{ip_name}'] = 0
    env[f'acbv1_sol_left_{ip_name}'] = 0
    for nn in tt_kicker_left.name:
        ee = env.get(nn)
        env.ref[nn].knl[0] += env.ref[f'acbh1_sol_left_{ip_name}']/l_tot * ee.length
        env.ref[nn].ksl[0] += env.ref[f'acbv1_sol_left_{ip_name}']/l_tot * ee.length

    # Insert markers and dedicated correctors for sol compensation
    line.insert([
        env.new('dy_match_r_'+ip_name, xt.Marker, at=11.95, from_=ip_name),
        env.new('dy_match_l_'+ip_name, xt.Marker, at=-11.95, from_=ip_name),
        env.new(f'corr_sol_right_{ip_name}', xt.Multipole, length=1., isthick=False,
                anchor='end', at=0, from_=f'dy_match_r_{ip_name}@start'),
        env.new(f'corr_sol_left_{ip_name}', xt.Multipole, length=1., isthick=False,
                anchor='start', at=0, from_=f'dy_match_l_{ip_name}@end'),
    ])


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
        'qd6l.2', 'qf5l.2', 'qd4l.2', 'qd3l.2', 'qf2l.2', 'qf1dl.2', 'qf1cl.2',
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

# Start with clean machine
line['on_sol_ipa'] = 0
line['on_sol_ipd'] = 0
line['on_sol_ipg'] = 0
line['on_sol_ipj'] = 0
line['on_comp_sol_ipa'] = 0
line['on_comp_sol_ipd'] = 0
line['on_comp_sol_ipg'] = 0
line['on_comp_sol_ipj'] = 0

optimizers = {}
for ip_name in ip_names:

    print(f'IP {ip_name}:')
    line.cycle(f'end_ds_start_straight_{ip_name}')

    # Reference twiss with solenoids off
    tw0 = line.twiss4d(strengths=True)

    # Switch on solenoid
    line['on_sol_' + ip_name] = 1

    # Switch on compensation solenoid
    line['on_comp_sol_' + ip_name] = 1

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
        # init=tw0, # more noisy on vertical dispersion
        betx=tw0['betx', ip_name],
        bety=tw0['bety', ip_name],
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

    optimizers[ip_name + '_orbit'] = opt_orbit
    optimizers[ip_name + '_optics'] = opt_optics

    # Control all correction with a single knob
    line[f'on_sol_corr_{ip_name}'] = 0

    line[f'on_comp_sol_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_rot_doublet_right_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_rot_doublet_left_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_sol_orbit_corr_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_sol_optics_corr_{ip_name}'] = f'on_sol_corr_{ip_name}'

    # Switch main solenoid off to leave the machine clean for the next IP correction
    line['on_sol_' + ip_name] = 0

# # Cycle to ipa before saving
# line.cycle('ipa')

# line['on_sol_ipa'] = 0
# line['on_sol_ipd'] = 0
# line['on_sol_ipg'] = 0
# line['on_sol_ipj'] = 0
# line['on_sol_corr_ipa'] = 0
# line['on_sol_corr_ipd'] = 0
# line['on_sol_corr_ipg'] = 0
# line['on_sol_corr_ipj'] = 0
# tw_off = line.twiss4d(strengths=True, zero_at=ip_name)

# line['on_sol_ipa'] = 1
# line['on_sol_ipd'] = 1
# line['on_sol_ipg'] = 1
# line['on_sol_ipj'] = 1
# line['on_sol_corr_ipa'] = 1
# line['on_sol_corr_ipd'] = 1
# line['on_sol_corr_ipg'] = 1
# line['on_sol_corr_ipj'] = 1
# tw_on_corr = line.twiss4d(strengths=True, zero_at='ipg')
# # nl_chrom_on_corr = line.get_non_linear_chromaticity(delta0_range=(-1e-2, 1e-2))
# two_on_corr = line.twiss(
#     strengths=True,
#     init=tw_off,
#     init_at='ipg')

line.particle_ref.anomalous_magnetic_moment = 0.00115965218128

# Slice the IP regions
line.cycle('end_ds_start_straight_ipa')
tt = line.get_table()
for ip_name in ip_names:
    s_cut_right = np.arange(tt['s', ip_name] + 2.4, tt['s', ip_name] + 11, 0.2)
    line.cut_at_s(s_cut_right)
    s_cut_left = np.arange(tt['s', ip_name] - 11, tt['s', ip_name] - 2.4, 0.2)
    line.cut_at_s(s_cut_left)

tw4d = line.twiss4d(strengths=True, polarization_analysis=True, radiation_integrals=True)
tw4d['bs'] = tw4d.ks * line.particle_ref.rigidity0[0]

for ip_name in tw4d.rows['ip.*'].name:
    tw4d['bs', ip_name] = np.nan # to avoid seeing zero at ips

t2 = time.time()
print(f'Took {t2-t1} s')

import matplotlib.pyplot as plt
plt.close('all')
ip_plot = 'ipa'
tw_off.zero_at(ip_plot)
tw4d.zero_at(ip_plot)

fig1 = plt.figure(figsize=(6.4, 4.8 * 1.8))
ax1 = fig1.add_subplot(5,1,1)
plty = tw4d.plot(ax=ax1)

ax2 = fig1.add_subplot(5,1,2, sharex=ax1)
ax2.plot(tw4d.s, tw4d.bs)
ax2.set_ylabel(r'$B_s$ [T]')
ax2.grid(True)

ax3 = fig1.add_subplot(5,1,3, sharex=ax1)
ax3.plot(tw4d.s, tw4d.y * 1e3)
ax3.set_ylabel('y [mm]')
ax3.set_ylim(-0.3, 0.3)
ax3.grid(True)

ax4 = fig1.add_subplot(5,1,4, sharex=ax1)
ax4.plot(tw4d.s, tw4d.dy * 1e3)
ax4.set_ylabel(r'$D_y$ [mm]')
ax4.set_ylim(-0.3, 0.3)
ax4.grid(True)

ax5 = fig1.add_subplot(5,1,5, sharex=ax1)
ax5.plot(tw4d.s, tw4d.betx2)
ax5.plot(tw4d.s, tw4d.bety1)
ax5.set_ylabel(r'$\beta_{x2,y1}$')
ax5.grid(True)

ax1.set_xlabel('')
ax5.set_xlabel('s [m]')

fig1.subplots_adjust(hspace=.25, top=0.95, bottom=0.06, left=0.14)

ax5.set_xlim(-20, 20)

out = {
    'sol_half_length': sol_half_length,
    'B0': 'B0',
    'r0': r0,
    'ds_start': ds_start,
    'ds_end': ds_end,
    'gemitt_y': float(tw4d.rad_int_eq_gemitt_y) * 4 # for 4 ips
}


