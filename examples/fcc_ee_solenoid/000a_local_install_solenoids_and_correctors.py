
# Based on E. Boscolo, A. Ciarma, E. Burkhardt, https://cds.cern.ch/record/2948247
# Nuclear Instruments and Methods in Physics Research A 1083 (2026) 171135

import xtrack as xt

from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
import numpy as np

env = xt.load('fccee_z_lcc.json')
line = env.fccee_p_ring

ip_names = ['ipa', 'ipd', 'ipg', 'ipj']

# Tilt with respect to the beam axis
theta = -0.015

for ip_name in ip_names:

    line.cycle(f'end_ds_start_straight_{ip_name}')
    tt = line.get_table()

    print(f'IP {ip_name}:')

    # Analytic field map
    sf = SolenoidField(L=1.23*2, a=0.13, B0=2., z0=0)
    sf_comp_right = SolenoidField(L=0.8, a=0.13, B0=1, z0=1.63)
    sf_comp_left = SolenoidField(L=0.8, a=0.13, B0=1, z0=-1.63)

    # s coordinate along the beam axis
    s = np.linspace(-2.399, 2.399, 201)

    # Corresponding coordinates of the beam reference trajectory in the solenoid frame
    s_sol = s * np.cos(theta)
    x_sol = s * np.sin(theta)
    y_sol = 0 * x_sol

    # Compute field on the beam reference trajectory in the solenoid frame
    bx_sol, by_sol, bz_sol = sf.get_field(x_sol, y_sol, s_sol)
    bx_comp_right_sol, by_comp_right_sol, bz_comp_right_sol = sf_comp_right.get_field(x_sol, y_sol, s_sol)
    bx_comp_left_sol, by_comp_left_sol, bz_comp_left_sol = sf_comp_left.get_field(x_sol, y_sol, s_sol)

    # Transform field to the beam frame
    bx = bx_sol * np.cos(theta) - bz_sol * np.sin(theta)
    bz = bx_sol * np.sin(theta) + bz_sol * np.cos(theta)
    by = by_sol

    bx_comp_right = bx_comp_right_sol * np.cos(theta) - bz_comp_right_sol * np.sin(theta)
    bz_comp_right = bx_comp_right_sol * np.sin(theta) + bz_comp_right_sol * np.cos(theta)
    by_comp_right = by_comp_right_sol * np.cos(theta) - bz_comp_right_sol * np.sin(theta)

    bx_comp_left = bx_comp_left_sol * np.cos(theta) - bz_comp_left_sol * np.sin(theta)
    bz_comp_left = bx_comp_left_sol * np.sin(theta) + bz_comp_left_sol * np.cos(theta)
    by_comp_left = by_comp_left_sol * np.cos(theta) - bz_comp_left_sol * np.sin(theta)

    # Normalized strengths
    rigidity0 = line.particle_ref.rigidity0[0]
    ks = bz / rigidity0
    k0s = bx / rigidity0
    k0 = by / rigidity0

    ks_comp_right = bz_comp_right / rigidity0
    k0s_comp_right = bx_comp_right / rigidity0
    k0_comp_right = by_comp_right / rigidity0

    ks_comp_left = bz_comp_left / rigidity0
    k0s_comp_left = bx_comp_left / rigidity0
    k0_comp_left = by_comp_left / rigidity0

    # Build solenoid slices
    env[f'on_sol_{ip_name}'] = 1
    env[f'field_comp_sol_right_{ip_name}'] = 1.
    env[f'field_comp_sol_left_{ip_name}'] = 1.
    env[f'on_comp_sol_right_{ip_name}'] = 1
    env[f'on_comp_sol_left_{ip_name}'] = 1

    ele_names = []
    for ii in range(len(s)-1):
        ks_entry = ks[ii]
        ks_exit = ks[ii+1]
        k0s_entry = k0s[ii]
        k0s_exit = k0s[ii+1]
        k0_entry = k0[ii]
        k0_exit = k0[ii+1]

        ks_comp_right_entry = ks_comp_right[ii]
        ks_comp_right_exit = ks_comp_right[ii+1]
        k0s_comp_right_entry = k0s_comp_right[ii]
        k0s_comp_right_exit = k0s_comp_right[ii+1]
        k0_comp_right_entry = k0_comp_right[ii]
        k0_comp_right_exit = k0_comp_right[ii+1]
        ks_comp_left_entry = ks_comp_left[ii]
        ks_comp_left_exit = ks_comp_left[ii+1]
        k0s_comp_left_entry = k0s_comp_left[ii]
        k0s_comp_left_exit = k0s_comp_left[ii+1]
        k0_comp_left_entry = k0_comp_left[ii]
        k0_comp_left_exit = k0_comp_left[ii+1]

        s_entry = s[ii]
        s_exit = s[ii+1]

        length = s_exit - s_entry
        s_mid = 0.5 * (s_entry + s_exit)

        ks_entry_var = (ks_entry * env.ref[f'on_sol_{ip_name}']
            + ks_comp_right_entry * env.ref[f'on_comp_sol_right_{ip_name}'] * env.ref[f'field_comp_sol_right_{ip_name}']
            + ks_comp_left_entry * env.ref[f'on_comp_sol_left_{ip_name}'] * env.ref[f'field_comp_sol_left_{ip_name}'])
        ks_exit_var = (ks_exit * env.ref[f'on_sol_{ip_name}']
            + ks_comp_right_exit * env.ref[f'on_comp_sol_right_{ip_name}'] * env.ref[f'field_comp_sol_right_{ip_name}']
            + ks_comp_left_exit * env.ref[f'on_comp_sol_left_{ip_name}'] * env.ref[f'field_comp_sol_left_{ip_name}'])

        k0l_var = (0.5 * length * (k0_exit + k0_entry) * env.ref[f'on_sol_{ip_name}']
            + 0.5 * length * (k0_comp_right_exit + k0_comp_right_entry) * env.ref[f'on_comp_sol_right_{ip_name}'] * env.ref[f'field_comp_sol_right_{ip_name}']
            + 0.5 * length * (k0_comp_left_exit + k0_comp_left_entry) * env.ref[f'on_comp_sol_left_{ip_name}'] * env.ref[f'field_comp_sol_left_{ip_name}'])
        k0sl_var = (0.5 * length * (k0s_exit + k0s_entry) * env.ref[f'on_sol_{ip_name}']
            + 0.5 * length * (k0s_comp_right_exit + k0s_comp_right_entry) * env.ref[f'on_comp_sol_right_{ip_name}'] * env.ref[f'field_comp_sol_right_{ip_name}']
            + 0.5 * length * (k0s_comp_left_exit + k0s_comp_left_entry) * env.ref[f'on_comp_sol_left_{ip_name}'] * env.ref[f'field_comp_sol_left_{ip_name}'])

        env.new(f'sol_slice_{ii}_{ip_name}', xt.VariableSolenoid,
            length=length,
            ks_profile=[ks_entry_var, ks_exit_var],
            knl=[k0l_var],
            ksl=[k0sl_var],
        )
        ele_names.append(f'sol_slice_{ii}_{ip_name}')

    # Force solenoid field at edges to be zero (ax, ay zero at entry and exit)
    env[ele_names[0]].ks_profile[0] = 0
    env[ele_names[-1]].ks_profile[-1] = 0

    # Assemble the solenoid
    line_solenoid = env.new_line(components=ele_names)

    # Measure integrated field of the main solenoid
    env[f'on_sol_{ip_name}'] = 1
    env[f'on_comp_sol_right_{ip_name}'] = 0
    env[f'on_comp_sol_left_{ip_name}'] = 0
    ksol_l_main_solenoid = 0
    for nn in line_solenoid.element_names:
        ee = env.get(nn)
        if isinstance(ee, xt.VariableSolenoid):
            ksol_l_main_solenoid += ee.ks_profile.mean() * ee.length

    # Meaure integrated field of the compensation solenoids
    env[f'on_sol_{ip_name}'] = 0
    env[f'on_comp_sol_right_{ip_name}'] = 1
    env[f'on_comp_sol_left_{ip_name}'] = 0
    ksol_l_comp_solenoid = 0
    for nn in line_solenoid.element_names:
        ee = env.get(nn)
        if isinstance(ee, xt.VariableSolenoid):
            ksol_l_comp_solenoid += ee.ks_profile.mean() * ee.length

    # All back on
    env[f'on_sol_{ip_name}'] = 1
    env[f'on_comp_sol_right_{ip_name}'] = 1
    env[f'on_comp_sol_left_{ip_name}'] = 1

    # Scale to have zero integrated field (main + compensation)
    env[f'field_comp_sol_right_{ip_name}'] = -ksol_l_main_solenoid / ksol_l_comp_solenoid / 2
    env[f'field_comp_sol_left_{ip_name}'] = -ksol_l_main_solenoid / ksol_l_comp_solenoid / 2

    # Put the solenoids in the fcc lattice
    s_ip = tt['s', ip_name]
    line.remove(ip_name)
    line.insert([
        env.place(line_solenoid, anchor='center', at=s_ip),
        env.place(ip_name, at=s_ip), # Put back the ip
    ], s_tol=1e-9)

    # Overlay dipole corrector with solenoid in between 1.23m and 2.29
    tt_region = line.get_table().rows[f'end_ds_start_straight_{ip_name}':f'end_straight_start_ds_{ip_name}']
    s_ip = tt_region['s', ip_name]

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

    # Insert markers and dedicated correctors for sol compensation
    line.insert([
        env.new('dy_match_r_'+ip_name, xt.Marker, at=11.95, from_=ip_name),
        env.new('dy_match_l_'+ip_name, xt.Marker, at=-11.95, from_=ip_name),
        env.new(f'corr_sol_right_{ip_name}', xt.Multipole, length=1., isthick=False,
                anchor='end', at=0, from_=f'dy_match_r_{ip_name}@start'),
        env.new(f'corr_sol_left_{ip_name}', xt.Multipole, length=1., isthick=False,
                anchor='start', at=0, from_=f'dy_match_l_{ip_name}@end'),
    ])

env.to_json('temp_fcc_ee_lcc_local_solenoid.json')

