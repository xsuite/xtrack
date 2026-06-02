from pathlib import Path

import matplotlib.pyplot as plt
import xtrack as xt


HERE = Path(__file__).parent
INPUT_LATTICE_JSON = HERE / 'temp_fcc_ee_lcc_splineboris_solenoids.json'
OUTPUT_LATTICE_JSON = (
    HERE / 'fccee_z_lcc_splineboris_solenoids_coupling_corrected.json')

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']


def measure_ksol_l_main_solenoid(line, env, ip_name):
    ksol_l = 0.0
    rigidity0 = line.particle_ref.rigidity0[0]
    table_solenoid_region = line.get_table().rows[
        'dy_match_l_' + ip_name: 'dy_match_r_' + ip_name]

    for nn in table_solenoid_region.name:
        element_type = table_solenoid_region['element_type', nn]
        element = env.get(table_solenoid_region['env_name', nn])

        if element_type == 'VariableSolenoid':
            ksol_l += element.ks_profile.mean() * element.length
        elif element_type == 'SplineBoris':
            ksol_l += element.scale_b * element.bs[4] * element.length / rigidity0

    return ksol_l


#####################################
# Load installed SplineBoris lattice #
#####################################

env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring


##################################################
# Correction configuration copied from 005g setup #
##################################################

config = {}
config['ipa'] = {
    'quad_for_optics_correction': [
        'qd0ar.0', 'qd0br.0', 'qd0cr.0', 'qf1ar.0', 'qf1br.0',
        'qf1cr.0', 'qf1dr.0', 'qf2r.0', 'qd3r.0', 'qd4r.0',
        'qf5r.0', 'qd6r.0', 'qd6l.3', 'qf5l.3', 'qd4l.3',
        'qd3l.3', 'qf2l.3', 'qf1dl.3', 'qf1cl.3', 'qf1bl.3',
        'qf1al.3', 'qd0cl.3', 'qd0bl.3', 'qd0al.3',
    ],
    'doublet_quad_left': [
        'qd0al.3', 'qd0bl.3', 'qd0cl.3', 'qf1al.3', 'qf1bl.3',
        'qf1cl.3', 'qf1dl.3',
    ],
    'doublet_quad_right': [
        'qd0ar.0', 'qd0br.0', 'qd0cr.0', 'qf1ar.0', 'qf1br.0',
        'qf1cr.0', 'qf1dr.0',
    ],
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
        'qd0ar.1', 'qd0br.1', 'qd0cr.1', 'qf1ar.1', 'qf1br.1',
        'qf1cr.1', 'qf1dr.1', 'qf2r.1', 'qd3r.1', 'qd4r.1',
        'qf5r.1', 'qd6r.1', 'qd6l.0', 'qf5l.0', 'qd4l.0',
        'qd3l.0', 'qf2l.0', 'qf1dl.0', 'qf1cl.0', 'qf1bl.0',
        'qf1al.0', 'qd0cl.0', 'qd0bl.0', 'qd0al.0',
    ],
    'doublet_quad_left': [
        'qd0al.0', 'qd0bl.0', 'qd0cl.0', 'qf1al.0', 'qf1bl.0',
        'qf1cl.0', 'qf1dl.0',
    ],
    'doublet_quad_right': [
        'qd0ar.1', 'qd0br.1', 'qd0cr.1', 'qf1ar.1', 'qf1br.1',
        'qf1cr.1', 'qf1dr.1',
    ],
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
        'qd0ar.2', 'qd0br.2', 'qd0cr.2', 'qf1ar.2', 'qf1br.2',
        'qf1cr.2', 'qf1dr.2', 'qf2r.2', 'qd3r.2', 'qd4r.2',
        'qf5r.2', 'qd6r.2', 'qd6l.1', 'qf5l.1', 'qd4l.1',
        'qd3l.1', 'qf2l.1', 'qf1dl.1', 'qf1cl.1', 'qf1bl.1',
        'qf1al.1', 'qd0cl.1', 'qd0bl.1', 'qd0al.1',
    ],
    'doublet_quad_left': [
        'qd0al.1', 'qd0bl.1', 'qd0cl.1', 'qf1al.1', 'qf1bl.1',
        'qf1cl.1', 'qf1dl.1',
    ],
    'doublet_quad_right': [
        'qd0ar.2', 'qd0br.2', 'qd0cr.2', 'qf1ar.2', 'qf1br.2',
        'qf1cr.2', 'qf1dr.2',
    ],
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
        'qd0ar.3', 'qd0br.3', 'qd0cr.3', 'qf1ar.3', 'qf1br.3',
        'qf1cr.3', 'qf1dr.3', 'qf2r.3', 'qd3r.3', 'qd4r.3',
        'qf5r.3', 'qd6r.3', 'qd6l.2', 'qf5l.2', 'qd4l.2',
        'qd3l.2', 'qf2l.2', 'qf1dl.2', 'qf1cl.2', 'qf1bl.2',
        'qf1al.2', 'qd0cl.2', 'qd0bl.2', 'qd0al.2',
    ],
    'doublet_quad_left': [
        'qd0al.2', 'qd0bl.2', 'qd0cl.2', 'qf1al.2', 'qf1bl.2',
        'qf1cl.2', 'qf1dl.2',
    ],
    'doublet_quad_right': [
        'qd0ar.3', 'qd0br.3', 'qd0cr.3', 'qf1ar.3', 'qf1br.3',
        'qf1cr.3', 'qf1dr.3',
    ],
    'corr_1_right_on_quad': 'qd0ar.3',
    'corr_2_right_on_quad': 'qd0br.3',
    'corr_3_right_on_quad': 'qf1ar.3',
    'corr_4_right_on_quad': 'qf1br.3',
    'corr_1_left_on_quad': 'qd0al.2',
    'corr_2_left_on_quad': 'qd0bl.2',
    'corr_3_left_on_quad': 'qf1al.2',
    'corr_4_left_on_quad': 'qf1bl.2',
}


################################
# Build one correction per IP  #
################################

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 0
    line[f'on_comp_sol_{ip_name}'] = 0

optimizers = {}
for ip_name in IP_NAMES:

    print(f'IP {ip_name}:')
    line.cycle(f'end_ds_start_straight_{ip_name}')

    # Reference optics with all solenoids off at this IP.
    tw0 = line.twiss4d(strengths=True)

    # Turn on only the solenoid system being corrected.
    line[f'on_sol_{ip_name}'] = 1
    line[f'on_comp_sol_{ip_name}'] = 1

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

    # Rotate the final doublets by half of the main-solenoid rotation.
    ksol_l_main_solenoid = measure_ksol_l_main_solenoid(line, env, ip_name)
    env[f'phi_rot_doublet_{ip_name}'] = (ksol_l_main_solenoid / 2) / 2
    env[f'on_rot_doublet_left_{ip_name}'] = 1
    env[f'on_rot_doublet_right_{ip_name}'] = 1
    for nn in doublet_quad_left:
        env[nn].rot_s_rad = (
            +env.ref[f'phi_rot_doublet_{ip_name}']
            * env.ref[f'on_rot_doublet_left_{ip_name}'])
    for nn in doublet_quad_right:
        env[nn].rot_s_rad = (
            -env.ref[f'phi_rot_doublet_{ip_name}']
            * env.ref[f'on_rot_doublet_right_{ip_name}'])

    # Orbit corrector knobs. The first pair was installed inside the main
    # solenoid by 004b; the others are attached here to nearby quadrupoles and
    # to the dedicated compensation-solenoid correctors.
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
    env[f'corr_sol_right_{ip_name}'].knl[0] += (
        env.ref[f'acbh6_sol_right_{ip_name}'])

    env[corr_1_left_on_quad].knl[0] += env.ref[f'acbh2_sol_left_{ip_name}']
    env[corr_2_left_on_quad].knl[0] += env.ref[f'acbh3_sol_left_{ip_name}']
    env[corr_3_left_on_quad].knl[0] += env.ref[f'acbh4_sol_left_{ip_name}']
    env[corr_4_left_on_quad].knl[0] += env.ref[f'acbh5_sol_left_{ip_name}']
    env[f'corr_sol_left_{ip_name}'].knl[0] += (
        env.ref[f'acbh6_sol_left_{ip_name}'])

    env[corr_1_right_on_quad].ksl[0] += env.ref[f'acbv2_sol_right_{ip_name}']
    env[corr_2_right_on_quad].ksl[0] += env.ref[f'acbv3_sol_right_{ip_name}']
    env[corr_3_right_on_quad].ksl[0] += env.ref[f'acbv4_sol_right_{ip_name}']
    env[corr_4_right_on_quad].ksl[0] += env.ref[f'acbv5_sol_right_{ip_name}']
    env[f'corr_sol_right_{ip_name}'].ksl[0] += (
        env.ref[f'acbv6_sol_right_{ip_name}'])

    env[corr_1_left_on_quad].ksl[0] += env.ref[f'acbv2_sol_left_{ip_name}']
    env[corr_2_left_on_quad].ksl[0] += env.ref[f'acbv3_sol_left_{ip_name}']
    env[corr_3_left_on_quad].ksl[0] += env.ref[f'acbv4_sol_left_{ip_name}']
    env[corr_4_left_on_quad].ksl[0] += env.ref[f'acbv5_sol_left_{ip_name}']
    env[f'corr_sol_left_{ip_name}'].ksl[0] += (
        env.ref[f'acbv6_sol_left_{ip_name}'])

    # Match orbit and vertical dispersion across the solenoid region.
    opt_orbit = line.match_knob(
        knob_name=f'on_sol_orbit_corr_{ip_name}',
        run=False,
        betx=tw0['betx', ip_name],
        bety=tw0['bety', ip_name],
        start=f'dy_match_l_{ip_name}',
        end=f'dy_match_r_{ip_name}',
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
            xt.TargetSet(x=0, px=0, y=0, py=0, dy=0, dpy=0, at=xt.START),
        ])
    opt_orbit.solve()

    two = line.twiss(
        strengths=True,
        init_at=ip_name,
        betx=tw0['betx', ip_name],
        bety=tw0['bety', ip_name],
    )

    # Match optics and horizontal dispersion with normal quadrupole trims.
    k1_knobs = []
    for nn in quad_for_optics_correction:
        nn_knob = f'k1_{nn}_sol_corr'
        env[nn_knob] = 0
        env[nn].k1 += env.ref[nn_knob]
        k1_knobs.append(nn_knob)

    name_start = f'end_ds_start_straight_{ip_name}'
    name_end = f'end_straight_start_ds_{ip_name}'

    # Skew quadrupole knobs for the additional linear-coupling/vertical-
    # dispersion correction. Use all quadrupoles from the IP to the right edge
    # and from the left edge to the IP.
    table_for_skew = line.get_table()
    k1s_quads_for_coupling_correction = []
    for table_part in (
            table_for_skew.rows[name_start:ip_name],
            table_for_skew.rows[ip_name:name_end]):
        for element_type, env_name in zip(
                table_part.element_type, table_part.env_name):
            if (
                    element_type == 'Quadrupole'
                    and env_name not in k1s_quads_for_coupling_correction):
                k1s_quads_for_coupling_correction.append(env_name)

    k1s_knobs = []
    for nn in k1s_quads_for_coupling_correction:
        nn_knob = f'k1s_{nn}_sol_coupling_corr'
        env[nn_knob] = 0
        env[nn].k1s += env.ref[nn_knob]
        k1s_knobs.append(nn_knob)

    opt_optics = line.match_knob(
        knob_name=f'on_sol_optics_corr_{ip_name}',
        run=False,
        betx=tw0['betx', ip_name],
        bety=tw0['bety', ip_name],
        init_at=ip_name,
        start=name_start,
        end=name_end,
        vary=xt.VaryList(k1_knobs, step=1e-6),
        targets=[
            xt.TargetSet(
                betx=tw0['betx', name_start],
                bety=tw0['bety', name_start],
                tol=1e-5,
                at=xt.START),
            xt.TargetSet(
                alfx=tw0['alfx', name_start],
                alfy=tw0['alfy', name_start],
                tol=1e-8,
                at=xt.START),
            xt.TargetSet(
                dx=tw0['dx', name_start],
                dpx=tw0['dpx', name_start],
                tol=1e-8,
                at=xt.START),
            xt.TargetSet(
                betx=tw0['betx', name_end],
                bety=tw0['bety', name_end],
                tol=1e-5,
                at=xt.END),
            xt.TargetSet(
                alfx=tw0['alfx', name_end],
                alfy=tw0['alfy', name_end],
                tol=1e-8,
                at=xt.END),
            xt.TargetSet(
                dx=tw0['dx', name_end],
                dpx=tw0['dpx', name_end],
                tol=1e-8,
                at=xt.END),
        ])
    opt_optics.solve()

    # Iterate to improve consistency of orbit and optics corrections.
    opt_orbit.solve()
    opt_optics.solve()
    opt_orbit.solve()
    opt_optics.solve()

    # Try an additional correction of linear coupling and vertical dispersion
    # at the straight-section edges using the skew quadrupole knobs.
    opt_coupling = line.match_knob(
        knob_name=f'on_sol_coupling_corr_{ip_name}',
        run=False,
        betx=tw0['betx', ip_name],
        bety=tw0['bety', ip_name],
        init_at=ip_name,
        start=name_start,
        end=name_end,
        vary=xt.VaryList(k1s_knobs, step=1e-6),
        targets=[
            xt.TargetSet(
                betx2=0, bety1=0, alfx2=0, alfy1=0, dy=0, dpy=0,
                tol=1e-8,
                at=xt.START),
            xt.TargetSet(
                betx2=0, bety1=0, alfx2=0, alfy1=0, dy=0, dpy=0,
                tol=1e-8,
                at=xt.END),
        ])

    coupling_correction_solved = True
    try:
        opt_coupling.step(10)
    except Exception as err:
        coupling_correction_solved = False
        print(
            f'Coupling/vertical-dispersion correction failed at {ip_name}: '
            f'{err!r}')
        line.set(k1s_knobs, 0)

    opt_orbit.generate_knob()
    opt_optics.generate_knob()
    if coupling_correction_solved:
        opt_coupling.generate_knob()

    optimizers[f'{ip_name}_orbit'] = opt_orbit
    optimizers[f'{ip_name}_optics'] = opt_optics
    optimizers[f'{ip_name}_coupling'] = opt_coupling

    # One user knob turns on compensation solenoid, doublet rotations, and all
    # generated correction knobs for this IP.
    line[f'on_sol_corr_{ip_name}'] = 0
    line[f'on_comp_sol_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_rot_doublet_right_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_rot_doublet_left_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_sol_orbit_corr_{ip_name}'] = f'on_sol_corr_{ip_name}'
    line[f'on_sol_optics_corr_{ip_name}'] = f'on_sol_corr_{ip_name}'
    if coupling_correction_solved:
        line[f'on_sol_coupling_corr_{ip_name}'] = f'on_sol_corr_{ip_name}'

    # Leave the main solenoid off while preparing the next IP.
    line[f'on_sol_{ip_name}'] = 0


######################
# Save corrected line #
######################

line.cycle('ipa')

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 0
    line[f'on_sol_corr_{ip_name}'] = 0

tw_off = line.twiss4d(strengths=True, zero_at='ipg')

for ip_name in IP_NAMES:
    line[f'on_sol_{ip_name}'] = 1
    line[f'on_sol_corr_{ip_name}'] = 1

tw_on_corr = line.twiss4d(strengths=True, zero_at='ipg')

env.to_json(OUTPUT_LATTICE_JSON)
print(f'Wrote {OUTPUT_LATTICE_JSON}')


################
# Check plots  #
################

plt.close('all')

fig1 = plt.figure(1)
tw_on_corr.rows[-20:20:'s'].plot('betx2 bety1', figure=fig1)

fig2 = plt.figure(2)
tw_on_corr.rows[-20:20:'s'].plot('x y', figure=fig2)

fig3 = plt.figure(3)
ax = fig3.add_subplot(3, 1, 1)
tw_off.plot(ax=ax)

ax2 = fig3.add_subplot(3, 1, 2, sharex=ax)
ax2.plot(tw_off.s, tw_off.muy - tw_on_corr.muy, label='muy error')
ax2.legend(loc='best')

ax3 = fig3.add_subplot(3, 1, 3, sharex=ax)
ax3.plot(tw_off.s, tw_off.muy)
ax3.plot(tw_on_corr.s, tw_on_corr.muy, label='muy with solenoid')
ax3.legend(loc='best')

plt.show()
