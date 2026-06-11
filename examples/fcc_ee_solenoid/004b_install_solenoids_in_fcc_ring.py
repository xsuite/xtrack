from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt


HERE = Path(__file__).parent
INPUT_LATTICE_JSON = HERE / 'fccee_z_lcc.json'
INPUT_SOLENOID_LINES_JSON = HERE / '004_solenoid_lines.json'
OUTPUT_LATTICE_JSON = HERE / 'temp_fcc_ee_lcc_splineboris_solenoids.json'

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']

COMP_SOLENOID_DISTANCE_FROM_IP = 12.0
MAIN_SOLENOID_CORRECTOR_DS_START = 1.23
MAIN_SOLENOID_CORRECTOR_DS_END = 2.29
COMPENSATION_CORRECTOR_MARKER_DS = 11.95
COMPENSATION_CORRECTOR_LENGTH = 1.0

SOLENOID_INSERTION_S_TOL = 1e-8

SET_SOLENOID_KNOBS_FOR_PLOT = True
ON_SOL_VALUE_FOR_PLOT = 1.0
ON_COMP_SOL_VALUE_FOR_PLOT = 1.0


######################################################
# Load the FCC lattice and the isolated solenoid lines #
######################################################

env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring

line_data = xt.json.load(INPUT_SOLENOID_LINES_JSON)
solenoid_templates = {
    name: xt.Line.from_dict(line_dict)
    for name, line_dict in line_data['lines'].items()
}

main_solenoid_template = solenoid_templates['main_solenoid']
comp_solenoid_template = solenoid_templates['compensation_solenoid']


#########################################################
# Install independent SplineBoris solenoid clones at IPs #
#########################################################

for ip_name in IP_NAMES:

    # Use the same local line orientation around each IP as in the original
    # FCC solenoid installation scripts.
    line.cycle(f'end_ds_start_straight_{ip_name}')
    table_before_insertion = line.get_table()
    s_ip = table_before_insertion['s', ip_name]

    print(f'Installing SplineBoris solenoids and correctors around {ip_name}')

    # One knob for the main detector solenoid and one for the two compensation
    # solenoids at this IP.
    env[f'on_sol_{ip_name}'] = 0
    env[f'on_comp_sol_{ip_name}'] = 0

    solenoid_lines = {}
    clone_specs = [
        ('main', main_solenoid_template,
         f'sol_slice_{ip_name}', env.ref[f'on_sol_{ip_name}']),
        ('comp_left', comp_solenoid_template,
         f'comp_sol_slice_left_{ip_name}', env.ref[f'on_comp_sol_{ip_name}']),
        ('comp_right', comp_solenoid_template,
         f'comp_sol_slice_right_{ip_name}', env.ref[f'on_comp_sol_{ip_name}']),
    ]

    # Clone the isolated templates into the environment. The compensation
    # template already carries the scale needed to cancel the main integral;
    # here it is only multiplied by the local on/off knob.
    for clone_name, template_line, element_prefix, knob_ref in clone_specs:
        element_names = []
        name_width = len(str(max(0, len(template_line.element_names) - 1)))

        for ii, template_element in enumerate(template_line.elements):
            element_name = f'{element_prefix}_{ii:0{name_width}d}'
            cloned_element = template_element.copy()

            saved_scale_b = getattr(cloned_element, 'scale_b', 1.0)
            if saved_scale_b is None:
                saved_scale_b = 1.0

            env.elements[element_name] = cloned_element
            env.ref[element_name].scale_b = float(saved_scale_b) * knob_ref
            element_names.append(element_name)

        solenoid_lines[clone_name] = env.new_line(components=element_names)

    # Insert the main solenoid centered on the IP, restore the IP marker at the
    # center, and place the two compensation solenoids symmetrically outside.
    line.remove(ip_name)
    line.insert([
        env.place(solenoid_lines['main'], anchor='center', at=s_ip),
        env.place(ip_name, at=s_ip),
        env.place(solenoid_lines['comp_left'], anchor='end',
                  at=-COMP_SOLENOID_DISTANCE_FROM_IP, from_=ip_name),
        env.place(solenoid_lines['comp_right'], anchor='start',
                  at=COMP_SOLENOID_DISTANCE_FROM_IP, from_=ip_name),
    ], s_tol=SOLENOID_INSERTION_S_TOL)

    # The first corrector on each side is distributed over the outer part of
    # the main solenoid, following the same interval used in 001a.
    env[f'acbh1_sol_right_{ip_name}'] = 0
    env[f'acbv1_sol_right_{ip_name}'] = 0
    env[f'acbh1_sol_left_{ip_name}'] = 0
    env[f'acbv1_sol_left_{ip_name}'] = 0

    table_region = line.get_table().rows[
        f'end_ds_start_straight_{ip_name}':f'end_straight_start_ds_{ip_name}']
    s_ip = table_region['s', ip_name]

    table_corrector_right = table_region.rows[
        s_ip + MAIN_SOLENOID_CORRECTOR_DS_START:
        s_ip + MAIN_SOLENOID_CORRECTOR_DS_END:'s']
    assert np.all(table_corrector_right.element_type == 'SplineBoris')
    assert all(
        env_name.startswith(f'sol_slice_{ip_name}_')
        for env_name in table_corrector_right.env_name)
    length_corrector_right = (
        table_corrector_right['s_end'][-1]
        - table_corrector_right['s_start'][0])

    for env_name in table_corrector_right.env_name:
        element = env.get(env_name)
        env.ref[env_name].knl[0] += (
            env.ref[f'acbh1_sol_right_{ip_name}']
            / length_corrector_right * element.length)
        env.ref[env_name].ksl[0] += (
            env.ref[f'acbv1_sol_right_{ip_name}']
            / length_corrector_right * element.length)

    table_corrector_left = table_region.rows[
        s_ip - MAIN_SOLENOID_CORRECTOR_DS_END:
        s_ip - MAIN_SOLENOID_CORRECTOR_DS_START:'s']
    assert np.all(table_corrector_left.element_type == 'SplineBoris')
    assert all(
        env_name.startswith(f'sol_slice_{ip_name}_')
        for env_name in table_corrector_left.env_name)
    length_corrector_left = (
        table_corrector_left['s_end'][-1]
        - table_corrector_left['s_start'][0])

    for env_name in table_corrector_left.env_name:
        element = env.get(env_name)
        env.ref[env_name].knl[0] += (
            env.ref[f'acbh1_sol_left_{ip_name}']
            / length_corrector_left * element.length)
        env.ref[env_name].ksl[0] += (
            env.ref[f'acbv1_sol_left_{ip_name}']
            / length_corrector_left * element.length)

    # Markers and correctors used later by the orbit/coupling correction near
    # the compensation solenoids.
    line.insert([
        env.new(
            f'dy_match_r_{ip_name}', xt.Marker,
            at=COMPENSATION_CORRECTOR_MARKER_DS,
            from_=ip_name,
        ),
        env.new(
            f'dy_match_l_{ip_name}', xt.Marker,
            at=-COMPENSATION_CORRECTOR_MARKER_DS,
            from_=ip_name,
        ),
        env.new(
            f'corr_sol_right_{ip_name}', xt.Multipole,
            length=COMPENSATION_CORRECTOR_LENGTH,
            isthick=False,
            anchor='end',
            at=0,
            from_=f'dy_match_r_{ip_name}@start',
        ),
        env.new(
            f'corr_sol_left_{ip_name}', xt.Multipole,
            length=COMPENSATION_CORRECTOR_LENGTH,
            isthick=False,
            anchor='start',
            at=0,
            from_=f'dy_match_l_{ip_name}@end',
        ),
    ])


########################
# Save installed lattice #
########################

env.to_json(OUTPUT_LATTICE_JSON)
print(f'Wrote {OUTPUT_LATTICE_JSON}')


###############################################
# Inspect installed SplineBoris mean field Bs #
###############################################

if SET_SOLENOID_KNOBS_FOR_PLOT:
    for ip_name in IP_NAMES:
        env[f'on_sol_{ip_name}'] = ON_SOL_VALUE_FOR_PLOT
        env[f'on_comp_sol_{ip_name}'] = ON_COMP_SOL_VALUE_FOR_PLOT

table = line.get_table(attr=True)
idx_splineboris = np.where(table.element_type == 'SplineBoris')[0]

s_bs_plot_chunks = []
bs_plot_chunks = []
sampled_segments = []
bs_integral_main_solenoids = 0.0
bs_integral_compensation_solenoids = 0.0

for ii in idx_splineboris:
    name = table.name[ii]
    bs_mean = table.bs[ii]
    length = table.length[ii]
    s_ring = np.array([table.s_start[ii], table.s_end[ii]])
    bs_local = np.array([bs_mean, bs_mean])
    bs_integral = bs_mean * length

    if name.startswith('sol_slice_'):
        bs_integral_main_solenoids += bs_integral
    elif name.startswith('comp_sol_slice_'):
        bs_integral_compensation_solenoids += bs_integral

    sampled_segments.append((s_ring, bs_local))

bs_integral_all_solenoids = (
    bs_integral_main_solenoids + bs_integral_compensation_solenoids)

# Keep the plot continuous over each installed solenoid region, while avoiding
# artificial vertical returns to zero at every internal SplineBoris slice.
current_s = []
current_bs = []
previous_s_end = None
for s_ring, bs_local in sampled_segments:
    starts_new_region = (
        previous_s_end is None
        or abs(s_ring[0] - previous_s_end) > 1e-9
    )

    if starts_new_region and current_s:
        s_region = np.concatenate(current_s)
        bs_region = np.concatenate(current_bs)
        s_bs_plot_chunks.append(np.r_[s_region[0], s_region, s_region[-1]])
        bs_plot_chunks.append(np.r_[0.0, bs_region, 0.0])
        current_s = []
        current_bs = []

    if current_s and abs(s_ring[0] - previous_s_end) <= 1e-9:
        current_s.append(s_ring[1:])
        current_bs.append(bs_local[1:])
    else:
        current_s.append(s_ring)
        current_bs.append(bs_local)

    previous_s_end = s_ring[-1]

if current_s:
    s_region = np.concatenate(current_s)
    bs_region = np.concatenate(current_bs)
    s_bs_plot_chunks.append(np.r_[s_region[0], s_region, s_region[-1]])
    bs_plot_chunks.append(np.r_[0.0, bs_region, 0.0])

s_bs = np.concatenate(s_bs_plot_chunks)
bs = np.concatenate(bs_plot_chunks)


########################
# Plot installed fields #
########################

plt.close('all')

fig_bs_ring, ax_bs_ring = plt.subplots(figsize=(12, 5))
ax_bs_ring.plot(s_bs, bs, '-', color='C0', label='SplineBoris $B_s$')
ax_bs_ring.axhline(0.0, color='0.4', linewidth=0.8)
ax_bs_ring.set_xlabel('ring s [m]')
ax_bs_ring.set_ylabel('$B_s$ [T]')
ax_bs_ring.set_title(
    'Longitudinal field from installed SplineBoris solenoids in the FCC ring\n'
    r'$\int B_s ds$ main='
    f'{bs_integral_main_solenoids:.6g} T m, compensation='
    f'{bs_integral_compensation_solenoids:.6g} T m, total='
    f'{bs_integral_all_solenoids:.6g} T m')
ax_bs_ring.grid(True, alpha=0.3)
ax_bs_ring.legend(loc='best')
fig_bs_ring.tight_layout()

print(f'Found {len(idx_splineboris)} SplineBoris elements')
print(f'Extracted {len(s_bs)} B_s plot samples from line attributes')
print(
    f'Integral main solenoids: '
    f'{bs_integral_main_solenoids:.12g} T m')
print(
    f'Integral compensation solenoids: '
    f'{bs_integral_compensation_solenoids:.12g} T m')
print(f'Integral all solenoids: {bs_integral_all_solenoids:.12g} T m')

plt.show()
