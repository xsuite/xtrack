from pathlib import Path

import xtrack as xt


HERE = Path(__file__).parent
INPUT_LATTICE_JSON = HERE / 'fccee_z_lcc.json'
LINES_JSON = HERE / '005_solenoid_lines.json'
OUTPUT_LATTICE_JSON = HERE / 'temp_fcc_ee_lcc_non_local_boris_solenoid.json'

IP_NAMES = ['ipa', 'ipd', 'ipg', 'ipj']
COMP_SOLENOID_DISTANCE_FROM_IP = 12.0
MAIN_SOLENOID_CORRECTOR_DS = 1.8
COMPENSATION_CORRECTOR_MARKER_DS = 11.95
COMPENSATION_CORRECTOR_LENGTH = 1.0
SOLENOID_INSERTION_S_TOL = 1e-8
SOLENOID_CORRECTOR_INSERTION_S_TOL = 0.01


# Load the FCC-ee environment and the isolated SplineBoris solenoid lines
# prepared by 005b.
env = xt.load(INPUT_LATTICE_JSON)
line = env.fccee_p_ring
tw0 = line.twiss4d()

line_data = xt.json.load(LINES_JSON)
solenoid_templates = {
    name: xt.Line.from_dict(line_dict)
    for name, line_dict in line_data['lines'].items()
}

main_solenoid_template = solenoid_templates['main_solenoid']
comp_solenoid_template = solenoid_templates['compensation_solenoid']


for ip_name in IP_NAMES:

    # Work in the same local line orientation used by the original 004a script.
    line.cycle(f'end_ds_start_straight_{ip_name}')
    table_before_insertion = line.get_table()
    s_ip = table_before_insertion['s', ip_name]

    print(f'Installing solenoids around {ip_name}')

    # Knobs controlling the main and compensation solenoids at this IP.
    env[f'on_sol_{ip_name}'] = 0
    env[f'on_comp_sol_{ip_name}'] = 0

    # Make independent element clones for this IP. Each clone keeps the base
    # scale saved in 005_solenoid_lines.json and multiplies it by the local knob.
    solenoid_lines = {}
    clone_specs = [
        ('main', main_solenoid_template,
         f'sol_slice_{ip_name}', env.ref[f'on_sol_{ip_name}']),
        ('comp_left', comp_solenoid_template,
         f'comp_sol_slice_left_{ip_name}', env.ref[f'on_comp_sol_{ip_name}']),
        ('comp_right', comp_solenoid_template,
         f'comp_sol_slice_right_{ip_name}', env.ref[f'on_comp_sol_{ip_name}']),
    ]

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

    # Replace the IP marker temporarily so the thick main solenoid can be
    # installed centered on the IP and the marker can be put back at the center.
    line.remove(ip_name)
    line.insert([
        env.place(solenoid_lines['main'], anchor='center', at=s_ip),
        env.place(ip_name, at=s_ip),
        env.place(solenoid_lines['comp_left'], anchor='end',
                  at=-COMP_SOLENOID_DISTANCE_FROM_IP, from_=ip_name),
        env.place(solenoid_lines['comp_right'], anchor='start',
                  at=COMP_SOLENOID_DISTANCE_FROM_IP, from_=ip_name),
    ], s_tol=SOLENOID_INSERTION_S_TOL)

    # Add one horizontal and one vertical knob to each of the two dipole
    # correctors installed inside the main solenoid.
    env[f'acbh1_sol_right_{ip_name}'] = 0
    env[f'acbv1_sol_right_{ip_name}'] = 0
    env[f'acbh1_sol_left_{ip_name}'] = 0
    env[f'acbv1_sol_left_{ip_name}'] = 0

    line.insert([
        env.new(
            f'corr_sol_1_right_{ip_name}', xt.Multipole,
            knl=[env.ref[f'acbh1_sol_right_{ip_name}']],
            ksl=[env.ref[f'acbv1_sol_right_{ip_name}']],
            at=MAIN_SOLENOID_CORRECTOR_DS,
            from_=ip_name,
        ),
        env.new(
            f'corr_sol_1_left_{ip_name}', xt.Multipole,
            knl=[env.ref[f'acbh1_sol_left_{ip_name}']],
            ksl=[env.ref[f'acbv1_sol_left_{ip_name}']],
            at=-MAIN_SOLENOID_CORRECTOR_DS,
            from_=ip_name,
        ),
    ], s_tol=SOLENOID_CORRECTOR_INSERTION_S_TOL)

    # Add the matching markers and the last pair of orbit correctors used by
    # the downstream correction script around the compensation-solenoid region.
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


# Save the environment containing the FCC ring with the solenoid insertions.
env.to_json(OUTPUT_LATTICE_JSON)
print(f'Wrote {OUTPUT_LATTICE_JSON}')
