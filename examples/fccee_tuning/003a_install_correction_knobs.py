"""Install correction knobs, record them in line.metadata, and save the line.

Run after 000_install_errors.py. No correction is performed here.
All settings are below; 003b_correct_with_optimizers.py consumes the saved line.
"""

from pathlib import Path

import xtrack as xt

from error_recipe import get_bpm_alignment

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'fccee_z_lcc_errors.json'
OUTPUT = HERE / 'fccee_z_lcc_correction_knobs.json'
TUNE_KNOBS = ['kqf2', 'kqd1']

# Install independent correction circuits on the lattice with errors.
env = xt.load(INPUT)
line = env.fccee_p_ring
if 'tuning_errors' not in env.metadata:
    raise ValueError('Run 000_install_errors.py first')
if 'correction_knobs' in line.metadata:
    raise ValueError('Correction knobs already installed')
if 'orbit_optics_correction' in env.metadata:
    raise ValueError('Input already corrected: use the output of 000')
# Complete the presentation's corrector-in-every-quadrupole layout. Existing
# correctors are preserved; missing zero-length trims go at the quad entrance.
added_correctors = []
placements = []
layout = line.get_table()
for name in layout.env_name[layout.element_type == 'Quadrupole']:
    for prefix in ('hcor_', 'vcor_'):
        corrector = prefix + name
        if corrector not in line.element_dict:
            placements.append(
                env.new(corrector, xt.Multipole, knl=[0], ksl=[0], at=name + '@start')
            )
            added_correctors.append(corrector)
if placements:
    line.insert(placements)
print(
    'Added', len(added_correctors), 'H/V correctors to complete the quadrupole layout'
)
monitor_names = list(get_bpm_alignment(env))
corrector_names = [
    [n for n in line.element_names if n.startswith(prefix)]
    for prefix in ('hcor_', 'vcor_')
]
line['on_orbit_corr'] = 1.0
orbit_knobs = []
for plane, names, field, sign in zip('xy', corrector_names, ('knl', 'ksl'), (-1, 1)):
    for name in names:
        knob = f'orbit_corr_{name}_{plane}'
        if knob in line.vars:
            raise ValueError(f'Correction knob already exists: {knob}')
        line[knob] = 0.0
        getattr(line.ref[name], field)[0] += (
            sign * line.ref['on_orbit_corr'] * line.ref[knob]
        )
        orbit_knobs.append(knob)
line.steering_monitors_x = line.steering_monitors_y = monitor_names
line.steering_correctors_x, line.steering_correctors_y = corrector_names
print('BPMs:', len(monitor_names), 'H/V correctors:', [len(n) for n in corrector_names])

line['on_nonlinear'] = 1.0
controlled = []
magnet_types = (
    xt.Bend,
    xt.RBend,
    xt.Quadrupole,
    xt.Sextupole,
    xt.Octupole,
    xt.Multipole,
)
for name in dict.fromkeys(line.element_names):
    element = line.element_dict[name]
    if not isinstance(element, magnet_types):
        continue
    main_is_nonlinear = isinstance(element, (xt.Sextupole, xt.Octupole)) or (
        isinstance(element, xt.Multipole) and element.main_order >= 2
    )
    changed = False
    # Preserve existing family expressions, including currently zero ones.
    for field in ('k2', 'k2s', 'k3', 'k3s'):
        if hasattr(element, field):
            strength = getattr(line.ref[name], field)
            strength *= line.ref['on_nonlinear']
            setattr(line.ref[name], field, strength)
            changed = True
    fields = ['knl', 'ksl']
    # Relative errors in sextupoles/octupoles/etc. already follow their
    # ramped main strength. Multiplying these again would give a squared
    # ramp. Relative nonlinear errors in dipoles/quads need the knob here.
    if not main_is_nonlinear:
        fields += ['knl_rel', 'ksl_rel']
    for field in fields:
        if not hasattr(element, field):
            continue
        for order in range(2, len(getattr(element, field))):
            ref = getattr(line.ref[name], field)[order]
            if getattr(element, field)[order] != 0 or ref._expr is not None:
                getattr(line.ref[name], field)[order] *= line.ref['on_nonlinear']
                changed = True
    if changed:
        controlled.append(name)

nonlinear_magnets = controlled

# Independent skew-quadrupole circuits in sextupoles (one per physical magnet).
# ksl[1] is absolute integrated strength [1/m], and is deliberately not ramped
# with k2. Main sextupoles can be off while the skew trims remain powered.
skew_groups = [
    record['names']
    for record in env.metadata['tuning_errors']['magnets'].values()
    if isinstance(line.get(record['names'][0]), xt.Sextupole)
]
skew_knobs = [f'dk1sl_coupling_{names[0]}' for names in skew_groups]
line['on_coupling_corr'] = 1.0
for names, knob in zip(skew_groups, skew_knobs):
    line[knob] = 0.0
    length = sum(line.get(name).length for name in names)
    for name in names:
        line.ref[name].ksl[1] += (
            line.ref['on_coupling_corr']
            * line.ref[knob]
            * line.get(name).length
            / length
        )
print('Independent skew-quadrupole trims:', len(skew_knobs))


# Independent normal-quadrupole trims.
tt = line.get_table()
quad_names = list(tt.rows[tt.element_type == 'Quadrupole'].name)
quad_knobs = [f'dk1_optics_{name}' for name in quad_names]
line['on_optics_corr'] = 1.0
for name, knob in zip(quad_names, quad_knobs):
    line[knob] = 0.0
    line.ref[name].k1 += line.ref['on_optics_corr'] * line.ref[knob]

# Existing tune-family knobs are reused; keep their nominal values for the ramp.
line.metadata['tuning_errors'] = env.metadata['tuning_errors']
line.metadata['correction_knobs'] = dict(
    orbit={
        plane: dict(
            knobs=[f'orbit_corr_{name}_{plane}' for name in names],
            correctors=names,
            monitors=monitor_names,
            units='rad',
        )
        for plane, names in zip('xy', corrector_names)
    },
    optics=dict(knobs=quad_knobs, quadrupoles=quad_names, units='m^-2'),
    coupling=dict(knobs=skew_knobs, magnet_groups=skew_groups, units='m^-1'),
    tunes=dict(
        knobs=TUNE_KNOBS,
        nominal_values={name: float(line[name]) for name in TUNE_KNOBS},
    ),
    switches=dict(
        orbit='on_orbit_corr',
        optics='on_optics_corr',
        coupling='on_coupling_corr',
        nonlinear='on_nonlinear',
        errors='on_errors',
    ),
    nonlinear_magnets=nonlinear_magnets,
    added_correctors=added_correctors,
)
line.to_json(OUTPUT)
print('Saved', OUTPUT)
