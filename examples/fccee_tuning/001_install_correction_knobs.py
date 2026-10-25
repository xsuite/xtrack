"""Install correction knobs, record them in line.metadata, and save the line.

Run after 000_install_errors.py. No correction is performed here.
All settings are below; 002_correct_with_state.py consumes the saved line.
"""

from pathlib import Path
import argparse

import xtrack as xt

import corr_orbit
import corr_optics
import corr_coupling
import corr_tune
import corr_chromaticity
import corr_second_order_chrom

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'fccee_z_lcc_errors.json'
OUTPUT = HERE / 'fccee_z_lcc_correction_knobs.json'
TUNE_KNOBS = ['kqf2', 'kqd1']

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=INPUT)
parser.add_argument('--output', type=Path, default=OUTPUT)
args = parser.parse_args()
INPUT, OUTPUT = args.input, args.output

# Install independent correction circuits on the lattice with errors.
loaded = xt.load(INPUT)
line = loaded if isinstance(loaded, xt.Line) else loaded.fccee_p_ring
env = line.env
if 'tuning_errors' in line.metadata:
    env.metadata['tuning_errors'] = line.metadata['tuning_errors']
if 'tuning_errors' not in env.metadata:
    raise ValueError('Run 000_install_errors.py first')
if 'correction_knobs' in line.metadata:
    raise ValueError('Correction knobs already installed')
if 'orbit_optics_correction' in env.metadata:
    raise ValueError('Input already corrected: use the output of 000')
corr_orbit.install_orbit_knobs(line)

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

# Keep nominal tune families and attach independent correction offsets below.
line.metadata['tuning_errors'] = env.metadata['tuning_errors']
circuits = line.metadata['correction_knobs']
circuits['switches'].update(
    optics='on_corr_optics',
    coupling='on_coupling_corr',
    tunes='on_tune_corr',
    nonlinear='on_nonlinear',
    errors='on_errors',
)
circuits['nonlinear_magnets'] = nonlinear_magnets
corr_optics.install_optics_knobs(line)
skew_knobs = corr_coupling.install_coupling_knobs(line)
print('Independent skew-quadrupole trims:', len(skew_knobs))
corr_tune.install_tune_knobs(line, families=TUNE_KNOBS)
corr_chromaticity.install_chromaticity_knobs(line)
corr_second_order_chrom.install_second_order_chrom_knobs(line)
line.to_json(OUTPUT)
print('Saved', OUTPUT)
