"""Install the static errors of K. Andre et al., 2 July 2026, slide 7.

Run with an editable Xsuite installation. Output includes deferred expressions,
the seed, sampled errors, and BPM alignment variable names.
"""
from pathlib import Path

import xtrack as xt

from error_recipe import install_errors, get_bpm_alignment


HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / 'fcc_ee_solenoid' / 'fccee_z_lcc.json'
OUTPUT = HERE / 'fccee_z_lcc_errors.json'
SEED = 12345

env = xt.load(INPUT)
line = env.fccee_p_ring
# Keep the input lattice's tracking models and integration settings.
line.configure_radiation(model=None)
install_errors(line, seed=SEED)
env.to_json(OUTPUT)
print(f'Saved {OUTPUT}')
print(line.metadata['tuning_errors']['counts'])

# All error knobs default to one. Examples (also work after reloading the JSON):
# env['on_errors'] = 0                       # recover the input model
# env['on_errors'] = 1
# env['on_error_alignment'] = 0              # mechanical magnet alignment
# env['on_error_girder'] = 0                 # common support offsets
# env['on_error_field'] = 0                  # includes magnetic-centre errors
# env['on_error_field_k1s'] = 0              # all skew quadrupole field errors
# env['on_error_arc_quad_b2'] = 0            # arc quadrupole gradient errors
# env['on_error_random'] = 0                 # retain only systematic errors
# env['on_error_bpm'] = 0                    # residual post-BBA offsets/roll
#
# BPMs are markers, so their alignment is supplied to the correction algorithm:
# alignment = get_bpm_alignment(line)         # evaluate after changing knobs
# correction = line.correct_trajectory(
#     twiss_table=twiss_reference, monitor_alignment=alignment, run=False)
