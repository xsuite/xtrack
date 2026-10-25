"""FCC-ee correction with inspectable state and module-defined optimizers.

Inspect correction.optimizers, each optimizer.settings, and correction.last_accepted_twiss.
Each correction module computes its own metrics and checks its stopping goals;
correction.compute_metrics(tw) combines diagnostics from all modules.
Twiss is evaluated by correction and supplied to the optimizers explicitly.
Edit FCCeeCorrection.correct_at_fixed_strengths to change the correction sequence.
The corrected line is saved to response_api_class/machine.json.
"""

from copy import copy
from pathlib import Path

import xtrack as xt
from correction_workflow import FCCeeCorrection, prepare_reference, DEFAULT_STEPS_R_MATRIX
from corr_optics import DEFAULT_OPTICS_CORR_SETTINGS
from corr_orbit import DEFAULT_ORBIT_CORR_SETTINGS
from corr_coupling import DEFAULT_COUPLING_CORR_SETTINGS
from corr_tune import DEFAULT_TUNE_CORR_SETTINGS
from corr_chromaticity import DEFAULT_CHROMATICITY_CORR_SETTINGS
from corr_second_order_chrom import DEFAULT_SECOND_ORDER_CHROM_SETTINGS

# Settings
HERE = Path(__file__).resolve().parent
INPUT = HERE / 'fccee_z_lcc_correction_knobs.json'
OUTPUT_DIR = HERE / 'response_api_class'
TARGET_ERROR_FRACTION = 1.0
INITIAL_INCREMENT = 0.0125
MIN_INCREMENT = 1e-5
MAX_INCREMENT = 0.10
STEPS_R_MATRIX = DEFAULT_STEPS_R_MATRIX.copy()

# Independently configured correction modules.
optimizer_settings = dict(
    orbit=copy(DEFAULT_ORBIT_CORR_SETTINGS),
    optics=copy(DEFAULT_OPTICS_CORR_SETTINGS),
    coupling=copy(DEFAULT_COUPLING_CORR_SETTINGS),
    tunes=copy(DEFAULT_TUNE_CORR_SETTINGS),
    chromaticity=copy(DEFAULT_CHROMATICITY_CORR_SETTINGS),
    second_order_chrom=copy(DEFAULT_SECOND_ORDER_CHROM_SETTINGS),
)

# Prepare the clean reference; inspect reference_twiss before building optimizers.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
line = xt.load(INPUT)
reference_twiss = prepare_reference(
    line, delta_chrom=optimizer_settings['second_order_chrom'].delta_chrom,
    steps_R_matrix=STEPS_R_MATRIX)

# Build the optimizers from the shared reference.
correction = FCCeeCorrection(
    line=line, reference_twiss=reference_twiss,
    intermediate_max_iterations=5, final_max_iterations=40,
    optimizer_settings=optimizer_settings, steps_R_matrix=STEPS_R_MATRIX)

# Increase errors and correct, then refine at the requested fraction
success = correction.run(
    target_error_fraction=TARGET_ERROR_FRACTION,
    initial_increment=INITIAL_INCREMENT,
    min_increment=MIN_INCREMENT,
    max_increment=MAX_INCREMENT,
)

# Save the accepted machine, including a partial result after failure
full_correction = success and abs(correction.error_fraction - 1.) < 1e-12
output = OUTPUT_DIR / ('machine.json' if full_correction else 'partial.json')
line.to_json(output, indent=None)
correction.log(f'Saved {output}')
correction.log('Finished')
if correction.correction_failed:
    raise RuntimeError(
        f'{correction.failure_reason}; stopped at {correction.error_fraction:g}. '
        f'Accepted machine saved in {output}'
    )
