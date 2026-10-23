"""FCC-ee correction with inspectable state and module-defined optimizers.

003b_correct_with_optimizers.py remains the reference implementation.
Inspect correction.optimizers, each optimizer.settings, and correction.history.
Each correction module computes its own metrics and checks its stopping goals;
correction.compute_metrics(tw) combines diagnostics from all modules.
Twiss is evaluated by correction and supplied to the optimizers explicitly.
Edit FCCeeCorrection.correct_at_fixed_strengths to change the correction sequence.
Set 003c's INPUT to response_api_class/machine.json to analyze the saved line.
"""

from copy import copy
from pathlib import Path

import xtrack as xt
from correction_workflow import FCCeeCorrection
from corr_optics import DEFAULT_OPTICS_CORR_SETTINGS
from corr_orbit import DEFAULT_ORBIT_CORR_SETTINGS
from corr_coupling import DEFAULT_COUPLING_CORR_SETTINGS
from corr_tune import DEFAULT_TUNE_CORR_SETTINGS

# Settings
HERE = Path(__file__).resolve().parent
INPUT = HERE / 'fccee_z_lcc_correction_knobs.json'
OUTPUT_DIR = HERE / 'response_api_class'
TARGET_ERROR_FRACTION = 1.0
INITIAL_INCREMENT = 0.0125
MIN_INCREMENT = 1e-5
MAX_INCREMENT = 0.10

# Independently configured correction modules.
optimizer_settings = dict(
    orbit=copy(DEFAULT_ORBIT_CORR_SETTINGS),
    optics=copy(DEFAULT_OPTICS_CORR_SETTINGS),
    coupling=copy(DEFAULT_COUPLING_CORR_SETTINGS),
    tunes=copy(DEFAULT_TUNE_CORR_SETTINGS),
)

# Construct the correction object
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
line = xt.load(INPUT)
correction = FCCeeCorrection(
    line, intermediate_max_iterations=5, final_max_iterations=40,
    optimizer_settings=optimizer_settings)

# Increase errors and correct, then refine at the requested fraction
correction.run(
    target_error_fraction=TARGET_ERROR_FRACTION,
    initial_increment=INITIAL_INCREMENT,
    min_increment=MIN_INCREMENT,
    max_increment=MAX_INCREMENT,
)

# Save the accepted machine, including a partial result after failure
output = OUTPUT_DIR / ('machine.json' if correction.completed else 'partial.json')
correction.save_result(output)
correction.log('Finished')
if correction.correction_failed:
    raise RuntimeError(
        f'{correction.failure_reason}; stopped at {correction.error_fraction:g}. '
        f'Accepted machine saved in {output}'
    )
