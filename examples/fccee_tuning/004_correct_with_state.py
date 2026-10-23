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
from time import perf_counter
import builtins

import xtrack as xt
from correction_workflow import CorrectionSettings, FCCeeCorrection
from corr_optics import OPTICS_SETTINGS
from corr_orbit import ORBIT_SETTINGS
from corr_coupling import COUPLING_SETTINGS
from corr_tune import TUNE_SETTINGS

START_TIME = perf_counter()


def log(*args, **kwargs):
    kwargs.setdefault('flush', True)
    builtins.print(f'[+{perf_counter() - START_TIME:8.1f} s]', *args, **kwargs)


# %% Settings
HERE = Path(__file__).resolve().parent
INPUT = HERE / 'fccee_z_lcc_correction_knobs.json'
OUTPUT_DIR = HERE / 'response_api_class'
TARGET_ERROR_FRACTION = 1.0
INITIAL_INCREMENT = 0.0125
MIN_INCREMENT = 1e-5
MAX_INCREMENT = 0.10

# Workflow budgets and independently configured correction modules.
settings = CorrectionSettings(intermediate_max_iterations=5, final_max_iterations=40)
optimizer_settings = dict(
    orbit=copy(ORBIT_SETTINGS),
    optics=copy(OPTICS_SETTINGS),
    coupling=copy(COUPLING_SETTINGS),
    tunes=copy(TUNE_SETTINGS),
)

# %% Construct the correction object
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
line = xt.load(INPUT)
correction = FCCeeCorrection(line, settings=settings,
                             optimizer_settings=optimizer_settings, log=log)

# %% Increase errors and correct, then refine at the requested fraction
correction.run(
    target_error_fraction=TARGET_ERROR_FRACTION,
    initial_increment=INITIAL_INCREMENT,
    min_increment=MIN_INCREMENT,
    max_increment=MAX_INCREMENT,
)

# %% Save the accepted machine, including a partial result after failure
output = OUTPUT_DIR / ('machine.json' if correction.completed else 'partial.json')
correction.save_result(output)
log(f'Finished in {perf_counter() - START_TIME:.1f} s')
if correction.correction_failed:
    raise RuntimeError(
        f'{correction.failure_reason}; stopped at {correction.error_fraction:g}. '
        f'Accepted machine saved in {output}'
    )
