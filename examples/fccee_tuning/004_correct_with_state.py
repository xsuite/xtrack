"""FCC-ee correction with inspectable state and an explicit ramp.

003b_correct_with_optimizers.py remains the reference implementation.
Inspect correction.optimizers, each optimizer.settings, and correction.history.
Each correction module computes its own metrics and checks its stopping goals;
correction.compute_metrics(tw) combines diagnostics from all modules.
Twiss is evaluated by correction and supplied to the optimizers explicitly.
edit/subclass FCCeeCorrection to change the fixed-strength correction sequence.
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
if not 0 < TARGET_ERROR_FRACTION <= 1:
    raise ValueError('TARGET_ERROR_FRACTION must be in (0, 1]')

# Workflow budgets and independently configured correction modules.
settings = CorrectionSettings(ramp_max_iterations=5, final_max_iterations=40)
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

# %% Ramp the errors and nonlinear strengths
correction_failed = False
increment = INITIAL_INCREMENT
while correction.error_fraction < TARGET_ERROR_FRACTION - 1e-12:
    next_error_fraction = correction.error_fraction + increment
    if next_error_fraction > TARGET_ERROR_FRACTION:
        next_error_fraction = TARGET_ERROR_FRACTION
    before = correction.snapshot()
    log(f'Ramp: error fraction {next_error_fraction:g}')

    try:
        correction.scale_corrections(next_error_fraction)
        correction.set_error_fraction(next_error_fraction)
        tw = correction.twiss()
        tw, iterations = correction.correct_at_fixed_strengths(tw)

    except correction.correction_errors as err:
        correction.restore(before)
        increment /= 2
        log(f'Rejected: {err}')
        if increment < MIN_INCREMENT:
            correction_failed = True
            break
        continue

    correction.record(tw, iterations=iterations)
    if iterations <= 3:
        increment *= 1.25
        if increment > MAX_INCREMENT:
            increment = MAX_INCREMENT

# %% Refine at the target error fraction with the tighter final goals
if not correction_failed:
    correction.final_refinement = True
    log('Final refinement')
    if not correction.goals_met(correction.accepted_twiss):
        before = correction.snapshot()
        try:
            tw = correction.accepted_twiss
            tw, iterations = correction.correct_at_fixed_strengths(tw)
            correction.record(tw, iterations=iterations)
        except correction.correction_errors as err:
            correction.restore(before)
            correction_failed = True
            log(f'Refinement failed: {err}')

# %% Save the accepted machine for independent analysis
completed = not correction_failed and abs(correction.error_fraction - 1.0) < 1e-12
output = OUTPUT_DIR / ('machine.json' if completed else 'partial.json')
correction.save_result(
    output,
    completed=completed,
    ramp_settings=dict(
        initial_increment=INITIAL_INCREMENT,
        min_increment=MIN_INCREMENT,
        max_increment=MAX_INCREMENT,
    ),
)
log(f'Finished in {perf_counter() - START_TIME:.1f} s')
if correction_failed:
    raise RuntimeError(
        f'Correction stopped at {correction.error_fraction:g}; accepted machine saved in {output}'
    )
