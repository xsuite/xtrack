"""FCC-ee correction with inspectable state and an explicit ramp.

003b_correct_with_optimizers.py remains the reference implementation.
Inspect correction.optimizers, correction.settings and correction.history;
correction.goal_optimizers expose the aggregate RMS/max stopping targets.
Twiss is evaluated by correction and supplied to the optimizers explicitly.
edit/subclass FCCeeCorrection to change the fixed-strength correction sequence.
Set 003c's INPUT to response_api_class/machine.json to analyze the saved line.
"""

from pathlib import Path
from time import perf_counter
import builtins

import xtrack as xt
from correction_workflow import CorrectionSettings, FCCeeCorrection

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

# All correction settings are inspectable on correction.settings. These are
# the main goals and solve cutoffs; the remaining defaults match 003b.
settings = CorrectionSettings(
    orbit_tol=10e-6,
    beta_rms_tol=0.03,
    beta_max_tol=0.50,
    tune_tol=1e-5,
    optics_rcond=1e-5,
    orbit_rcond=1e-6,
    coupling_rcond=1e-6,
    max_iterations=40,
)

# %% Construct the correction object
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
line = xt.load(INPUT)
correction = FCCeeCorrection(line, settings=settings, log=log)

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
