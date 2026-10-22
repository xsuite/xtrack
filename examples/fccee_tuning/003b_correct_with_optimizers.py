"""Apply the correction algorithm from 001 using Optimize response matrices.

Input: the line saved by 003a_install_correction_knobs.py. Settings are below.
Read the `Correction loop` cell for the workflow. The optimizer construction and
stopping criteria are explicit above it; trial mechanics, observables and I/O
live in correction_steps.py, correction_observables.py and correction_io.py.
Run 003c_analyze_correction.py afterwards for plots and radiation integrals.

A single optimizer shares one Twiss action across all target groups. Activation
selects the response block; repeated selections reuse their cached SVD.

The original correction helpers retain their ordering, trial evaluation,
acceptance, rollback and stopping rules. get_correction(residuals=...) reuses
an existing Twiss result; run_jacobian is deliberately not used to introduce
another line search. The tune response stays fixed at its nominal value.
"""

from pathlib import Path
from functools import partial
from time import perf_counter
import builtins

import numpy as np
import xtrack as xt

from response_builders import (
    normal_quadrupole_response,
    orbit_response,
    skew_quadrupole_response,
)

import correction_observables as obs
import correction_steps as steps
from correction_observables import rms_arrays
from correction_steps import orbit_guess
from correction_io import save_checkpoint, load_checkpoint, history_row, report, save_result

START_TIME = perf_counter()


def print(*args, **kwargs):
    """Timestamp this script's progress messages with cumulative wall time."""
    kwargs.setdefault('flush', True)
    builtins.print(f'[+{perf_counter() - START_TIME:8.1f} s]', *args, **kwargs)


# %% Settings
HERE = Path(__file__).resolve().parent
INPUT = HERE / 'fccee_z_lcc_correction_knobs.json'
OUTPUT_DIR = HERE / 'response_api'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTPUT_DIR / 'machine.json'
RESUME = False
STOP_AT = 1.0
if not 0 < STOP_AT <= 1:
    raise ValueError('STOP_AT must be in (0, 1]')
BETA_RMS_TOL = 0.03
BETA_MAX_TOL = 0.50  # Allow localized peaks while maintaining low global beating.
TUNE_TOL = 1e-5
ALPHA_WEIGHT = 0.1  # Constrain the envelope slope without dominating beta targets.
IR_BETA_WEIGHT = 5.0
IR_HALF_LENGTH = 500.0  # Metres on either side of each interaction point.
HORIZONTAL_DISPERSION_RMS_TOL = 1e-3  # Model Dx - Dx_ref at BPMs plus IPs [m].
HORIZONTAL_DISPERSION_IP_TOL = 100e-6
RAMP_HORIZONTAL_DISPERSION_RMS_TOL = 5e-3
RAMP_HORIZONTAL_DISPERSION_IP_TOL = 500e-6
HORIZONTAL_DISPERSION_WEIGHT = BETA_RMS_TOL / HORIZONTAL_DISPERSION_RMS_TOL  # [1/m]
# Give the collision points the same extra emphasis as IR beta observations.
HORIZONTAL_DISPERSION_IP_WEIGHT = (
    IR_BETA_WEIGHT * HORIZONTAL_DISPERSION_RMS_TOL / HORIZONTAL_DISPERSION_IP_TOL
)
ORBIT_TOL = 10e-6  # RMS at all BPMs, including modes discarded by the SVD.
# Small singular modes amplify inaccuracies of the uncoupled analytical model.
# A directional finite-difference check supports this cutoff.
OPTICS_RCOND = 1e-5
BOOTSTRAP_FRACTION = 0.15  # Error fraction reached before nonlinear magnets turn on.
BOOTSTRAP_RCOND = 1e-7
BOOTSTRAP_BETA_MAX = 0.10
ORBIT_RCOND = 1e-6
COUPLING_RCOND = 1e-6
COUPLING_IP_WEIGHT = 20.0
COUPLING_RMS_TOL = 3e-3
COUPLING_IP_TOL = 2e-3
C_MINUS_TOL = 3e-4
DISPERSION_RMS_TOL = 1e-3  # Model Dy at BPMs plus IPs [m].
DISPERSION_IP_TOL = 20e-6  # Model Dy at the four collision points [m].
RAMP_C_MINUS_TOL = 3e-3
RAMP_COUPLING_RMS_TOL = 3e-2
RAMP_COUPLING_IP_TOL = 3e-2
RAMP_DISPERSION_RMS_TOL = 5e-3
RAMP_DISPERSION_IP_TOL = 100e-6
FINAL_REFINEMENT = False
# Normalize the IP emphasis to its tighter dispersion tolerance. Keep the
# global Dy/RDT weight independent of coupling acceptance tolerances.
DISPERSION_IP_WEIGHT = COUPLING_IP_WEIGHT * DISPERSION_RMS_TOL / DISPERSION_IP_TOL
DISPERSION_WEIGHT = 1.0  # [1/m]
COUPLING_ORBIT_TOL = 0.05e-6  # Settle correctable modes before comparing RDTs.
COUPLING_STEP = 0.5  # Re-steer after each fractional skew-trim update.
MAX_CYCLES = 40
INITIAL_INCREMENT = 0.0125
MIN_INCREMENT = 1e-5
MAX_INCREMENT = 0.10
NONLINEAR_INITIAL_INCREMENT = 0.025
NONLINEAR_MAX_INCREMENT = 0.10
# Nonlinear increments above are fractions of magnetic strength, not progress.
COMBINED_INITIAL_INCREMENT = (1 - BOOTSTRAP_FRACTION) * NONLINEAR_INITIAL_INCREMENT
COMBINED_MAX_INCREMENT = (1 - BOOTSTRAP_FRACTION) * NONLINEAR_MAX_INCREMENT
CHECKPOINT = OUTPUT_DIR / 'checkpoint.json'
HISTORY = OUTPUT_DIR / 'history.csv'


def ramp_settings(progress):
    """Return error and nonlinear fractions on the combined continuation path."""
    if not 0 <= progress <= 1:
        raise ValueError('Ramp progress must be in [0, 1]')
    return progress, max(0.0, (progress - BOOTSTRAP_FRACTION) / (1 - BOOTSTRAP_FRACTION))


print('Starting 4D correction; resume =', RESUME)
print('Combined ramp: nonlinear magnets turn on at error fraction', BOOTSTRAP_FRACTION)
# %% Nominal reference and observation points
line = xt.load(INPUT)
env = line.env
circuits = line.metadata['correction_knobs']
monitor_names = circuits['orbit']['x']['monitors']
corrector_names = [circuits['orbit'][plane]['correctors'] for plane in 'xy']
orbit_knobs = circuits['orbit']['x']['knobs'] + circuits['orbit']['y']['knobs']
quad_names = circuits['optics']['quadrupoles']
quad_knobs = circuits['optics']['knobs']
skew_groups = circuits['coupling']['magnet_groups']
skew_knobs = circuits['coupling']['knobs']
TUNE_KNOBS = circuits['tunes']['knobs']

R_STEPS = dict(dx=1e-8, dpx=1e-9, dy=1e-8, dpy=1e-9, dzeta=1e-7, ddelta=1e-8)
stats = dict(calls=0, seconds=0.0, orbit_calls=0, orbit_seconds=0.0)
compute_twiss = partial(steps.compute_twiss, line=line, stats=stats, steps=R_STEPS)
values = partial(steps.values, line)
set_values = partial(steps.set_values, line)


def full_twiss(seed=None, co_guess=None, radiation_integrals=False):
    return compute_twiss(seed, co_guess, radiation_integrals)


def orbit_twiss(seed=None, co_guess=None):
    return compute_twiss(seed, co_guess, only_orbit=True)


ensure_full_twiss = partial(steps.ensure_full_twiss, full_twiss=full_twiss)

original = circuits['tunes']['nominal_values']


# Each BPM variable includes the master error knob, so evaluate at each step.
def monitor_alignment():
    return [
        np.array(
            [
                env[line.metadata['tuning_errors']['bpm_alignment_variables'][n][a]]
                for n in monitor_names
            ],
            dtype=float,
        )
        for a in ('shift_x', 'shift_y', 'rot_s_rad')
    ]


set_values(orbit_knobs, np.zeros(len(orbit_knobs)))
line['on_errors'] = 0.0
for name in TUNE_KNOBS:
    line[name] = original[name]
line.twiss_default['co_guess'] = dict(x=0, px=0, y=0, py=0, zeta=0, delta=0)
# Verify that one nominal response is suitable for both stages.
tw_nonlinear_on = full_twiss()
line['on_nonlinear'] = 0.0
tw_ref = full_twiss()
nominal_beta_change = max(
    np.max(np.abs(tw_nonlinear_on['bet' + p] / tw_ref['bet' + p] - 1)) for p in 'xy'
)
print('Nominal nonlinear on/off: max beta change', nominal_beta_change)
if nominal_beta_change > 1e-4:
    raise RuntimeError('Aligned optics depend on nonlinear setting; rebuild responses per stage')

tt = line.get_table()
quad_indices = tw_ref.rows.indices[quad_names]
# Add observations at 1/16-turn phase intervals in both planes. Sampling only
# quadrupole entrances can hide beta beating in the intervening drifts and IPs.
selected = set(quad_indices) | set(tw_ref.rows.indices[list(tt.rows['ip.*'].name)])
for plane in 'xy':
    mu = tw_ref['mu' + plane]
    selected.update(np.searchsorted(mu, np.arange(mu[0], mu[-1], 1 / 16)).tolist())
obs_indices = np.array(sorted(selected))
obs_names = list(tw_ref.name[obs_indices])
alpha_indices = tw_ref.rows.indices[list(dict.fromkeys(quad_names + list(tt.rows['ip.*'].name)))]
monitor_indices = tw_ref.rows.indices[monitor_names]
ip_names = ['ipa', 'ipd', 'ipg', 'ipj']
ip_indices = tw_ref.rows.indices[ip_names]
dispersion_names = list(dict.fromkeys(monitor_names + ip_names))
dispersion_indices = tw_ref.rows.indices[dispersion_names]

# Residual definitions are separate from the optimizers that consume them.
optics_vector = partial(
    obs.optics_vector,
    tw_ref=tw_ref,
    obs_indices=obs_indices,
    alpha_indices=alpha_indices,
    dispersion_indices=dispersion_indices,
)
coupling_vector = partial(obs.coupling_vector, tw_ref=tw_ref, coupling_indices=dispersion_indices)
orbit_errors = partial(
    obs.orbit_errors, monitor_indices=monitor_indices, monitor_alignment=monitor_alignment
)
lab_orbit_errors = partial(
    obs.lab_orbit_errors, monitor_indices=monitor_indices, monitor_alignment=monitor_alignment
)
measure = partial(
    obs.measure,
    tw_ref=tw_ref,
    dispersion_indices=dispersion_indices,
    ip_indices=ip_indices,
    orbit_errors=orbit_errors,
)
horizontal_dispersion_errors = partial(
    obs.horizontal_dispersion_errors, tw_ref=tw_ref, dispersion_indices=dispersion_indices
)


set_values(orbit_knobs, np.zeros(len(orbit_knobs)))
line['on_errors'] = 0.0
for name in TUNE_KNOBS:
    line[name] = original[name]
# %% Fixed nominal responses and target weights
print(f'Using {len(quad_knobs)} installed quadrupole trims', flush=True)
print('Preparing nominal optics SVD', flush=True)
response_started = perf_counter()
nominal_response, strength_scale = normal_quadrupole_response(
    line,
    tw_ref,
    quad_names,
    obs_names,
    list(tw_ref.name[alpha_indices]),
    dispersion_names,
)
dx_weights = HORIZONTAL_DISPERSION_WEIGHT * np.array(
    [HORIZONTAL_DISPERSION_IP_WEIGHT if n in ip_names else 1.0 for n in dispersion_names]
)
dx_weights /= np.sqrt(len(dispersion_names))
optics_weights = np.r_[
    np.full(2 * len(obs_indices), 1 / np.sqrt(len(obs_indices))),
    np.full(2 * len(alpha_indices), ALPHA_WEIGHT / np.sqrt(len(alpha_indices))),
    dx_weights,
    1000.0,
    1000.0,
]
ip_s = tw_ref.s[tw_ref.rows.indices[list(tt.rows['ip.*'].name)]]
ip_distance = np.abs(tw_ref.s[obs_indices, None] - ip_s[None, :])
near_ip = np.min(np.minimum(ip_distance, tw_ref.line_length - ip_distance), axis=1) < IR_HALF_LENGTH
for k in range(2):
    optics_weights[k * len(obs_indices) : (k + 1) * len(obs_indices)] *= np.where(
        near_ip, IR_BETA_WEIGHT, 1.0
    )
print(f'Upweighting beta at {np.sum(near_ip)} interaction-region observations', flush=True)
steps.check_optics_response(
    line,
    quad_knobs=quad_knobs,
    strength_scale=strength_scale,
    nominal_response=nominal_response,
    optics_weights=optics_weights,
    tw_ref=tw_ref,
    full_twiss=full_twiss,
    optics_vector=optics_vector,
    horizontal_dispersion_errors=horizontal_dispersion_errors,
    n_dispersion=len(dispersion_names),
    log=print,
)
# Labels make response rows inspectable; weights and knob scales become
# ordinary Target/Vary settings in the optimizer below.
optics_labels = (
    [f'log_beta_{p}@{n}' for p in 'xy' for n in obs_names]
    + [f'normalized_alpha_{p}@{n}' for p in 'xy' for n in tw_ref.name[alpha_indices]]
    + [f'dx@{n}' for n in dispersion_names]
    + ['qx', 'qy']
)
coupling_names = dispersion_names
coupling_indices = dispersion_indices
coupling_response, skew_scale = skew_quadrupole_response(line, tw_ref, skew_groups, coupling_names)
coupling_weights = np.tile(
    np.array([COUPLING_IP_WEIGHT if n in ip_names else 1.0 for n in coupling_names]), 4
)
coupling_weights = np.r_[
    coupling_weights,
    DISPERSION_WEIGHT
    * np.array([DISPERSION_IP_WEIGHT if n in ip_names else 1.0 for n in coupling_names]),
]
coupling_weights /= np.sqrt(len(coupling_names))


steps.check_coupling_response(
    line,
    skew_knobs=skew_knobs,
    skew_scale=skew_scale,
    coupling_response=coupling_response,
    tw_ref=tw_ref,
    full_twiss=full_twiss,
    coupling_vector=coupling_vector,
    n_observations=len(coupling_names),
    log=print,
)
coupling_labels = [
    f'{part}_{rdt}@{n}'
    for rdt in ('f1001', 'f1010')
    for part in ('real', 'imag')
    for n in coupling_names
] + [f'dy@{n}' for n in coupling_names]
print('Preparing aligned 2x2 tune response (two Twiss probes)')
tune_response = steps.build_tune_response(
    tw_ref, line=line, tune_knobs=TUNE_KNOBS, full_twiss=full_twiss
)

# %% One optimizer, one action, independently selectable response blocks
# The optics block keeps its own qx/qy rows and quadrupole-to-tune response.
# The tune block fits those same physical observables with the arc-family knobs.
# Separate target rows preserve the original weights and independent solves.
responses = dict(
    orbit_x=orbit_response(tw_ref, monitor_names, corrector_names[0], 'x'),
    orbit_y=orbit_response(tw_ref, monitor_names, corrector_names[1], 'y'),
    tunes=tune_response,
    coupling=coupling_response,
    optics=nominal_response,
)
knobs = dict(
    orbit_x=circuits['orbit']['x']['knobs'],
    orbit_y=circuits['orbit']['y']['knobs'],
    tunes=TUNE_KNOBS,
    coupling=skew_knobs,
    optics=quad_knobs,
)
scales = dict(
    orbit_x=np.ones(len(knobs['orbit_x'])),
    orbit_y=np.ones(len(knobs['orbit_y'])),
    tunes=np.ones(2),
    coupling=skew_scale,
    optics=strength_scale,
)
weights = dict(
    orbit_x=np.ones(len(monitor_names)),
    orbit_y=np.ones(len(monitor_names)),
    tunes=np.ones(2),
    coupling=coupling_weights,
    optics=optics_weights,
)
labels = dict(
    orbit_x=[f'x@{n}' for n in monitor_names],
    orbit_y=[f'y@{n}' for n in monitor_names],
    tunes=['qx', 'qy'],
    coupling=coupling_labels,
    optics=optics_labels,
)
cutoffs = dict(
    orbit_x=ORBIT_RCOND,
    orbit_y=ORBIT_RCOND,
    tunes=0.0,
    coupling=COUPLING_RCOND,
    optics=OPTICS_RCOND,
)

# Every block occupies a known slice of the full target and knob vectors.
rows, columns = {}, {}
n_rows = n_columns = 0
for group, response in responses.items():
    rows[group] = slice(n_rows, n_rows + response.shape[0])
    columns[group] = slice(n_columns, n_columns + response.shape[1])
    n_rows += response.shape[0]
    n_columns += response.shape[1]


def residuals_from_twiss(tw):
    # One Twiss supplies all groups, including the two copies of the tune residuals.
    x, y = lab_orbit_errors(tw)
    return np.r_[x, y, tw.qx - tw_ref.qx, tw.qy - tw_ref.qy, coupling_vector(tw), optics_vector(tw)]


measurement = xt.Action(lambda: residuals_from_twiss(full_twiss()))
opt = line.match(
    solve=False,
    name='FCC-ee correction',
    vary=[
        xt.Vary(knob, step=1e-7, weight=scale, tag=group)
        for group in responses
        for knob, scale in zip(knobs[group], scales[group])
    ],
    targets=[
        measurement.target(
            rows[group].start + i,
            value=0.0,
            tol=TUNE_TOL / 2 if group == 'tunes' else 1e-15,
            weight=weight,
            tag=f'{group}@{label}',
        )
        for group in responses
        for i, (label, weight) in enumerate(zip(labels[group], weights[group]))
    ],
    assert_within_tol=False,
    show_call_counter=False,
)
# One logical block diagonal response. Blocks retain separate SVDs and cutoffs;
# the zero cross terms are not allocated, even when several groups are active.
opt.set_response(
    blocks={
        group: dict(matrix=response, targets=rows[group], vary=columns[group], rcond=cutoffs[group])
        for group, response in responses.items()
    }
)


def select_group(group):
    opt.disable(target=True, vary=True)
    opt.enable(target=f'{group}@.*', vary=group)


def correction(group, residuals, rcond=None):
    # Reuse an existing measurement, including orbit-only Twiss tables.
    select_group(group)
    all_residuals = np.zeros(n_rows)
    all_residuals[rows[group]] = residuals
    return opt.get_correction(all_residuals, rcond=rcond)[columns[group]]


svds = {}
for group in responses:
    select_group(group)
    svds[group] = opt.response.svd
    singular_values = svds[group].s
    print(
        group,
        'SVD range',
        singular_values[[0, -1]],
        'retained',
        np.sum(singular_values > cutoffs[group] * singular_values[0]),
    )
opt_s = svds['optics'].s
print(f'Response construction and SVDs: {perf_counter() - response_started:.1f} s')


# %% Stopping criteria and trial steps
def coupling_goal(tw):
    m = measure(tw, orbit_errors=None)
    rdt_rms, rdt_ip, c_minus, dy_rms, dy_ip = (
        (COUPLING_RMS_TOL, COUPLING_IP_TOL, C_MINUS_TOL, DISPERSION_RMS_TOL, DISPERSION_IP_TOL)
        if FINAL_REFINEMENT
        else (
            RAMP_COUPLING_RMS_TOL,
            RAMP_COUPLING_IP_TOL,
            RAMP_C_MINUS_TOL,
            RAMP_DISPERSION_RMS_TOL,
            RAMP_DISPERSION_IP_TOL,
        )
    )
    return (
        np.all(m['coupling_rms'] < rdt_rms)
        and m['coupling_ip'] < rdt_ip
        and m['c_minus'] < c_minus
        and m['dy_rms'] < dy_rms
        and m['dy_ip'] < dy_ip
    )


def optical_goal(tw):
    m = measure(tw, orbit_errors=None)
    peak_limit = BOOTSTRAP_BETA_MAX if line['on_errors'] <= BOOTSTRAP_FRACTION else BETA_MAX_TOL
    dx_rms, dx_ip = (
        (HORIZONTAL_DISPERSION_RMS_TOL, HORIZONTAL_DISPERSION_IP_TOL)
        if FINAL_REFINEMENT
        else (RAMP_HORIZONTAL_DISPERSION_RMS_TOL, RAMP_HORIZONTAL_DISPERSION_IP_TOL)
    )
    return (
        np.all(m['beta_rms'] < BETA_RMS_TOL)
        and max(m['beta_max']) < peak_limit
        and m['dx_rms'] < dx_rms
        and m['dx_ip'] < dx_ip
        and max(abs(m['tune_error'])) < TUNE_TOL
    )


def optical_cost(tw):
    # Tunes are checked separately after numerical arc-family cleanup.
    return np.linalg.norm((optics_weights * optics_vector(tw))[:-2])


def coupling_cost(tw):
    return np.linalg.norm(coupling_weights * coupling_vector(tw))


def optical_direction(tw):
    cutoff = BOOTSTRAP_RCOND if line['on_errors'] <= BOOTSTRAP_FRACTION else OPTICS_RCOND
    keep = opt_s > cutoff * opt_s[0]
    print('  optics SVD cutoff', cutoff, 'modes', np.sum(keep), flush=True)
    return correction('optics', optics_vector(tw), rcond=cutoff)


def steering_direction(tw, fraction):
    return np.concatenate(
        [
            correction(group, error, rcond=ORBIT_RCOND)
            for group, error in zip(('orbit_x', 'orbit_y'), lab_orbit_errors(tw))
        ]
    )


# Trial steps accept/reject their own candidates; the loops below own the ramp.
correctable_orbit_errors = partial(
    obs.correctable_orbit_errors,
    lab_orbit_errors=lab_orbit_errors,
    orbit_svds=[svds['orbit_x'], svds['orbit_y']],
    rcond=ORBIT_RCOND,
)
tune_cleanup = partial(
    steps.tune_cleanup,
    line=line,
    tune_knobs=TUNE_KNOBS,
    tw_ref=tw_ref,
    tune_tol=TUNE_TOL,
    tune_direction=partial(correction, 'tunes', rcond=0.0),
    ensure_full_twiss=ensure_full_twiss,
    full_twiss=full_twiss,
    log=print,
)
orbit_step = partial(
    steps.orbit_step,
    line=line,
    orbit_knobs=orbit_knobs,
    steering_direction=steering_direction,
    orbit_errors=orbit_errors,
    orbit_twiss=orbit_twiss,
    full_twiss=full_twiss,
    log=print,
)
optics_step = partial(
    steps.optics_step,
    line=line,
    quad_knobs=quad_knobs,
    tune_knobs=TUNE_KNOBS,
    orbit_knobs=orbit_knobs,
    optical_direction=optical_direction,
    optical_cost=optical_cost,
    strength_scale=strength_scale,
    orbit_twiss=orbit_twiss,
    orbit_step=orbit_step,
    tune_cleanup=tune_cleanup,
    measure=measure,
    log=print,
)
settle_coupling_orbit = partial(
    steps.settle_coupling_orbit,
    line=line,
    correctable_orbit_errors=correctable_orbit_errors,
    coupling_orbit_tol=COUPLING_ORBIT_TOL,
    orbit_step=orbit_step,
    tune_cleanup=tune_cleanup,
)
coupling_step = partial(
    steps.coupling_step,
    line=line,
    skew_knobs=skew_knobs,
    orbit_knobs=orbit_knobs,
    tune_knobs=TUNE_KNOBS,
    coupling_direction=lambda tw: correction('coupling', coupling_vector(tw), rcond=COUPLING_RCOND),
    step_fraction=COUPLING_STEP,
    coupling_cost=coupling_cost,
    coupling_goal=coupling_goal,
    settle_coupling_orbit=settle_coupling_orbit,
    orbit_twiss=orbit_twiss,
    measure=measure,
    log=print,
)

# %% Initial state / resume
# Start from nominal settings, or restore the last accepted checkpoint.
set_values(orbit_knobs, np.zeros(len(orbit_knobs)))
set_values(quad_knobs, np.zeros(len(quad_knobs)))
line['on_errors'] = 0
for name in TUNE_KNOBS:
    line[name] = original[name]
ramp_tw = full_twiss(tw_ref)
ramp_fraction = 0.0
ramp_history = []
if RESUME:
    ramp_fraction, ramp_history, tune_response = load_checkpoint(
        line, CHECKPOINT, BOOTSTRAP_FRACTION
    )
    if not np.array_equal(tune_response, responses['tunes']):
        responses['tunes'] = tune_response
        opt.set_response(
            blocks={
                group: dict(
                    matrix=response, targets=rows[group], vary=columns[group], rcond=cutoffs[group]
                )
                for group, response in responses.items()
            }
        )
    ramp_tw = full_twiss()
    np.testing.assert_allclose(
        rms_arrays(orbit_errors(ramp_tw)),
        ramp_history[-1][1:3],
        rtol=1e-5,
        atol=1e-10,
        err_msg='Checkpoint BPM orbit was not restored',
    )
    np.testing.assert_allclose(
        [ramp_tw.qx, ramp_tw.qy],
        ramp_history[-1][7:9],
        rtol=0,
        atol=1e-7,
        err_msg='Checkpoint tunes were not restored',
    )
    report(print, 'RESUMED', measure(ramp_tw))
    np.testing.assert_allclose(
        [line['on_errors'], line['on_nonlinear']],
        ramp_settings(ramp_fraction),
        rtol=0,
        atol=1e-12,
        err_msg='Checkpoint knobs disagree with the combined ramp',
    )
    if STOP_AT < ramp_fraction:
        raise ValueError('STOP_AT precedes the checkpoint')
# %% Correction loop
# Ramp errors/nonlinear strengths, then tighten the goals at the same endpoint.
reached_stop = True
for final_pass in (False, True):
    if final_pass:
        if not reached_stop or abs(ramp_fraction - 1.0) >= 1e-12:
            break
        print('FINAL REFINEMENT at full errors and nonlinear strengths')
    FINAL_REFINEMENT = final_pass
    stop_fraction = 1.0 if FINAL_REFINEMENT else STOP_AT
    increment = (
        max(
            INITIAL_INCREMENT,
            min(MAX_INCREMENT, ramp_history[-1][0] - ramp_history[-2][0]),
        )
        if len(ramp_history) > 1
        else INITIAL_INCREMENT
    )
    if ramp_fraction >= BOOTSTRAP_FRACTION:
        increment = min(increment, COMBINED_INITIAL_INCREMENT)
    control_names = (
        orbit_knobs + quad_knobs + skew_knobs + TUNE_KNOBS + ['on_errors', 'on_nonlinear']
    )
    # Refit an accepted checkpoint if targets changed (e.g. adding Dy), even
    # if it already reached the requested ramp endpoint. Never silently reuse
    # an endpoint that fails the current goals.
    refine = ramp_fraction > 0 and not (
        optical_goal(ramp_tw)
        and coupling_goal(ramp_tw)
        and np.all(
            rms_arrays(orbit_errors(ramp_tw)) <= max(0.002e-6, min(ramp_fraction, 1.0) * ORBIT_TOL)
        )
    )
    while refine or ramp_fraction < stop_fraction - 1e-12:
        boundary = (
            min(BOOTSTRAP_FRACTION, stop_fraction)
            if ramp_fraction < BOOTSTRAP_FRACTION - 1e-12
            else stop_fraction
        )
        fraction = ramp_fraction if refine else min(boundary, ramp_fraction + increment)
        baseline = values(control_names)
        stage_started = perf_counter()
        calls_before, seconds_before = stats['calls'], stats['seconds']
        print('TRY', fraction, 'increment', increment, flush=True)
        try:
            fraction, nonlinear = ramp_settings(fraction)
            previous = float(line['on_errors'])
            guess = orbit_guess(ramp_tw)
            # Extrapolate the accepted correction to the proposed error strength.
            if previous > 0 and fraction != previous:
                ratio = fraction / previous
                set_values(orbit_knobs, ratio * values(orbit_knobs))
                set_values(quad_knobs, ratio * values(quad_knobs))
                set_values(skew_knobs, ratio * values(skew_knobs))
                families = np.array([original[name] for name in TUNE_KNOBS])
                set_values(TUNE_KNOBS, families + ratio * (values(TUNE_KNOBS) - families))
                reference_guess = orbit_guess(tw_ref)
                guess = {
                    key: reference_guess[key] + ratio * (value - reference_guess[key])
                    for key, value in guess.items()
                }
            line['on_errors'] = fraction
            line['on_nonlinear'] = nonlinear
            # Steer before tune cleanup: a large uncorrected orbit can shift the tunes.
            tw = full_twiss(co_guess=guess)
            orbit_tolerance = max(0.002e-6, fraction * ORBIT_TOL)
            # Correct at fixed strengths; re-check goals after each accepted step.
            for cycle in range(MAX_CYCLES):
                orbit_ok = np.all(rms_arrays(orbit_errors(tw)) <= orbit_tolerance)
                optics_ok = optical_goal(tw)
                coupling_ok = coupling_goal(tw)
                if orbit_ok and optics_ok and coupling_ok:
                    break
                print(
                    ' cycle',
                    cycle + 1,
                    'errors/nonlinear',
                    fraction,
                    nonlinear,
                    'orbit/optics/coupling OK',
                    orbit_ok,
                    optics_ok,
                    coupling_ok,
                )
                if not orbit_ok or not optics_ok:
                    try:
                        candidate = orbit_step(tw, fraction)
                    except RuntimeError:
                        if not orbit_ok:
                            raise
                    else:
                        tw = tune_cleanup(candidate)
                if not coupling_goal(tw):
                    tw = coupling_step(tw)
                if not optical_goal(tw):
                    tw = optics_step(tw)
            else:
                raise RuntimeError('Interleaved stage iteration limit')
            candidate, cycles = tw, cycle
        except (ValueError, RuntimeError, xt.twiss.ClosedOrbitSearchError) as err:
            set_values(control_names, baseline)
            line['on_errors'], line['on_nonlinear'] = ramp_settings(ramp_fraction)
            line.twiss_default['co_guess'] = orbit_guess(ramp_tw)
            increment /= 2
            print(
                'REJECT',
                type(err).__name__,
                str(err),
                'restored',
                ramp_fraction,
                'increment',
                increment,
            )
            if refine or increment < MIN_INCREMENT:
                reached_stop = False
                break
            continue
        finally:
            print(
                f'  Stage timing: {perf_counter() - stage_started:.1f} s total; '
                f'{stats["seconds"] - seconds_before:.1f} s in '
                f'{stats["calls"] - calls_before} Twiss calls',
                flush=True,
            )
        ramp_tw, ramp_fraction = candidate, fraction
        metrics = measure(ramp_tw)
        row = history_row(ramp_tw, fraction, line, metrics)
        if refine:
            ramp_history[-1] = row
        else:
            ramp_history.append(row)
        refine = False
        report(print, f'ACCEPT progress {fraction:g}', metrics)
        checkpoint_started = perf_counter()
        line.twiss_default['co_guess'] = orbit_guess(ramp_tw)
        save_checkpoint(
            line,
            path=CHECKPOINT,
            history_path=HISTORY,
            input_path=INPUT,
            bootstrap_fraction=BOOTSTRAP_FRACTION,
            tw=ramp_tw,
            progress=ramp_fraction,
            final_refinement=FINAL_REFINEMENT,
            tune_response=tune_response,
            history=ramp_history,
        )
        if fraction == BOOTSTRAP_FRACTION:
            line.to_json(OUTPUT_DIR / 'bootstrap.json', indent=None)
            print('  Saved full lattice at the end of the errors-only bootstrap')
        print(f'  Checkpoint and history save: {perf_counter() - checkpoint_started:.3f} s')
        if fraction == BOOTSTRAP_FRACTION:
            increment = COMBINED_INITIAL_INCREMENT
        elif cycles <= (3 if fraction > BOOTSTRAP_FRACTION else 5):
            limit = COMBINED_MAX_INCREMENT if fraction > BOOTSTRAP_FRACTION else MAX_INCREMENT
            increment = min(limit, increment * 1.25)

ramp_completed = reached_stop and abs(ramp_fraction - 1.0) < 1e-12

# %% Save the accepted machine
# Final verification and independently reloadable output.
tw_after = full_twiss(ramp_tw)
metrics = measure(tw_after)
if ramp_completed:
    assert np.all(rms_arrays(orbit_errors(tw_after)) <= ORBIT_TOL)
    assert optical_goal(tw_after) and coupling_goal(tw_after)
line.twiss_default['co_guess'] = orbit_guess(tw_after)
line.twiss_default['steps_R_matrix'] = R_STEPS.copy()
print(
    'Ramp completed:', ramp_completed, 'errors/nonlinear:', line['on_errors'], line['on_nonlinear']
)
report(print, 'AFTER', metrics)
if not ramp_completed:
    OUTPUT = OUTPUT_DIR / 'partial.json'
save_result(
    line,
    OUTPUT,
    tw_ref=tw_ref,
    progress=ramp_fraction,
    completed=ramp_completed,
    metrics=metrics,
    settings={
        'nonlinear_initial_increment': NONLINEAR_INITIAL_INCREMENT,
        'nonlinear_max_increment': NONLINEAR_MAX_INCREMENT,
        'horizontal_dispersion_rms_tolerance_m': HORIZONTAL_DISPERSION_RMS_TOL,
        'horizontal_dispersion_ip_tolerance_m': HORIZONTAL_DISPERSION_IP_TOL,
        'horizontal_dispersion_weight_per_m': HORIZONTAL_DISPERSION_WEIGHT,
        'horizontal_dispersion_ip_weight': HORIZONTAL_DISPERSION_IP_WEIGHT,
        'dispersion_rms_tolerance_m': DISPERSION_RMS_TOL,
        'dispersion_ip_tolerance_m': DISPERSION_IP_TOL,
        'dispersion_weight_per_m': DISPERSION_WEIGHT,
        'dispersion_ip_weight': DISPERSION_IP_WEIGHT,
        'coupling_rms_tolerance': COUPLING_RMS_TOL,
        'coupling_ip_tolerance': COUPLING_IP_TOL,
        'c_minus_tolerance': C_MINUS_TOL,
        'ramp_targets': {
            'c_minus': RAMP_C_MINUS_TOL,
            'rdt_rms': RAMP_COUPLING_RMS_TOL,
            'rdt_ip_max': RAMP_COUPLING_IP_TOL,
            'dispersion_rms_m': RAMP_DISPERSION_RMS_TOL,
            'dispersion_ip_max_m': RAMP_DISPERSION_IP_TOL,
            'horizontal_dispersion_rms_m': RAMP_HORIZONTAL_DISPERSION_RMS_TOL,
            'horizontal_dispersion_ip_max_m': RAMP_HORIZONTAL_DISPERSION_IP_TOL,
        },
        'optics_svd_rcond': OPTICS_RCOND,
        'optics_svd_modes': int(np.sum(opt_s > OPTICS_RCOND * opt_s[0])),
        'bootstrap_fraction': BOOTSTRAP_FRACTION,
        'bootstrap_svd_rcond': BOOTSTRAP_RCOND,
        'bootstrap_beta_max': BOOTSTRAP_BETA_MAX,
        'ir_beta_weight': IR_BETA_WEIGHT,
        'ir_half_length_m': IR_HALF_LENGTH,
        'orbit_rms_tolerance_m': ORBIT_TOL,
        'coupling_ip_weight': COUPLING_IP_WEIGHT,
        'coupling_svd_rcond': COUPLING_RCOND,
        'coupling_step_fraction': COUPLING_STEP,
        'coupling_correctable_orbit_tolerance_m': COUPLING_ORBIT_TOL,
        'correctable_orbit_rms_m': rms_arrays(correctable_orbit_errors(tw_after)).tolist(),
        'beta_rms_tolerance': BETA_RMS_TOL,
        'beta_max_tolerance': BETA_MAX_TOL,
        'tune_tolerance': TUNE_TOL,
    },
)
print('Saved', OUTPUT)
print('Run 003c_analyze_correction.py for plots and radiation-integral estimates')

if not reached_stop:
    raise RuntimeError(
        f'Ramp stopped at {ramp_fraction:g}; last accepted machine saved in {CHECKPOINT}'
    )

print(
    f'Finished; {stats["calls"]} Twiss calls, {stats["seconds"]:.1f} s in Twiss, {perf_counter() - START_TIME:.1f} s total'
)
print(
    f'  Full optics: {stats["calls"] - stats["orbit_calls"]} calls, {stats["seconds"] - stats["orbit_seconds"]:.1f} s; '
    f'orbit only: {stats["orbit_calls"]} calls, {stats["orbit_seconds"]:.1f} s'
)
