"""Apply the correction algorithm from 001 using Optimize response matrices.

Input: the line saved by 003a_install_correction_knobs.py. Settings are below.
Read the `Correction loop` cell for the workflow. Optimizer construction and
stopping criteria are explicit above it. Reference and response calculations
live in correction_setup.py; trial mechanics, observables and I/O live in
correction_steps.py, correction_observables.py and correction_io.py.
Run 003c_analyze_correction.py afterwards for plots and radiation integrals.

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

from response_builders import orbit_response
import correction_setup as setup

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
RESUME_MAX_INCREMENT = 0.025  # Cap the first increment reconstructed from checkpoint history.
CHECKPOINT = OUTPUT_DIR / 'checkpoint.json'
HISTORY = OUTPUT_DIR / 'history.csv'


print('Starting 4D correction; resume =', RESUME)
print('Ramping errors and nonlinear strengths with the same fraction')
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


tw_ref = setup.nominal_reference(line, full_twiss=full_twiss, log=print)
interaction_points = list(line.get_table().rows['ip.*'].name)
obs_names, alpha_names = setup.optics_observation_names(
    tw_ref, quad_names, interaction_points, phase_step=1 / 16,
)
obs_indices = tw_ref.rows.indices[obs_names]
alpha_indices = tw_ref.rows.indices[alpha_names]
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

set_values(orbit_knobs, np.zeros(len(orbit_knobs)))
line['on_errors'] = 0.0
for name in TUNE_KNOBS:
    line[name] = original[name]
# %% Explicit optimizers and fixed nominal responses
print(f'Using {len(quad_knobs)} installed quadrupole trims', flush=True)
print('Preparing nominal optics SVD', flush=True)
response_started = perf_counter()
nominal_response, strength_scale, optics_weights, optics_labels = setup.prepare_optics_response(
    line, tw_ref, quad_names, obs_names, alpha_names, dispersion_names,
    interaction_points=interaction_points, dispersion_ips=ip_names,
    alpha_weight=ALPHA_WEIGHT,
    ir_beta_weight=IR_BETA_WEIGHT, ir_half_length=IR_HALF_LENGTH,
    dispersion_weight=HORIZONTAL_DISPERSION_WEIGHT,
    dispersion_ip_weight=HORIZONTAL_DISPERSION_IP_WEIGHT,
    knobs=quad_knobs, full_twiss=full_twiss, log=print,
)
# All corrections use the same OptimizeLine API. Labels make response rows
# inspectable; weights and knob scales are ordinary Target/Vary settings.
optics_action = xt.Action(lambda: optics_vector(full_twiss()))
optics_optimizer = line.match(
    solve=False,
    name='normal optics and horizontal dispersion',
    vary=[
        xt.Vary(knob, step=1e-7, weight=scale) for knob, scale in zip(quad_knobs, strength_scale)
    ],
    targets=[
        optics_action.target(i, value=0.0, tol=1e-15, weight=weight, tag=label)
        for i, (label, weight) in enumerate(zip(optics_labels, optics_weights))
    ],
    assert_within_tol=False,
    show_call_counter=False,
)
optics_optimizer.set_response(nominal_response)
opt_s = optics_optimizer.response.svd.s
print('Optics SVD range', opt_s[[0, -1]], 'retained', np.sum(opt_s > OPTICS_RCOND * opt_s[0]))
del nominal_response
print('Preparing nominal 4D orbit SVDs', flush=True)
orbit_action_x = xt.Action(lambda: full_twiss()['x'][monitor_indices] - monitor_alignment()[0])
orbit_optimizer_x = line.match(
    solve=False,
    name='orbit x',
    vary=[xt.Vary(knob, step=1e-7) for knob in circuits['orbit']['x']['knobs']],
    targets=[
        orbit_action_x.target(i, value=0.0, tol=1e-15, tag=f'x@{name}')
        for i, name in enumerate(monitor_names)
    ],
    assert_within_tol=False,
    show_call_counter=False,
)
orbit_optimizer_x.set_response(orbit_response(tw_ref, monitor_names, corrector_names[0], 'x'))
orbit_action_y = xt.Action(lambda: full_twiss()['y'][monitor_indices] - monitor_alignment()[1])
orbit_optimizer_y = line.match(
    solve=False,
    name='orbit y',
    vary=[xt.Vary(knob, step=1e-7) for knob in circuits['orbit']['y']['knobs']],
    targets=[
        orbit_action_y.target(i, value=0.0, tol=1e-15, tag=f'y@{name}')
        for i, name in enumerate(monitor_names)
    ],
    assert_within_tol=False,
    show_call_counter=False,
)
orbit_optimizer_y.set_response(orbit_response(tw_ref, monitor_names, corrector_names[1], 'y'))
orbit_optimizers = [orbit_optimizer_x, orbit_optimizer_y]
for optimizer in orbit_optimizers:
    singular_values = optimizer.response.svd.s
    print(
        'Orbit SVD ready',
        optimizer.name,
        'shape',
        optimizer.response.matrix.shape,
        'retained',
        np.sum(singular_values > ORBIT_RCOND * singular_values[0]),
    )
coupling_names = dispersion_names
coupling_indices = dispersion_indices
coupling_response, skew_scale, coupling_weights, coupling_labels = setup.prepare_coupling_response(
    line, tw_ref, skew_groups, coupling_names, ip_names=ip_names,
    coupling_ip_weight=COUPLING_IP_WEIGHT,
    dispersion_weight=DISPERSION_WEIGHT, dispersion_ip_weight=DISPERSION_IP_WEIGHT,
    knobs=skew_knobs, full_twiss=full_twiss, log=print,
)
coupling_action = xt.Action(lambda: coupling_vector(full_twiss()))
coupling_optimizer = line.match(
    solve=False,
    name='coupling and vertical dispersion',
    vary=[xt.Vary(knob, step=1e-7, weight=scale) for knob, scale in zip(skew_knobs, skew_scale)],
    targets=[
        coupling_action.target(i, value=0.0, tol=1e-15, weight=weight, tag=label)
        for i, (label, weight) in enumerate(zip(coupling_labels, coupling_weights))
    ],
    assert_within_tol=False,
    show_call_counter=False,
)
coupling_optimizer.set_response(coupling_response)
coupling_s = coupling_optimizer.response.svd.s
print(
    'Coupling SVD retained',
    np.sum(coupling_s > COUPLING_RCOND * coupling_s[0]),
    'modes',
)
del coupling_response


print('INTERLEAVED SETUP READY', flush=True)
print(
    f'Response construction and SVDs: {perf_counter() - response_started:.1f} s',
    flush=True,
)


print('Preparing aligned 2x2 tune response (two Twiss probes)')
tune_response = setup.build_tune_response(
    tw_ref, line=line, tune_knobs=TUNE_KNOBS, full_twiss=full_twiss
)


def tune_residuals():
    tw = full_twiss()
    return np.array([tw.qx - tw_ref.qx, tw.qy - tw_ref.qy])


tune_action = xt.Action(tune_residuals)
tune_optimizer = line.match(
    solve=False,
    name='tunes',
    vary=[xt.Vary(knob, step=1e-7) for knob in TUNE_KNOBS],
    targets=[
        tune_action.target(i, value=0.0, tol=TUNE_TOL / 2, tag=q)
        for i, q in enumerate(('qx', 'qy'))
    ],
    assert_within_tol=False,
    show_call_counter=False,
)
tune_optimizer.set_response(tune_response)


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
    dx_rms, dx_ip = (
        (HORIZONTAL_DISPERSION_RMS_TOL, HORIZONTAL_DISPERSION_IP_TOL)
        if FINAL_REFINEMENT
        else (RAMP_HORIZONTAL_DISPERSION_RMS_TOL, RAMP_HORIZONTAL_DISPERSION_IP_TOL)
    )
    return (
        np.all(m['beta_rms'] < BETA_RMS_TOL)
        and max(m['beta_max']) < BETA_MAX_TOL
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
    keep = opt_s > OPTICS_RCOND * opt_s[0]
    print('  optics SVD cutoff', OPTICS_RCOND, 'modes', np.sum(keep), flush=True)
    return optics_optimizer.get_correction(optics_vector(tw), rcond=OPTICS_RCOND)


def steering_direction(tw, fraction):
    return np.concatenate(
        [
            opt.get_correction(error, rcond=ORBIT_RCOND)
            for error, opt in zip(lab_orbit_errors(tw), orbit_optimizers)
        ]
    )


# Trial steps accept/reject their own candidates; the loops below own the ramp.
correctable_orbit_errors = partial(
    obs.correctable_orbit_errors,
    lab_orbit_errors=lab_orbit_errors,
    orbit_optimizers=orbit_optimizers,
    rcond=ORBIT_RCOND,
)
tune_cleanup = partial(
    steps.tune_cleanup,
    line=line,
    tune_knobs=TUNE_KNOBS,
    tw_ref=tw_ref,
    tune_tol=TUNE_TOL,
    tune_optimizer=tune_optimizer,
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
    coupling_optimizer=coupling_optimizer,
    coupling_vector=coupling_vector,
    coupling_rcond=COUPLING_RCOND,
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
    ramp_fraction, ramp_history, tune_response = load_checkpoint(line, CHECKPOINT)
    tune_optimizer.set_response(tune_response)
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
        [ramp_fraction, ramp_fraction],
        rtol=0,
        atol=1e-12,
        err_msg='Checkpoint knobs disagree with the equal-fraction ramp',
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
    increment = min(increment, RESUME_MAX_INCREMENT)
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
        if refine:
            fraction = ramp_fraction
        else:
            fraction = min(stop_fraction, ramp_fraction + increment)
        baseline = values(control_names)
        stage_started = perf_counter()
        calls_before, seconds_before = stats['calls'], stats['seconds']
        print('TRY', fraction, 'increment', increment, flush=True)
        try:
            previous = float(line['on_errors'])
            # Extrapolate the accepted correction to the proposed error strength.
            if previous > 0 and fraction != previous:
                ratio = fraction / previous
                set_values(orbit_knobs, ratio * values(orbit_knobs))
                set_values(quad_knobs, ratio * values(quad_knobs))
                set_values(skew_knobs, ratio * values(skew_knobs))
                families = np.array([original[name] for name in TUNE_KNOBS])
                set_values(TUNE_KNOBS, families + ratio * (values(TUNE_KNOBS) - families))
            line['on_errors'] = line['on_nonlinear'] = fraction
            # Steer before tune cleanup: a large uncorrected orbit can shift the tunes.
            # Use the last accepted orbit directly as the closed-orbit search guess.
            tw = full_twiss(co_guess=orbit_guess(ramp_tw))
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
                    fraction,
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
            line['on_errors'] = line['on_nonlinear'] = ramp_fraction
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
            tw=ramp_tw,
            progress=ramp_fraction,
            final_refinement=FINAL_REFINEMENT,
            tune_response=tune_response,
            history=ramp_history,
        )
        print(f'  Checkpoint and history save: {perf_counter() - checkpoint_started:.3f} s')
        if cycles <= 3:
            increment = min(MAX_INCREMENT, increment * 1.25)

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
        'initial_increment': INITIAL_INCREMENT,
        'max_increment': MAX_INCREMENT,
        'resume_max_increment': RESUME_MAX_INCREMENT,
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
