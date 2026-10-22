"""Apply the correction algorithm from 001 using Optimize response matrices.

Input: the line saved by 003a_install_correction_knobs.py. Settings are below.
The original correction helpers retain their ordering, trial evaluation,
acceptance, rollback and stopping rules. get_correction(residuals=...) reuses
an existing Twiss result; run_jacobian is deliberately not used to introduce
another line search. The tune response stays fixed at its nominal value.
"""

from pathlib import Path
import json
from time import perf_counter
import builtins

import numpy as np
import xtrack as xt
import matplotlib.pyplot as plt

from response_builders import (
    normal_quadrupole_response,
    orbit_response,
    skew_quadrupole_response,
)

START_TIME = perf_counter()


def print(*args, **kwargs):
    """Timestamp this script's progress messages with cumulative wall time."""
    kwargs.setdefault('flush', True)
    builtins.print(f'[+{perf_counter() - START_TIME:8.1f} s]', *args, **kwargs)


HERE = Path(__file__).resolve().parent
INPUT = HERE / 'fccee_z_lcc_correction_knobs.json'
OUTPUT_DIR = HERE / 'response_api'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTPUT_DIR / 'machine.json'
PLOT = OUTPUT_DIR / 'correction.png'
SHOW_PLOTS = False
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
HISTORY_COLUMNS = 'progress,orbit_rms_x_m,orbit_rms_y_m,beta_rms_x,beta_rms_y,beta_max_x,beta_max_y,qx,qy,delta0,error_fraction,nonlinear_fraction,c_minus,f1001_rms,f1010_rms,dy_rms_m,dy_ip_max_m,dx_rms_m,dx_ip_max_m'


def ramp_settings(progress):
    """Return error and nonlinear fractions on the combined continuation path."""
    if not 0 <= progress <= 1:
        raise ValueError('Ramp progress must be in [0, 1]')
    return progress, max(
        0.0, (progress - BOOTSTRAP_FRACTION) / (1 - BOOTSTRAP_FRACTION)
    )


print('Starting 4D correction; resume =', RESUME)
print('Combined ramp: nonlinear magnets turn on at error fraction', BOOTSTRAP_FRACTION)
print(
    'Ramp targets: c_minus <',
    RAMP_C_MINUS_TOL,
    'RDT RMS <',
    RAMP_COUPLING_RMS_TOL,
    'Dy RMS/IP max [mm] <',
    1e3 * RAMP_DISPERSION_RMS_TOL,
    1e3 * RAMP_DISPERSION_IP_TOL,
)
print(
    'Final targets: c_minus <',
    C_MINUS_TOL,
    'RDT RMS <',
    COUPLING_RMS_TOL,
    'Dy RMS/IP max [mm] <',
    1e3 * DISPERSION_RMS_TOL,
    1e3 * DISPERSION_IP_TOL,
)
print(
    'Dx error RMS/IP max targets [mm]: ramp',
    1e3 * RAMP_HORIZONTAL_DISPERSION_RMS_TOL,
    1e3 * RAMP_HORIZONTAL_DISPERSION_IP_TOL,
    'final',
    1e3 * HORIZONTAL_DISPERSION_RMS_TOL,
    1e3 * HORIZONTAL_DISPERSION_IP_TOL,
)
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
nonlinear_magnets = circuits['nonlinear_magnets']
added_correctors = circuits['added_correctors']

R_STEPS = dict(dx=1e-8, dpx=1e-9, dy=1e-8, dpy=1e-9, dzeta=1e-7, ddelta=1e-8)
TWISS_CALLS = 0
TWISS_SECONDS = 0.0
ORBIT_ONLY_CALLS = 0
ORBIT_ONLY_SECONDS = 0.0


def orbit_guess(tw):
    return dict(
        x=float(tw.x[0]),
        px=float(tw.px[0]),
        y=float(tw.y[0]),
        py=float(tw.py[0]),
        zeta=0.0,
        delta=0.0,
    )


def compute_twiss(
    seed=None, co_guess=None, radiation_integrals=False, only_orbit=False
):
    global TWISS_CALLS, TWISS_SECONDS, ORBIT_ONLY_CALLS, ORBIT_ONLY_SECONDS
    kwargs = {} if seed is None else dict(co_guess=orbit_guess(seed))
    if co_guess is not None:
        kwargs['co_guess'] = co_guess
    started = perf_counter()
    TWISS_CALLS += 1
    if only_orbit:
        ORBIT_ONLY_CALLS += 1
    try:
        return line.twiss(
            chrom=False,
            steps_R_matrix=R_STEPS.copy(),
            radiation_integrals=radiation_integrals,
            only_orbit=only_orbit,
            **kwargs,
        )
    finally:
        elapsed = perf_counter() - started
        TWISS_SECONDS += elapsed
        if only_orbit:
            ORBIT_ONLY_SECONDS += elapsed


def full_twiss(seed=None, co_guess=None, radiation_integrals=False):
    return compute_twiss(seed, co_guess, radiation_integrals)


def orbit_twiss(seed=None, co_guess=None):
    return compute_twiss(seed, co_guess, only_orbit=True)


def ensure_full_twiss(tw):
    # Call only while the line still has the settings used to compute tw.
    # Use the tracked orbit as a guess; keep the ordinary closed-orbit search.
    if 'betx' in tw._col_names:
        return tw
    return full_twiss(tw)


def values(names):
    return np.array([line[name] for name in names])


def set_values(names, vals):
    env.vars.update({name: float(value) for name, value in zip(names, vals)})


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
print(
    'Aligned nonlinear magnets on/off: tunes',
    tw_nonlinear_on.qx,
    tw_nonlinear_on.qy,
    tw_ref.qx,
    tw_ref.qy,
    'max relative beta change',
    nominal_beta_change,
    flush=True,
)
if nominal_beta_change > 1e-4:
    raise RuntimeError(
        'Aligned optics depend on nonlinear setting; rebuild responses per stage'
    )

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
alpha_indices = tw_ref.rows.indices[
    list(dict.fromkeys(quad_names + list(tt.rows['ip.*'].name)))
]
monitor_indices = tw_ref.rows.indices[monitor_names]
ip_names = ['ipa', 'ipd', 'ipg', 'ipj']
ip_indices = tw_ref.rows.indices[ip_names]
dispersion_names = list(dict.fromkeys(monitor_names + ip_names))
dispersion_indices = tw_ref.rows.indices[dispersion_names]


def beta_errors(tw):
    return [tw['bet' + p] / tw_ref['bet' + p] - 1 for p in 'xy']


def horizontal_dispersion_errors(tw):
    # Ideal model observable, not a noisy RF scan. Preserve nominal Dx.
    return (tw.dx - tw_ref.dx)[dispersion_indices]


def horizontal_dispersion_rms(tw):
    return float(np.sqrt(np.mean(horizontal_dispersion_errors(tw) ** 2)))


def horizontal_dispersion_ip_max(tw):
    return float(np.max(np.abs((tw.dx - tw_ref.dx)[ip_indices])))


def report(label, tw, fraction=1.0):
    beta = beta_errors(tw)
    rms = [np.sqrt(np.mean(e**2)) for e in beta]
    maximum = [np.max(np.abs(e)) for e in beta]
    orb = rms_arrays(orbit_errors(tw, fraction)) * 1e6
    print(
        label,
        'beta rms',
        rms,
        'beta max',
        maximum,
        'tunes',
        tw.qx,
        tw.qy,
        'orbit_um',
        orb,
        'Dx error RMS/IP max [mm]',
        1e3 * horizontal_dispersion_rms(tw),
        1e3 * horizontal_dispersion_ip_max(tw),
        flush=True,
    )


def optics_vector(tw):
    return np.r_[
        *[
            np.log(tw['bet' + p][obs_indices] / tw_ref['bet' + p][obs_indices])
            for p in 'xy'
        ],
        *[
            (
                tw['alf' + p][alpha_indices]
                * tw_ref['bet' + p][alpha_indices]
                / tw['bet' + p][alpha_indices]
                - tw_ref['alf' + p][alpha_indices]
            )
            for p in 'xy'
        ],
        horizontal_dispersion_errors(tw),
        tw.qx - tw_ref.qx,
        tw.qy - tw_ref.qy,
    ]


set_values(orbit_knobs, np.zeros(len(orbit_knobs)))
line['on_errors'] = 0.0
for name in TUNE_KNOBS:
    line[name] = original[name]
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
nominal_dx_response = nominal_response[-len(dispersion_names) - 2 : -2]
dx_weights = HORIZONTAL_DISPERSION_WEIGHT * np.array(
    [
        HORIZONTAL_DISPERSION_IP_WEIGHT if n in ip_names else 1.0
        for n in dispersion_names
    ]
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
near_ip = (
    np.min(np.minimum(ip_distance, tw_ref.line_length - ip_distance), axis=1)
    < IR_HALF_LENGTH
)
for k in range(2):
    optics_weights[k * len(obs_indices) : (k + 1) * len(obs_indices)] *= np.where(
        near_ip, IR_BETA_WEIGHT, 1.0
    )
print(
    f'Upweighting beta at {np.sum(near_ip)} interaction-region observations', flush=True
)
# A central distributed probe checks the sign, integration and scaling in two
# Twiss calls, without probing every quadrupole. The central difference cancels
# quadratic tune shifts, which can dominate the small net first-order shift.
probe = 1e-4 * strength_scale * np.sin(np.arange(len(quad_knobs)) + 0.3)
set_values(quad_knobs, probe)
try:
    probe_plus = full_twiss(tw_ref)
    set_values(quad_knobs, -probe)
    probe_minus = full_twiss(tw_ref)
    measured = (optics_vector(probe_plus) - optics_vector(probe_minus)) / 2
    prediction = nominal_response @ probe
    discrepancy = np.linalg.norm(
        optics_weights * (measured - prediction)
    ) / np.linalg.norm(optics_weights * prediction)
    dx_prediction = nominal_dx_response @ probe
    dx_measured = (
        horizontal_dispersion_errors(probe_plus)
        - horizontal_dispersion_errors(probe_minus)
    ) / 2
    dx_discrepancy = np.linalg.norm(dx_measured - dx_prediction) / np.linalg.norm(
        dx_prediction
    )
    print(
        'Normal optics/Dx response directional check, relative errors:',
        discrepancy,
        dx_discrepancy,
    )
    if max(discrepancy, dx_discrepancy) > 0.05:
        raise RuntimeError('Normal optics/dispersion response disagrees with tracking')
finally:
    set_values(quad_knobs, np.zeros(len(quad_knobs)))
# All corrections use the same OptimizeLine API. Labels make response rows
# inspectable; weights and knob scales are ordinary Target/Vary settings.
optics_labels = (
    [f'log_beta_{p}@{n}' for p in 'xy' for n in obs_names]
    + [f'normalized_alpha_{p}@{n}' for p in 'xy' for n in tw_ref.name[alpha_indices]]
    + [f'dx@{n}' for n in dispersion_names]
    + ['qx', 'qy']
)
optics_action = xt.Action(lambda: optics_vector(full_twiss()))
optics_optimizer = line.match(
    solve=False,
    name='normal optics and horizontal dispersion',
    vary=[
        xt.Vary(knob, step=1e-7, weight=scale)
        for knob, scale in zip(quad_knobs, strength_scale)
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
print(
    'Optics SVD range',
    opt_s[0],
    opt_s[-1],
    'retained',
    np.sum(opt_s > OPTICS_RCOND * opt_s[0]),
    flush=True,
)
del nominal_response, nominal_dx_response
print('Preparing nominal 4D orbit SVDs', flush=True)
orbit_action_x = xt.Action(
    lambda: full_twiss()['x'][monitor_indices] - monitor_alignment()[0]
)
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
orbit_optimizer_x.set_response(
    orbit_response(tw_ref, monitor_names, corrector_names[0], 'x')
)
orbit_action_y = xt.Action(
    lambda: full_twiss()['y'][monitor_indices] - monitor_alignment()[1]
)
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
orbit_optimizer_y.set_response(
    orbit_response(tw_ref, monitor_names, corrector_names[1], 'y')
)
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
coupling_response, skew_scale = skew_quadrupole_response(
    line, tw_ref, skew_groups, coupling_names
)
coupling_weights = np.tile(
    np.array([COUPLING_IP_WEIGHT if n in ip_names else 1.0 for n in coupling_names]), 4
)
coupling_weights = np.r_[
    coupling_weights,
    DISPERSION_WEIGHT
    * np.array(
        [DISPERSION_IP_WEIGHT if n in ip_names else 1.0 for n in coupling_names]
    ),
]
coupling_weights /= np.sqrt(len(coupling_names))


def coupling_vector(tw):
    return np.concatenate(
        [
            part
            for key in ('f1001', 'f1010')
            for part in (
                (tw[key] - tw_ref[key])[coupling_indices].real,
                (tw[key] - tw_ref[key])[coupling_indices].imag,
            )
        ]
        + [dispersion_errors(tw)]
    )


def dispersion_errors(tw):
    # Ideal model observable, like the coupling RDTs; not a noisy RF scan.
    return (tw.dy - tw_ref.dy)[coupling_indices]


def dispersion_rms(tw):
    return float(np.sqrt(np.mean(dispersion_errors(tw) ** 2)))


def dispersion_ip_max(tw):
    return float(np.max(np.abs((tw.dy - tw_ref.dy)[ip_indices])))


def coupling_rms(tw):
    return np.array(
        [
            np.sqrt(
                np.mean(
                    np.abs(tw[key][coupling_indices] - tw_ref[key][coupling_indices])
                    ** 2
                )
            )
            for key in ('f1001', 'f1010')
        ]
    )


def coupling_ip_max(tw):
    return max(
        np.max(np.abs(tw[key][ip_indices] - tw_ref[key][ip_indices]))
        for key in ('f1001', 'f1010')
    )


def coupling_goal(tw):
    if FINAL_REFINEMENT:
        rdt_rms, rdt_ip, c_minus = COUPLING_RMS_TOL, COUPLING_IP_TOL, C_MINUS_TOL
        dy_rms, dy_ip = DISPERSION_RMS_TOL, DISPERSION_IP_TOL
    else:
        rdt_rms, rdt_ip, c_minus = (
            RAMP_COUPLING_RMS_TOL,
            RAMP_COUPLING_IP_TOL,
            RAMP_C_MINUS_TOL,
        )
        dy_rms, dy_ip = RAMP_DISPERSION_RMS_TOL, RAMP_DISPERSION_IP_TOL
    return (
        np.all(coupling_rms(tw) < rdt_rms)
        and coupling_ip_max(tw) < rdt_ip
        and tw.c_minus < c_minus
        and dispersion_rms(tw) < dy_rms
        and dispersion_ip_max(tw) < dy_ip
    )


def coupling_cost(tw):
    return np.linalg.norm(coupling_weights * coupling_vector(tw))


# Check the analytical response against tracking for a distributed probe.
probe = 1e-5 * skew_scale * np.sin(np.arange(len(skew_knobs)) + 0.3)
set_values(skew_knobs, probe)
try:
    probe_tw = full_twiss(tw_ref)
    prediction = coupling_response @ probe
    discrepancy = np.linalg.norm(
        coupling_vector(probe_tw) - prediction
    ) / np.linalg.norm(prediction)
    print('Coupling response directional check, relative error:', discrepancy)
    n_rdt = 4 * len(coupling_names)
    dy_discrepancy = np.linalg.norm(
        coupling_vector(probe_tw)[n_rdt:] - prediction[n_rdt:]
    ) / np.linalg.norm(prediction[n_rdt:])
    print('Dispersion response directional check, relative error:', dy_discrepancy)
    if max(discrepancy, dy_discrepancy) > 0.05:
        raise RuntimeError('Coupling response disagrees with tracking')
finally:
    set_values(skew_knobs, np.zeros(len(skew_knobs)))
coupling_labels = [
    f'{part}_{rdt}@{n}'
    for rdt in ('f1001', 'f1010')
    for part in ('real', 'imag')
    for n in coupling_names
] + [f'dy@{n}' for n in coupling_names]
coupling_action = xt.Action(lambda: coupling_vector(full_twiss()))
coupling_optimizer = line.match(
    solve=False,
    name='coupling and vertical dispersion',
    vary=[
        xt.Vary(knob, step=1e-7, weight=scale)
        for knob, scale in zip(skew_knobs, skew_scale)
    ],
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


def orbit_errors(tw, fraction=None):
    dx, dy, roll = monitor_alignment()
    x, y = tw.x[monitor_indices] - dx, tw.y[monitor_indices] - dy
    return [x * np.cos(roll) + y * np.sin(roll), -x * np.sin(roll) + y * np.cos(roll)]


def lab_orbit_errors(tw):
    dx, dy, _ = monitor_alignment()
    return [tw.x[monitor_indices] - dx, tw.y[monitor_indices] - dy]


def correctable_orbit_errors(tw):
    # Discarded SVD modes and the rectangular response's left null space
    # cannot be removed by the available correctors in the nominal model.
    residuals = []
    for error, optimizer in zip(lab_orbit_errors(tw), orbit_optimizers):
        svd = optimizer.response.svd
        u = svd.U[:, svd.s > ORBIT_RCOND * svd.s[0]]
        residuals.append(u @ (u.T @ error))
    return residuals


def rms_arrays(arrays):
    return np.array([np.sqrt(np.mean(a * a)) for a in arrays])


def optical_goal(tw):
    beta = beta_errors(tw)
    # Establish the initial match near the aligned solution before allowing
    # larger local peaks during the rest of the continuation.
    peak_limit = (
        BOOTSTRAP_BETA_MAX if line['on_errors'] <= BOOTSTRAP_FRACTION else BETA_MAX_TOL
    )
    dx_rms, dx_ip = (
        (HORIZONTAL_DISPERSION_RMS_TOL, HORIZONTAL_DISPERSION_IP_TOL)
        if FINAL_REFINEMENT
        else (RAMP_HORIZONTAL_DISPERSION_RMS_TOL, RAMP_HORIZONTAL_DISPERSION_IP_TOL)
    )
    return (
        np.all(rms_arrays(beta) < BETA_RMS_TOL)
        and max(np.max(np.abs(e)) for e in beta) < peak_limit
        and horizontal_dispersion_rms(tw) < dx_rms
        and horizontal_dispersion_ip_max(tw) < dx_ip
        and max(abs(tw.qx - tw_ref.qx), abs(tw.qy - tw_ref.qy)) < TUNE_TOL
    )


def build_tune_response(tw):
    """Numerical two-family tune response, with all probe settings restored."""
    baseline = values(TUNE_KNOBS)
    jac = np.empty((2, 2))
    try:
        for j in range(2):
            line[TUNE_KNOBS[j]] = baseline[j] + 1e-7
            probe = full_twiss(tw)
            jac[:, j] = (
                np.array([probe.qx, probe.qy]) - np.array([tw.qx, tw.qy])
            ) / 1e-7
            set_values(TUNE_KNOBS, baseline)
    finally:
        set_values(TUNE_KNOBS, baseline)
    return jac


def tune_cleanup(tw):
    """Use the fixed nominal 2x2 response with the original trial acceptance."""
    tw = ensure_full_twiss(tw)
    target = np.array([tw_ref.qx, tw_ref.qy])
    for it in range(7):
        error = np.array([tw.qx, tw.qy]) - target
        if max(abs(error)) < TUNE_TOL / 2:
            return tw
        baseline = values(TUNE_KNOBS)
        change = tune_optimizer.get_correction(error, rcond=0.0)
        for bt in range(4):
            set_values(TUNE_KNOBS, baseline + change * 0.5**bt)
            try:
                probe = full_twiss(tw)
            except (ValueError, xt.twiss.ClosedOrbitSearchError):
                continue
            new_error = np.array([probe.qx, probe.qy]) - target
            if np.linalg.norm(new_error) < np.linalg.norm(error):
                tw = probe
                print('  tune cached step', 0.5**bt, 'max error', max(abs(new_error)))
                break
        else:
            set_values(TUNE_KNOBS, baseline)
            raise RuntimeError('Tune cleanup failed with the fixed nominal response')
    if max(abs(np.array([tw.qx, tw.qy]) - target)) < TUNE_TOL / 2:
        return tw
    raise RuntimeError('Tune cleanup iteration limit')


print('Preparing aligned 2x2 tune response (two Twiss probes)')
tune_response = build_tune_response(tw_ref)


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


def optical_cost(tw):
    # Tunes are checked separately after numerical arc-family cleanup.
    return np.linalg.norm((optics_weights * optics_vector(tw))[:-2])


def optical_direction(tw):
    cutoff = (
        BOOTSTRAP_RCOND if line['on_errors'] <= BOOTSTRAP_FRACTION else OPTICS_RCOND
    )
    keep = opt_s > cutoff * opt_s[0]
    print('  optics SVD cutoff', cutoff, 'modes', np.sum(keep), flush=True)
    return optics_optimizer.get_correction(optics_vector(tw), rcond=cutoff)


def steering_direction(tw, fraction):
    return np.concatenate(
        [
            opt.get_correction(error, rcond=ORBIT_RCOND)
            for error, opt in zip(lab_orbit_errors(tw), orbit_optimizers)
        ]
    )


def orbit_step(tw, fraction, max_backtracks=13, only_orbit=False):
    baseline = values(orbit_knobs)
    direction = steering_direction(tw, fraction)
    before = np.linalg.norm(np.concatenate(orbit_errors(tw, fraction)))
    for backtrack in range(max_backtracks):
        step = 0.5**backtrack
        set_values(orbit_knobs, baseline + step * direction)
        try:
            trial = orbit_twiss(tw) if only_orbit else full_twiss(tw)
        except (ValueError, xt.twiss.ClosedOrbitSearchError):
            continue
        if np.linalg.norm(np.concatenate(orbit_errors(trial, fraction))) < before * (
            1 - 1e-6
        ):
            print(
                '  orbit step',
                step,
                'rms_um',
                rms_arrays(orbit_errors(trial, fraction)) * 1e6,
                flush=True,
            )
            return trial
    set_values(orbit_knobs, baseline)
    raise RuntimeError('No stable improving orbit step')


def optics_step(tw):
    """Evaluate each quadrupole trial together with its orbit/tune correction."""
    baseline = values(quad_knobs)
    families = values(TUNE_KNOBS)
    steering = values(orbit_knobs)
    fraction = float(line['on_errors'])
    direction = optical_direction(tw)
    before = optical_cost(tw)
    largest = np.max(np.abs(direction / strength_scale))
    initial_step = min(1.0, 0.5 / max(largest, 1e-30))
    for backtrack in range(10):
        step = initial_step * 0.5**backtrack
        set_values(quad_knobs, baseline + step * direction)
        set_values(TUNE_KNOBS, families)
        set_values(orbit_knobs, steering)
        try:
            trial = orbit_twiss(tw)
            try:
                trial = orbit_step(trial, fraction)
            except RuntimeError:
                pass
            trial = tune_cleanup(trial)
        except (ValueError, RuntimeError):
            continue
        if optical_cost(trial) < before * (1 - 1e-6):
            print(
                '  joint step',
                step,
                'cost',
                optical_cost(trial),
                'beta rms',
                rms_arrays(beta_errors(trial)),
                'beta max',
                [np.max(abs(e)) for e in beta_errors(trial)],
                'Dx error RMS/IP max [mm]',
                1e3 * horizontal_dispersion_rms(trial),
                1e3 * horizontal_dispersion_ip_max(trial),
                flush=True,
            )
            return trial
    set_values(quad_knobs, baseline)
    set_values(TUNE_KNOBS, families)
    set_values(orbit_knobs, steering)
    raise RuntimeError('No stable improving joint step')


def settle_coupling_orbit(tw):
    # Small remaining orbit changes feed down in the sextupoles and can obscure
    # the improvement of a skew trial. Compare baseline and trials after the
    # same settling procedure; exclude the nominal response's discarded modes.
    for iteration in range(3):
        if np.max(rms_arrays(correctable_orbit_errors(tw))) < COUPLING_ORBIT_TOL:
            break
        try:
            tw = orbit_step(
                tw, float(line['on_errors']), max_backtracks=4, only_orbit=iteration < 2
            )
        except RuntimeError:
            break
    return tune_cleanup(tw)


def coupling_step(tw):
    tw = settle_coupling_orbit(tw)
    if coupling_goal(tw):
        return tw
    baseline = values(skew_knobs)
    steering, families = values(orbit_knobs), values(TUNE_KNOBS)
    direction = coupling_optimizer.get_correction(
        coupling_vector(tw), rcond=COUPLING_RCOND
    )
    before = coupling_cost(tw)
    for backtrack in range(10):
        step = COUPLING_STEP * 0.5**backtrack
        set_values(skew_knobs, baseline + step * direction)
        set_values(orbit_knobs, steering)
        set_values(TUNE_KNOBS, families)
        try:
            trial = orbit_twiss(tw)
            # Re-steer after the fractional skew update before comparing RDTs.
            trial = settle_coupling_orbit(trial)
        except (ValueError, RuntimeError, xt.twiss.ClosedOrbitSearchError):
            continue
        if coupling_cost(trial) < before * (1 - 1e-6):
            print(
                '  coupling step',
                step,
                'RDT RMS',
                coupling_rms(trial),
                'c_minus',
                trial.c_minus,
                'IP max',
                coupling_ip_max(trial),
                'Dy RMS/IP max [mm]',
                1e3 * dispersion_rms(trial),
                1e3 * dispersion_ip_max(trial),
            )
            return trial
    set_values(skew_knobs, baseline)
    set_values(orbit_knobs, steering)
    set_values(TUNE_KNOBS, families)
    raise RuntimeError('No stable improving coupling step')


def interleaved_stage(progress, seed):
    fraction, nonlinear = ramp_settings(progress)
    previous = float(line['on_errors'])
    guess = orbit_guess(seed)
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
    for cycle in range(MAX_CYCLES):
        orbit_ok = np.all(rms_arrays(orbit_errors(tw)) <= orbit_tolerance)
        optics_ok = optical_goal(tw)
        coupling_ok = coupling_goal(tw)
        if orbit_ok and optics_ok and coupling_ok:
            return tw, cycle
        print(
            ' cycle',
            cycle + 1,
            'errors',
            fraction,
            'nonlinear',
            nonlinear,
            'orbit_ok',
            orbit_ok,
            'optics_ok',
            optics_ok,
            'coupling_ok',
            coupling_ok,
            flush=True,
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
    raise RuntimeError('Interleaved stage iteration limit')


def save_variable_checkpoint():
    # The base lattice and the expression setup above are unchanged by the ramp.
    # Preserve the expressions loaded from the input JSON. In particular,
    # re-parsing a BPM expression string can freeze its element-attribute link.
    independent = {
        name: value
        for name, value in line.vars.get_table().to_dict().items()
        if not isinstance(value, str)
    }
    checkpoint = {
        'format': 'fccee_tuning_correction_v7',
        'ramp_schedule': 'combined_after_bootstrap',
        'bootstrap_fraction': BOOTSTRAP_FRACTION,
        'error_seed': line.metadata['tuning_errors']['seed'],
        'base_lattice': INPUT.name,
        'variables': independent,
        'co_guess': orbit_guess(ramp_tw),
        'progress': ramp_fraction,
        'final_refinement': FINAL_REFINEMENT,
        'tune_response': tune_response.tolist(),
        'history': ramp_history,
    }
    temporary = CHECKPOINT.with_suffix('.tmp.json')
    xt.json.dump(checkpoint, temporary, indent=None)
    temporary.replace(CHECKPOINT)
    np.savetxt(
        HISTORY,
        np.array(ramp_history, dtype=float),
        delimiter=',',
        header=HISTORY_COLUMNS,
        comments='',
    )


def run_interleaved(stop_fraction=1.0):
    """Advance only accepted stages; retain a reloadable atomic checkpoint."""
    global ramp_tw, ramp_fraction, ramp_history
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
        orbit_knobs
        + quad_knobs
        + skew_knobs
        + TUNE_KNOBS
        + ['on_errors', 'on_nonlinear']
    )
    # Refit an accepted checkpoint if targets changed (e.g. adding Dy), even
    # if it already reached the requested ramp endpoint. Never silently reuse
    # an endpoint that fails the current goals.
    refine = ramp_fraction > 0 and not (
        optical_goal(ramp_tw)
        and coupling_goal(ramp_tw)
        and np.all(
            rms_arrays(orbit_errors(ramp_tw))
            <= max(0.002e-6, min(ramp_fraction, 1.0) * ORBIT_TOL)
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
        calls_before, seconds_before = TWISS_CALLS, TWISS_SECONDS
        print('TRY', fraction, 'increment', increment, flush=True)
        try:
            candidate, cycles = interleaved_stage(fraction, ramp_tw)
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
                flush=True,
            )
            if refine or increment < MIN_INCREMENT:
                return False
            continue
        finally:
            print(
                f'  Stage timing: {perf_counter() - stage_started:.1f} s total; '
                f'{TWISS_SECONDS - seconds_before:.1f} s in '
                f'{TWISS_CALLS - calls_before} Twiss calls',
                flush=True,
            )
        ramp_tw, ramp_fraction = candidate, fraction
        beta = beta_errors(ramp_tw)
        row = [
            fraction,
            *rms_arrays(orbit_errors(ramp_tw, min(fraction, 1.0))),
            *rms_arrays(beta),
            *[np.max(np.abs(e)) for e in beta],
            ramp_tw.qx,
            ramp_tw.qy,
            ramp_tw.delta[0],
            float(line['on_errors']),
            float(line['on_nonlinear']),
            ramp_tw.c_minus,
            *coupling_rms(ramp_tw),
            dispersion_rms(ramp_tw),
            dispersion_ip_max(ramp_tw),
            horizontal_dispersion_rms(ramp_tw),
            horizontal_dispersion_ip_max(ramp_tw),
        ]
        if refine:
            ramp_history[-1] = row
        else:
            ramp_history.append(row)
        refine = False
        print(
            'ACCEPT progress',
            fraction,
            'errors',
            line['on_errors'],
            'nonlinear',
            line['on_nonlinear'],
            'orbit_um',
            np.array(row[1:3]) * 1e6,
            'beta_rms',
            row[3:5],
            'beta_max',
            row[5:7],
            'tunes',
            row[7:9],
            'c_minus',
            ramp_tw.c_minus,
            'RDT RMS',
            coupling_rms(ramp_tw),
            'Dy RMS/IP max [mm]',
            1e3 * dispersion_rms(ramp_tw),
            1e3 * dispersion_ip_max(ramp_tw),
            flush=True,
        )
        print(
            '  Dx error RMS/IP max [mm]',
            1e3 * horizontal_dispersion_rms(ramp_tw),
            1e3 * horizontal_dispersion_ip_max(ramp_tw),
        )
        checkpoint_started = perf_counter()
        line.twiss_default['co_guess'] = orbit_guess(ramp_tw)
        save_variable_checkpoint()
        if fraction == BOOTSTRAP_FRACTION:
            line.to_json(OUTPUT_DIR / 'bootstrap.json', indent=None)
            print('  Saved full lattice at the end of the errors-only bootstrap')
        print(
            f'  Checkpoint and history save: {perf_counter() - checkpoint_started:.3f} s'
        )
        if fraction == BOOTSTRAP_FRACTION:
            increment = COMBINED_INITIAL_INCREMENT
        elif cycles <= (3 if fraction > BOOTSTRAP_FRACTION else 5):
            limit = (
                COMBINED_MAX_INCREMENT
                if fraction > BOOTSTRAP_FRACTION
                else MAX_INCREMENT
            )
            increment = min(limit, increment * 1.25)
    return True


set_values(orbit_knobs, np.zeros(len(orbit_knobs)))
set_values(quad_knobs, np.zeros(len(quad_knobs)))
line['on_errors'] = 0
for name in TUNE_KNOBS:
    line[name] = original[name]
ramp_tw = full_twiss(tw_ref)
ramp_fraction = 0.0
ramp_history = []
if RESUME:
    checkpoint = json.loads(CHECKPOINT.read_text())
    if (
        checkpoint['format'] != 'fccee_tuning_correction_v7'
        or checkpoint['ramp_schedule'] != 'combined_after_bootstrap'
        or checkpoint['bootstrap_fraction'] != BOOTSTRAP_FRACTION
    ):
        raise ValueError(
            'Checkpoint uses a different ramp schedule; restart without RESUME=True'
        )
    if checkpoint['error_seed'] != line.metadata['tuning_errors']['seed']:
        raise ValueError('Checkpoint belongs to a different error seed')
    # Keep the original deferred expressions loaded from the input model.
    line.vars.update(
        {
            name: value
            for name, value in checkpoint['variables'].items()
            if not isinstance(value, str)
        }
    )
    line.twiss_default['co_guess'] = checkpoint['co_guess']
    tune_response = np.array(checkpoint['tune_response'])
    tune_optimizer.set_response(tune_response)
    ramp_fraction = float(checkpoint['progress'])
    ramp_history = [
        row + [None] * (len(HISTORY_COLUMNS.split(',')) - len(row))
        for row in checkpoint['history']
    ]
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
    report('RESUMED', ramp_tw)
    np.testing.assert_allclose(
        [line['on_errors'], line['on_nonlinear']],
        ramp_settings(ramp_fraction),
        rtol=0,
        atol=1e-12,
        err_msg='Checkpoint knobs disagree with the combined ramp',
    )
    if STOP_AT < ramp_fraction:
        raise ValueError('STOP_AT precedes the checkpoint')
reached_stop = run_interleaved(stop_fraction=STOP_AT)
if reached_stop and abs(ramp_fraction - 1.0) < 1e-12:
    FINAL_REFINEMENT = True
    print('FINAL REFINEMENT at full errors and nonlinear strengths')
    reached_stop = run_interleaved(stop_fraction=1.0)
ramp_completed = reached_stop and abs(ramp_fraction - 1.0) < 1e-12

# Final verification and independently reloadable output.
tw_after = full_twiss(ramp_tw, radiation_integrals=ramp_completed)
if ramp_completed:
    assert np.all(rms_arrays(orbit_errors(tw_after)) <= ORBIT_TOL)
    assert optical_goal(tw_after) and coupling_goal(tw_after)
line.twiss_default['co_guess'] = orbit_guess(tw_after)
line.twiss_default['steps_R_matrix'] = R_STEPS.copy()
print(
    'Ramp completed:',
    ramp_completed,
    'errors:',
    line['on_errors'],
    'nonlinear:',
    line['on_nonlinear'],
)
report('AFTER', tw_after, min(ramp_fraction, 1.0))
if not ramp_completed:
    OUTPUT = OUTPUT_DIR / 'partial.json'
line.metadata['orbit_optics_correction'] = {
    'error_fraction': float(line['on_errors']),
    'nonlinear_fraction': float(line['on_nonlinear']),
    'method': '4d',
    'delta0': 0.0,
    'ramp_schedule': 'combined_after_bootstrap',
    'progress': ramp_fraction,
    'nonlinear_ramp_knob': 'on_nonlinear',
    'nonlinear_ramp_min_order': 2,
    'nonlinear_ramp_magnets': nonlinear_magnets,
    'nonlinear_initial_increment': NONLINEAR_INITIAL_INCREMENT,
    'nonlinear_max_increment': NONLINEAR_MAX_INCREMENT,
    'completed': bool(ramp_completed),
    'target_tunes': [float(tw_ref.qx), float(tw_ref.qy)],
    'nominal_tune_knobs': original,
    'nominal_reference': 'input model with on_errors=0',
    'orbit_response': 'aligned 4D optics at delta=0',
    'optics_response': 'aligned thick-quadrupole beta/alpha/Dx/tune response',
    'tune_response': 'fixed aligned 2x2 response',
    'orbit_trials': 'only_orbit for intermediate steering; ordinary full Twiss before optics/coupling checks',
    'fitted_optics': [
        'betx',
        'bety',
        'alfx',
        'alfy',
        'dx',
        'qx',
        'qy',
        'f1001',
        'f1010',
        'dy',
    ],
    'horizontal_dispersion_rms_tolerance_m': HORIZONTAL_DISPERSION_RMS_TOL,
    'horizontal_dispersion_ip_tolerance_m': HORIZONTAL_DISPERSION_IP_TOL,
    'horizontal_dispersion_rms_m': horizontal_dispersion_rms(tw_after),
    'horizontal_dispersion_ip_max_m': horizontal_dispersion_ip_max(tw_after),
    'horizontal_dispersion_weight_per_m': HORIZONTAL_DISPERSION_WEIGHT,
    'horizontal_dispersion_ip_weight': HORIZONTAL_DISPERSION_IP_WEIGHT,
    'dispersion_rms_tolerance_m': DISPERSION_RMS_TOL,
    'dispersion_ip_tolerance_m': DISPERSION_IP_TOL,
    'dispersion_rms_m': dispersion_rms(tw_after),
    'dispersion_ip_max_m': dispersion_ip_max(tw_after),
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
    'coupling_rms': coupling_rms(tw_after).tolist(),
    'skew_trim_unit': 'm^-1',
    'optics_svd_rcond': OPTICS_RCOND,
    'optics_svd_modes': int(np.sum(opt_s > OPTICS_RCOND * opt_s[0])),
    'bootstrap_fraction': BOOTSTRAP_FRACTION,
    'bootstrap_svd_rcond': BOOTSTRAP_RCOND,
    'bootstrap_beta_max': BOOTSTRAP_BETA_MAX,
    'ir_beta_weight': IR_BETA_WEIGHT,
    'ir_half_length_m': IR_HALF_LENGTH,
    'quadrupole_trim_prefix': 'dk1_optics_',
    'quadrupole_trim_unit': 'm^-2',
    'orbit_rms_tolerance_m': ORBIT_TOL,
    'added_correctors': added_correctors,
    'coupling_ip_weight': COUPLING_IP_WEIGHT,
    'coupling_svd_rcond': COUPLING_RCOND,
    'coupling_step_fraction': COUPLING_STEP,
    'coupling_correctable_orbit_tolerance_m': COUPLING_ORBIT_TOL,
    'coupling_ip_max': float(coupling_ip_max(tw_after)),
    'bpm_orbit_rms_m': rms_arrays(orbit_errors(tw_after)).tolist(),
    'correctable_orbit_rms_m': rms_arrays(correctable_orbit_errors(tw_after)).tolist(),
    'c_minus': float(tw_after.c_minus),
    'beta_rms_tolerance': BETA_RMS_TOL,
    'beta_max_tolerance': BETA_MAX_TOL,
    'tune_tolerance': TUNE_TOL,
}
if ramp_completed:
    integral_estimates = {
        key: float(tw_after[key])
        for key in (
            'rad_int_i2',
            'rad_int_i4x',
            'rad_int_i4y',
            'rad_int_i5x',
            'rad_int_i5y',
            'rad_int_eq_gemitt_x',
            'rad_int_eq_gemitt_y',
        )
    }
    line.metadata['orbit_optics_correction']['final_4d_radiation_integrals'] = (
        integral_estimates
    )
    print(
        'Final 4D radiation-integral emittance estimate x/y [pm]:',
        tw_after.rad_int_eq_gemitt_x * 1e12,
        tw_after.rad_int_eq_gemitt_y * 1e12,
    )
line.to_json(OUTPUT, indent=None)
print('Saved', OUTPUT)
print('Maximum quadrupole trim [m^-2]:', np.max(np.abs(values(quad_knobs))))
print(
    'One-turn eigenvalue moduli:', np.abs(np.linalg.eigvals(tw_after.R_matrix[:4, :4]))
)
print('Coupling c_minus:', tw_after.c_minus)
for plane in 'xy':
    targets = np.array([getattr(line[name], 'shift_' + plane) for name in quad_names])
    residual = tw_after[plane][quad_indices] - targets
    print(
        f'Quadrupole entrance residual {plane}, RMS/max [um]:',
        np.sqrt(np.mean(residual**2)) * 1e6,
        np.max(np.abs(residual)) * 1e6,
    )
    dispersion = tw_after['d' + plane] - tw_ref['d' + plane]
    print(
        f'Dispersion error {plane}, RMS/max [m]:',
        np.sqrt(np.mean(dispersion**2)),
        np.max(np.abs(dispersion)),
    )


fig, axes = plt.subplots(3, 2, sharex=True, figsize=(13, 9), layout='constrained')
for j, plane in enumerate('xy'):
    axes[0, j].semilogy(
        tw_ref.s * 1e-3,
        np.maximum(
            1e-7, 100 * np.abs(tw_after['bet' + plane] / tw_ref['bet' + plane] - 1)
        ),
        label='Interleaved correction',
        lw=0.8,
    )
    axes[0, j].set_ylabel(f'Absolute beta beating {plane} [%]')
    axes[0, j].legend()
    axes[1, j].plot(
        tw_ref.s[monitor_indices] * 1e-3,
        orbit_errors(tw_after, min(ramp_fraction, 1.0))[j] * 1e6,
        '.',
        ms=2,
    )
    axes[1, j].set_ylabel(f'Orbit residual {plane} [µm]')
    axes[2, j].plot(
        tw_ref.s * 1e-3, tw_after['d' + plane] - tw_ref['d' + plane], lw=0.8
    )
    axes[2, j].set_ylabel(f'Dispersion error {plane} [m]')
    axes[2, j].set_xlabel('Nominal s [km]')
for axis in axes.flat:
    axis.grid(alpha=0.25)
fig.suptitle(
    f'4D correction: errors={line["on_errors"]:.3f}, nonlinear={line["on_nonlinear"]:.3f}'
)
fig.savefig(PLOT, dpi=180)
print('Saved', PLOT)
history = np.asarray(ramp_history, dtype=float)
if len(history):
    history_fig, history_axes = plt.subplots(
        5, 1, sharex=True, figsize=(10, 12), layout='constrained'
    )
    for j, plane in enumerate('xy'):
        history_axes[0].plot(100 * history[:, 0], history[:, 1 + j] * 1e6, label=plane)
        history_axes[1].plot(
            100 * history[:, 0], 100 * history[:, 3 + j], label=f'{plane} RMS'
        )
        history_axes[1].plot(
            100 * history[:, 0], 100 * history[:, 5 + j], ls='--', label=f'{plane} max'
        )
        history_axes[2].plot(
            100 * history[:, 0], history[:, 7 + j] - tw_ref['q' + plane], label=plane
        )
    history_axes[0].set_ylabel('Orbit RMS [um]')
    history_axes[1].set_ylabel('Beta beating [%]')
    history_axes[2].set_ylabel('Tune error')
    history_axes[3].plot(
        100 * history[:, 0], history[:, 15] * 1e3, '.-', label='BPMs + IPs RMS'
    )
    history_axes[3].plot(
        100 * history[:, 0], history[:, 16] * 1e3, '.-', label='IP max'
    )
    history_axes[3].set_ylabel('Vertical dispersion error [mm]')
    history_axes[4].plot(
        100 * history[:, 0], history[:, 17] * 1e3, '.-', label='BPMs + IPs RMS'
    )
    history_axes[4].plot(
        100 * history[:, 0], history[:, 18] * 1e3, '.-', label='IP max'
    )
    history_axes[4].set_ylabel('Horizontal dispersion error [mm]')
    history_axes[4].set_xlabel(
        'Error fraction [%]; nonlinear strengths ramp together after 15%'
    )
    for ax in history_axes:
        ax.legend()
        ax.grid(alpha=0.25)
        ax.axvline(100 * BOOTSTRAP_FRACTION, color='grey', ls=':', lw=1)
    history_fig.savefig(OUTPUT_DIR / 'history.png', dpi=180)
if SHOW_PLOTS:
    plt.show()

if not reached_stop:
    raise RuntimeError(
        f'Ramp stopped at {ramp_fraction:g}; last accepted machine saved in {CHECKPOINT}'
    )

print(
    f'Finished; {TWISS_CALLS} Twiss calls, {TWISS_SECONDS:.1f} s in Twiss, {perf_counter() - START_TIME:.1f} s total'
)
print(
    f'  Full optics: {TWISS_CALLS - ORBIT_ONLY_CALLS} calls, {TWISS_SECONDS - ORBIT_ONLY_SECONDS:.1f} s; '
    f'orbit only: {ORBIT_ONLY_CALLS} calls, {ORBIT_ONLY_SECONDS:.1f} s'
)
