# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Hold two tunes with quadrupole trims using a driven, noisy BPM signal.
# The line's integrated gradients give a separate measure of the actual tunes.

################################################################################
# Required packages
################################################################################
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt
from scipy.constants import c

################################################################################
# User parameters
################################################################################

########################################
# Electron ring and tune response
########################################
P0C                     = 45.6e9        # Reference momentum times c [eV]
FREV                    = 3300.0        # Revolution frequency [Hz]
QX                      = 0.21          # Horizontal fractional tune
QY                      = 0.31          # Vertical fractional tune
DAMPING                 = 0.003         # Fractional amplitude loss per turn

BETAS                   = (             # (βx, βy) at families A, B [m]
    (20.0, 5.0), (5.0, 20.0),
)
PICKUP_FRACTION         = 0.73          # BPM position / circumference
PROBE                   = 1e-5          # Gradient calibration step [m^-1]

########################################
# Tracking, excitation and pickup
########################################
N_TURNS                 = 33000         # Turns per tracking case
N_PARTICLES             = 32            # Macroparticles
DISTURBANCE_FREQUENCY   = 0.2           # Gradient error frequency [Hz]
DRIVE_RMS               = 1e-8          # RMS dipole kick per plane [rad]
PICKUP_NOISE            = 20e-9         # RMS position noise [m]
SEED                    = 4181          # Launch, drive and noise seed

########################################
# Tune loop and quadrupole response
########################################
HISTORY_TURNS           = 512           # Samples per fit
UPDATE_TURNS            = 64            # Turns between fits
GAIN                    = 0.05          # Fraction of correction per update
HIGHER_GAIN             = 0.10          # Fraction of correction per update
PROCESSING_TIME         = 0.002         # Fit and calculation delay [s]
DELIVERY_TIME           = 0.003         # Delay from request to actuator [s]
FIELD_TIME              = 0.005         # Quadrupole field time constant [s]
TRIM_LIMIT              = 0.005         # Maximum integrated trim [m^-1]
TRIM_SLEW               = 0.01          # Maximum trim rate [m^-1 s^-1]

################################################################################
# Build the ring for the calibration and the tracking comparisons
################################################################################
def build_tune_ring(context, n_turns, *, drive=None, disturbance=None):
    """Build a two-family linear ring with a BPM and optional excitation.

    Parameters
    ----------
    context : xobjects.Context
        Context shared by particles and pickup.
    n_turns : int
        Number of turns the BPM can record.
    drive : ndarray, optional
        Prescribed dipole kicks [rad] by turn and plane.
    disturbance : ndarray, optional
        Prescribed integrated gradient errors [m^-1] by turn and family.

    Returns
    -------
    xtrack.Line
        Ring with independent error and feedback knobs on each family.
    """

    ########################################
    # Place the thin elements and BPM
    ########################################
    fraction = PICKUP_FRACTION
    if not 0 <= fraction <= 1:
        raise ValueError('Set PICKUP_FRACTION between 0 and 1')

    reference   = xt.Particles(
        _context = context, p0c = P0C, mass0 = xt.ELECTRON_MASS_EV, q0 = -1,
    )
    length      = float(reference.beta0[0]) * c / FREV
    elements    = {
        'quad_a': xt.Multipole(knl = [0.0, 0.0]),
        'quad_b': xt.Multipole(knl = [0.0, 0.0]),
        'bpm': xt.BeamStatsMonitor(
            stop_at_turn = n_turns, stats = ['mean_x', 'mean_y'],
        ),
    }

    ########################################
    # Apply the prescribed turn waveforms
    ########################################
    waveforms = {}
    if disturbance is not None:
        waveforms['gradient_a'] = (disturbance[:, 0], [0.0, 1.0], None)
        waveforms['gradient_b'] = (disturbance[:, 1], [0.0, 1.0], None)
    if drive is not None:
        # The normal dipole gives -knl[0] in px and the skew gives ksl[0] in py.
        waveforms['drive_x']    = (drive[:, 0], [-1.0], None)
        waveforms['drive_y']    = (drive[:, 1], None, [1.0])
    for name, (waveform, knl, ksl) in waveforms.items():
        # Two samples per turn put the reference arrival inside a plateau.
        # The same field applies over |zeta| < half the circumference.
        elements[name] = xt.Exciter(
            samples             = np.r_[waveform[0], np.repeat(waveform, 2)],
            knl                 = knl,
            ksl                 = ksl,
            frev                = FREV,
            sampling_frequency  = 2 * FREV,
            start_turn          = -1,
        )

    ########################################
    # Preserve optics across the pickup
    ########################################
    # Two half-ring maps exchange the horizontal and vertical beta functions.
    # Subdivide an arc at the BPM without changing its total phase or damping.
    order   = ['quad_a']
    if disturbance is not None:
        order.append('gradient_a')
    if drive is not None:
        order.extend(['drive_x', 'drive_y'])
    points  = [(0.0, BETAS[0]), (0.5, BETAS[1]), (1.0, BETAS[0])]
    if fraction not in (0.0, 0.5, 1.0):
        points.append((fraction, (10.0, 10.0)))
        points.sort()
    if fraction == 0.0:
        order.append('bpm')
    for index, ((s0, beta0), (s1, beta1)) in enumerate(zip(points, points[1:])):
        arc             = f'arc_{index}'
        span            = s1 - s0
        damping         = 1 - (1 - DAMPING) ** span
        elements[arc]   = xt.LineSegmentMap(
            length          = length * span,
            qx              = QX * span,
            qy              = QY * span,
            betx            = [beta0[0], beta1[0]],
            bety            = [beta0[1], beta1[1]],
            damping_rate_x  = damping,
            damping_rate_px = damping,
            damping_rate_y  = damping,
            damping_rate_py = damping,
        )
        order.append(arc)
        if s1 == 0.5:
            order.append('quad_b')
            if disturbance is not None:
                order.append('gradient_b')
        if s1 == fraction:
            order.append('bpm')

    ########################################
    # Expose independent calibration and feedback knobs
    ########################################
    line = xt.Line(
        elements = elements, element_names = order, particle_ref = reference,
    )
    for family in ('a', 'b'):
        line[f'error_{family}_kl']          = 0.0
        line[f'feedback_{family}_kl']       = 0.0
        line.ref[f'quad_{family}'].knl[1]   = (
            line.ref[f'error_{family}_kl'] + line.ref[f'feedback_{family}_kl']
        )
    return line


########################################
# Read a tune from each one-turn map
########################################
def tracked_tunes(line, context, strengths):
    """Measure the one-turn maps with tracked transverse basis particles.

    Parameters
    ----------
    line : xtrack.Line
        Calibration ring with the prescribed excitation absent.
    context : xobjects.Context
        Context of the ring.
    strengths : array_like
        Integrated gradient trims in families A and B [m^-1].

    Returns
    -------
    ndarray
        Horizontal and vertical fractional tunes.
    """
    line['error_a_kl'], line['error_b_kl'] = map(float, strengths)

    basis       = np.eye(4) * 1e-6
    particles   = xt.Particles(
        _context    = context,
        p0c         = P0C,
        mass0       = xt.ELECTRON_MASS_EV,
        q0          = -1,
        x           = basis[0],
        px          = basis[1],
        y           = basis[2],
        py          = basis[3],
    )
    line.track(particles, num_turns = 1)
    matrix = np.stack((particles.x, particles.px, particles.y, particles.py))
    matrix /= 1e-6
    tunes = []
    for i in (0, 2):
        block   = matrix[i : i + 2, i : i + 2]
        radius  = np.sqrt(np.linalg.det(block))
        tunes.append(np.arccos(np.trace(block) / (2 * radius)) / (2 * np.pi))
    return np.array(tunes)


################################################################################
# Measure the signed tune response of the two quadrupole families
################################################################################

########################################
# Find the unperturbed tunes
########################################
context = xo.ContextCpu()
line    = build_tune_ring(context, 1)
line.build_tracker(_context = context)
target  = tracked_tunes(line, context, [0.0, 0.0])

########################################
# Probe the two gradient families
########################################
columns = []
for probe in PROBE * np.eye(2):
    q_plus  = tracked_tunes(line, context, probe)
    q_minus = tracked_tunes(line, context, -probe)
    columns.append((q_plus - q_minus) / (2 * PROBE))
response = np.column_stack(columns)
if np.linalg.cond(response) > 1e3:
    raise ValueError('The two families do not resolve independent tune changes')
inverse = np.linalg.inv(response)

print('Quadrupole tune response\n------------------------')
print(f'Target fractional tunes: {target[0]:.5f}, {target[1]:.5f}')
print('Plane       Family A [m]     Family B [m]')
for plane, (a, b) in zip('xy', response):
    print(f'{plane:5s} {a:+18.5f} {b:+16.5f}')

################################################################################
# Estimate the betatron pole from the BPM and the applied excitation
################################################################################
def driven_tune(position_m, drive_rad, expected_tune, xp):
    """Fit the pole of a driven, damped betatron mode in one plane.

    Assumes a locally stationary mode and white pickup noise independent
    of the drive. The uncertainty is conditional on that single-mode model.

    Parameters
    ----------
    position_m : array_like
        One-plane BPM history [m].
    drive_rad : array_like
        Applied dipole kicks [rad] on the same turns as the BPM samples.
    expected_tune : float
        Tune prior used to select the admissible fitted branch.
    xp : module
        Array operations for the tracking context.

    Returns
    -------
    tuple
        Fitted tune, uncertainty scale, validity, pole radius, condition
        and individual quality checks.
    """

    ########################################
    # Normalize the paired samples
    ########################################
    position_m, drive_rad = position_m.reshape(-1), drive_rad.reshape(-1)
    if position_m.shape != drive_rad.shape or len(position_m) < 32:
        raise ValueError(
            'Aligned scalar pickup/drive histories need at least 32 turns'
        )
    finite  = xp.all(xp.isfinite(position_m)) & xp.all(
        xp.isfinite(drive_rad)
    )
    y       = xp.where(finite, position_m, 0.0)
    y -= xp.mean(y)
    rms = xp.maximum(xp.sqrt(xp.mean(y * y)), 1e-100)
    y /= rms
    u = xp.where(finite, drive_rad, 0.0)
    u -= xp.mean(u)
    drive_rms = xp.maximum(xp.sqrt(xp.mean(u * u)), 1e-100)
    u /= drive_rms

    ########################################
    # Fit the driven betatron pole
    ########################################
    # Three drive coefficients cover pickup phases with zero prompt response.
    design = xp.stack(
        (y[3:-1], y[2:-2], u[4:], u[3:-1], u[2:-2], xp.ones_like(y[4:])),
        axis = 1,
    )
    # y[n-3] and y[n-4] are uncorrelated with the MA(2) pickup error.
    instruments             = xp.stack(
        (y[1:-3], y[:-4], u[4:], u[3:-1], u[2:-2], xp.ones_like(y[4:])),
        axis = 1,
    )
    moment                  = instruments.T @ design
    fit, _, rank, singular  = xp.linalg.lstsq(
        moment, instruments.T @ y[4:], rcond = None
    )
    radius_squared          = -fit[1]
    radius                  = xp.sqrt(xp.maximum(radius_squared, 1e-100))
    cosine                  = fit[0] / (2 * radius)
    safe_cosine             = xp.clip(cosine, -0.999999, 0.999999)
    tune                    = xp.arccos(safe_cosine) / (2 * xp.pi)
    residual                = y[4:] - design @ fit

    ########################################
    # Estimate the tune uncertainty
    ########################################
    # Bartlett weights include the two lags of the induced BPM error.
    inverse_moment  = xp.linalg.pinv(moment)
    scores          = instruments * residual[:, None]
    meat            = scores.T @ scores
    for lag in [1, 2]:
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / 3) * (cross + cross.T)
    covariance      = inverse_moment @ meat @ inverse_moment.T
    root            = xp.sqrt(1 - safe_cosine**2)
    gradient        = xp.stack(
        (
            -1 / (4 * xp.pi * radius * root),
            -safe_cosine / (4 * xp.pi * radius**2 * root),
            xp.asarray(0.0),
            xp.asarray(0.0),
            xp.asarray(0.0),
            xp.asarray(0.0),
        )
    )
    sigma           = xp.sqrt(xp.maximum(gradient @ covariance @ gradient, 0.0))
    condition       = singular[0] / xp.maximum(singular[-1], 1e-100)
    residual_rms    = xp.sqrt(xp.mean(residual**2))

    ########################################
    # Screen the fit and residuals
    ########################################
    # Longer residual correlations flag departures from the single mode.
    residual_power      = xp.sum(residual**2)
    long_correlations   = xp.stack(
        [
            xp.sum(residual[lag:] * residual[:-lag])
            / xp.maximum(residual_power, 1e-100)
            for lag in range(3, 13)
        ]
    )
    # Small residuals can be correlated when the tune drifts over the window.
    # Apply the correlation test above 0.2% of the RMS BPM signal.
    model_ok    = (residual_rms < 2e-3) | (
        xp.max(xp.abs(long_correlations)) < 0.35
    )
    signal_ok   = finite & (rms > 5e-8) & (drive_rms > 1e-12)
    pole_ok     = (
        (radius_squared > 0) & (radius > 0.95) & (radius < 1.005)
        & (xp.abs(cosine) < 1) & (xp.abs(tune - expected_tune) < 0.015)
    )
    checks      = xp.stack((
        signal_ok, (rank == 6) & (condition < 1e6), pole_ok,
        sigma < 5e-4, residual_rms < 0.1, model_ok,
    ))
    return tune, sigma, xp.all(checks), radius, condition, checks

########################################
# Pair each BPM measurement with its drive
########################################
def acquire_pickup(inputs, event, *, drive_rad, pickup_noise_m):
    """Pair the BPM sample with the programmed drive on the same turn.

    Parameters
    ----------
    inputs : mapping
        Horizontal and vertical BPM centroids [m].
    event : mapping
        Completed turn count for this sample.
    drive_rad : array_like
        Prescribed dipole waveform [rad] on the tracking context.
    pickup_noise_m : array_like
        Prescribed BPM noise [m] on the tracking context.

    Returns
    -------
    dict, bool
        Noisy positions, matched drive and sensor validity.
    """
    turn_index  = event['completed_turn'] - 1
    noise       = pickup_noise_m[turn_index]
    return {
        'x': inputs['x'] + noise[0], 'y': inputs['y'] + noise[1],
        'drive': drive_rad[turn_index],
    }, True

################################################################################
# Integrate the correction when both estimates are valid
################################################################################
def tune_integral(inputs, parameters, previous, event, xp):
    """Invert the tune response and accumulate a correction on valid fits.

    Parameters
    ----------
    inputs : mapping
        Turn histories for each BPM plane [m] and the programmed drive [rad].
    parameters : mapping
        Gain, target tunes, inverse response [m^-1] and update period [s].
    previous : mapping
        Last requested integrated gradients [m^-1].
    event : mapping
        Completed-turn index and interval between calculations [s].
    xp : module
        Array operations for the tracking context.

    Returns
    -------
    dict
        New request, retained integral, and fit diagnostics.
    """
    fits        = [
        driven_tune(
            inputs[f'{plane}_history'], inputs['drive_history'][:, i],
            parameters['target'][i], xp,
        )
        for i, plane in enumerate('xy')
    ]
    tunes       = xp.stack([fit[0] for fit in fits])
    accepted    = fits[0][2] & fits[1][2]
    interval    = event['dt_seconds'] / parameters['update_period_seconds']
    gain        = 1 - (1 - parameters['gain']) ** interval
    correction  = -parameters['inverse_response'] @ (
        tunes - parameters['target']
    )
    command     = xp.where(
        accepted, previous['command'] + gain * correction, previous['command'],
    )
    return {
        'memory': {'command': command},
        'outputs': {'command': command},
        'diagnostics': {
            'tunes': tunes,
            'accepted': accepted,
            'checks': fits[0][5] & fits[1][5],
            'window_first_completed_turn': (
                event['completed_turn'] - len(inputs['x_history']) + 1
            ),
            'window_last_completed_turn': event['completed_turn'],
        },
    }


################################################################################
# Calculate the independent tune of the applied fields
################################################################################
def reference_tunes(strengths):
    """Evaluate the tunes with the gradient fields frozen at one traversal.

    Parameters
    ----------
    strengths : array_like
        Integrated gradients [m^-1]. Last axis is (family A, family B).

    Returns
    -------
    ndarray
        Horizontal and vertical fractional tunes from the analytic map trace.
    """
    # In normalized coordinates the two half-ring rotations have phase μ.
    # Each thin kick changes p_norm by ±β kL x_norm: minus x, plus y.
    kl      = np.asarray(strengths)
    a       = kl[..., 0, None] * np.array(BETAS[0]) * [-1, 1]
    b       = kl[..., 1, None] * np.array(BETAS[1]) * [-1, 1]
    mu      = np.pi * np.array([QX, QY])
    cosine  = (
        np.cos(2 * mu)
        + 0.5 * np.sin(2 * mu) * (a + b)
        + 0.5 * np.sin(mu)**2 * a * b
    )
    if np.any(np.abs(cosine) >= 1):
        raise ValueError('Unstable reference optics')
    return np.arccos(cosine) / (2 * np.pi)

################################################################################
# Prepare one launch, dipole excitation and BPM noise sequence
################################################################################

########################################
# Sample one launch
########################################
rng     = np.random.default_rng(SEED)
cloud   = rng.normal(size = (4, N_PARTICLES))
cloud -= cloud.mean(axis = 1, keepdims = True)
launch = xt.Particles(
    _context    = context,
    p0c         = P0C,
    mass0       = xt.ELECTRON_MASS_EV,
    q0          = -1,
    x           = cloud[0] * 1e-6,
    px          = cloud[1] * 1e-7,
    y           = cloud[2] * 1e-6,
    py          = cloud[3] * 1e-7,
    weight      = np.full(N_PARTICLES, 1e10 / N_PARTICLES),
)

########################################
# Generate the common input waveforms
########################################
# Use the sample precision of the Exciter for the estimator and optics check.
drive   = rng.normal(scale = DRIVE_RMS, size = (N_TURNS, 2)).astype(np.float32)
noise   = rng.normal(scale = PICKUP_NOISE, size = (N_TURNS, 2))
time    = np.arange(N_TURNS) / FREV
phase   = 2 * np.pi * DISTURBANCE_FREQUENCY * time
# Prescribed integrated quadrupole errors [m^-1], including a DC component.
disturbance = np.column_stack((
    0.001 + 3e-4 * np.sin(phase), -0.0008 + 2e-4 * np.sin(phase + 0.6),
)).astype(np.float32)

# The pickup transform reads these fixed waveforms on the tracking context.
xp              = context.nplike_lib
drive_rad       = xp.asarray(drive, dtype = float)
pickup_noise_m  = xp.asarray(noise)
no_noise_m      = xp.zeros_like(pickup_noise_m)

################################################################################
# Compare BPM noise and gain on matched tracks
################################################################################
cases = {}
for mode, bpm_noise, gain in [
    ('Feedback off',   PICKUP_NOISE, 0.0),
    ('No added noise', 0.0,          GAIN),
    ('Nominal gain',   PICKUP_NOISE, GAIN),
    ('Higher gain',    PICKUP_NOISE, HIGHER_GAIN),
]:
    ########################################
    # Prepare the matched ring and launch
    ########################################
    active  = mode != 'Feedback off'
    label   = (
        f'Gain {gain:.2f}, noise {bpm_noise * 1e9:g} nm' if active else mode
    )

    line        = build_tune_ring(
        context, N_TURNS, drive = drive, disturbance = disturbance,
    )
    particles   = launch.copy()

    ########################################
    # Connect the pickup, fit and trims
    ########################################
    pickup      = partial(
        acquire_pickup, drive_rad = drive_rad,
        pickup_noise_m = pickup_noise_m if bpm_noise else no_noise_m,
    )
    feedback    = xt.feedback.Feedback(
        name                        = 'tune',
        active                      = active,
        program                     = tune_integral,
        acquisition_transform       = pickup,
        history_length              = HISTORY_TURNS,
        update_turns                = UPDATE_TURNS,
        first_calculation_turn      = 2 * HISTORY_TURNS,
        processing_delay_seconds    = PROCESSING_TIME,
        command_delay_seconds       = DELIVERY_TIME,
        log_capacity                = N_TURNS if active else 1,
    )
    feedback.inputs.new(name = 'x', monitor = 'bpm', quantity = 'mean_x',
                        unit = 'm')
    feedback.inputs.new(name = 'y', monitor = 'bpm', quantity = 'mean_y',
                        unit = 'm')
    feedback.parameters.new(name = 'gain', value = gain)
    feedback.parameters.new(name = 'target', value = target)
    feedback.parameters.new(name = 'inverse_response', value = inverse,
                            unit = '1/m')
    feedback.parameters.new(name = 'update_period_seconds',
                            value = UPDATE_TURNS / FREV, unit = 's')
    feedback.state.new(name = 'command', initial_value = np.zeros(2),
                       unit = '1/m')

    actuator = xt.feedback.PhysicalActuator(
        target                          = ['feedback_a_kl', 'feedback_b_kl'],
        lower                           = -TRIM_LIMIT,
        upper                           = TRIM_LIMIT,
        response_time_constant_seconds  = FIELD_TIME,
        maximum_slew_per_second         = TRIM_SLEW,
        slew_mode                       = 'coordinated',
    )
    feedback.outputs.new(name = 'command', actuator = actuator, unit = '1/m')
    line.append('tune', feedback)

    ########################################
    # Track with the prescribed exciters and feedback trims
    ########################################
    line.build_tracker(_context = context)
    if active:
        line.feedback.enable()
    line.track(particles, num_turns = N_TURNS)

    ########################################
    # Read the trims used by the beam
    ########################################
    # The trim logged after turn n is applied on turn n + 1.
    trims = np.zeros((N_TURNS, 2))
    if active:
        trims[1:] = [row['applied_output'] for row in feedback.log[:-1]]
    strengths   = disturbance + trims
    cases[mode] = dict(
        true_tunes  = reference_tunes(strengths),
        fields      = strengths,
        fits        = [row for row in feedback.log if row['executed']],
        bpm_y       = line.get('bpm').mean_y.copy(),
        bpm_noise   = bpm_noise,
        gain        = gain if active else None,
        label       = label,
    )

################################################################################
# Compare the actual tunes and the fit quality
################################################################################

########################################
# Reduce the tune and fit records
########################################
completed_time  = time + 1 / FREV
late            = completed_time >= (N_TURNS / FREV) / 2
errors          = {
    mode: case['true_tunes'] - target for mode, case in cases.items()
}
rms             = {
    mode: np.sqrt(np.mean(error[late]**2, axis = 0))
    for mode, error in errors.items()
}
nominal         = cases['Nominal gain']
nominal_fits    = nominal['fits']

########################################
# Explain the fit windows
########################################
print('\nTune fits in a sliding window\n-----------------------------')
print(f'Window length [turns]: {HISTORY_TURNS}')
print(f'Window advance [turns]: {UPDATE_TURNS}')
print(f'Samples reused at each regular update: '
      f'{max(0, HISTORY_TURNS - UPDATE_TURNS)}')
print(f'First fit [completed turn]: {2 * HISTORY_TURNS}')
print()
print('The time-domain fit pairs each BPM position with the Exciter waveform.')
print('A finite window leaves some BPM noise in each tune estimate.')
print('Overlapping windows reuse noisy samples and correlate fit errors.')
print('Tune drift within a window also contributes to the fit error.')
print('A fit is used when both planes pass the signal, pole and fit checks.')
print('When either plane fails, the controller holds its previous command.')

########################################
# Show the overlapping sample windows
########################################
if nominal_fits:
    fit_index       = len(nominal_fits) // 2
    sample_end      = nominal_fits[fit_index]['completed_turn']
    sample_first    = sample_end - HISTORY_TURNS + 1
    print(f'Plotted fit window [completed turns]: '
          f'{sample_first} to {sample_end}')

########################################
# Compare RMS tune errors
########################################
print('\nTune error with different noise and gain\n'
      '----------------------------------------')
print('RMS uses the last half of the run and the map of the applied fields.')
print(f'Gain is the fraction of the correction requested every '
      f'{UPDATE_TURNS} turns.')
print()
print(f'{"Gain":>6} {"Noise [nm]":>12}'
      f' {"Fits used":>10} {"x [10^-3]":>10} {"y [10^-3]":>10}')
for mode, case in cases.items():
    fits            = case['fits']
    used            = sum(bool(row['diagnostics']['accepted']) for row in fits)
    x_rms, y_rms    = 1e3 * rms[mode]
    gain            = (
        f'{case["gain"]:.2f}' if case['gain'] is not None else 'Off'
    )
    fraction        = f'{used}/{len(fits)}' if fits else '—'
    print(f'{gain:>6} {case["bpm_noise"] * 1e9:12.0f}'
          f' {fraction:>10} {x_rms:10.3f} {y_rms:10.3f}')

########################################
# Show the fit rejection checks
########################################
print('\nChecks that rejected a fit\n--------------------------')
check_names = (
    'Weak signal', 'Ill-conditioned fit', 'Pole outside range',
    'Tune uncertainty', 'Large residual', 'Correlated residual',
)
for case in cases.values():
    failed = [
        ~row['diagnostics']['checks'] for row in case['fits']
        if not row['diagnostics']['accepted']
    ]
    if failed:
        counts  = np.sum(failed, axis = 0)
        reasons = ', '.join(f'{name.lower()} ({count})'
                            for name, count in zip(check_names, counts)
                            if count)
        print(f'{case["label"]}: {len(failed)} rejected. {reasons}.')
print('A rejected fit can fail more than one check.')
print('Fit error compares the estimate with the mean map tune in its window.')

################################################################################
# Plot the tune measurements and errors
################################################################################
if nominal_fits:
    detail_first    = max(sample_first, sample_end - 23)
    selected        = slice(detail_first - 1, sample_end)
    samples         = np.arange(detail_first, sample_end + 1)
    position        = (
        nominal['bpm_y'][selected] + noise[selected, 1]
    )
    kicks           = drive[selected, 1]

    sample_fig, sample_axes = plt.subplots(
        3, 1, figsize = (8, 6), height_ratios = (1, 2, 2),
    )

    timeline    = sample_axes[0]
    shown       = nominal_fits[max(0, fit_index - 1) : fit_index + 1]
    for row, fit in enumerate(reversed(shown)):
        end = fit['completed_turn']
        timeline.barh(
            row, HISTORY_TURNS, left = end - HISTORY_TURNS + 0.5,
            height = 0.5, color = ('tab:blue', '0.7')[row],
        )
    timeline.barh(
        0, len(samples), left = detail_first - 0.5,
        height = 0.5, color = 'tab:orange', label = 'Sample detail below',
    )
    timeline.set_yticks(
        range(len(shown)), ('Selected fit', 'Previous fit')[:len(shown)],
    )
    timeline.set_xlabel('Completed turn')
    timeline.legend(loc = 'lower left', bbox_to_anchor = (0, 1), fontsize = 9)
    sample_axes[2].sharex(sample_axes[1])
    sample_axes[1].tick_params(labelbottom = False)
    for axis, values, label, color in [
        (sample_axes[1], position * 1e9, 'Vertical BPM [nm]', 'tab:blue'),
        (sample_axes[2], kicks * 1e9,
         'Vertical dipole kick [nrad]', 'tab:green'),
    ]:
        axis.plot(samples, values, 'o-', ms = 3.5, lw = 0.7, color = color)
        axis.set_ylabel(label)
    sample_axes[-1].set_xlabel('Completed turn')
    sample_fig.suptitle(
        'Sliding window for the tune fit\n'
        f'{nominal["label"]}. Window advance {UPDATE_TURNS} turns',
    )
    sample_fig.tight_layout()

########################################
# Plot the tracked tune errors
########################################
colors      = ('0.4', 'tab:blue', 'tab:orange', 'tab:green')
fig, axes   = plt.subplots(2, 1, sharex = True, figsize = (8, 5))
for index, axis in enumerate(axes):
    for (mode, error), style, color in zip(
        errors.items(), ('-', '--', '-', '-'), colors,
    ):
        axis.plot(
            completed_time, 1e3 * error[:, index], style,
            color  = color, label = cases[mode]['label'],
            zorder = 4 if style == '--' else 2,
        )
    axis.set_ylabel(f'{"XY"[index]} tune error [$10^{{-3}}$]')
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc = 'upper center', ncols = 2, frameon = False,
)
axes[-1].set_xlabel('Time [s]')
fig.tight_layout(rect = (0.03, 0.0, 1.0, 0.89))

########################################
# Plot fit errors and skipped updates
########################################
fit_fig, fit_axes = plt.subplots(
    3, 1, sharex = True, figsize = (8, 6), height_ratios = (3, 3, 1.4),
)

active_cases = [case for case in cases.values() if case['gain'] is not None]
for i, (case, color, marker) in enumerate(
    zip(active_cases, colors[1:], ('.', '.', '^')),
):
    times, differences, rejected = [], [], []
    for row in case['fits']:
        fit     = row['diagnostics']
        end     = row['completed_turn']
        average = case['true_tunes'][end - HISTORY_TURNS : end].mean(axis = 0)
        if fit['accepted']:
            times.append(end / FREV)
            differences.append(fit['tunes'] - average)
        else:
            rejected.append(end / FREV)
    values = np.asarray(differences).reshape(-1, 2)
    for index, axis in enumerate(fit_axes[:2]):
        axis.plot(times, values[:, index] * 1e6, marker, ms = 2.5,
                  color = color, label = case['label'], alpha = 0.75)
        axis.set_ylabel(f'{"XY"[index]} fit error [$10^{{-6}}$]')
    fit_axes[2].plot(rejected, np.full(len(rejected), i), 'kx', ms = 4)
fit_axes[2].set_yticks(
    range(len(active_cases)),
    [f'{case["gain"]:.2f} / {case["bpm_noise"] * 1e9:g} nm'
     for case in active_cases],
)
fit_axes[2].set_ylim(len(active_cases) - 0.5, -0.5)
fit_axes[2].set_title('Skipped updates by gain and noise', loc = 'left')
fit_fig.legend(
    *fit_axes[0].get_legend_handles_labels(),
    loc = 'upper center', ncols = 2, frameon = False,
)
fit_axes[-1].set_xlabel('Time [s]')
fit_fig.tight_layout(rect = (0.03, 0.0, 1.0, 0.90))
plt.show()
