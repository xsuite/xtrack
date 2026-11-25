# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Control vertical separation at a strong–strong collision.
# Reconstruct the separation from position BPMs either side of the IP.

################################################################################
# Required packages
################################################################################
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import xfields as xf
import xobjects as xo
import xtrack as xt
from scipy.constants import physical_constants

################################################################################
# User parameters
################################################################################

########################################
# Beam population and sampling
########################################
POPULATION          = 1e11              # Physical particles per bunch
N_PARTICLES         = 16_384            # Macroparticles per bunch
N_SLICES            = 25                # Longitudinal slices per bunch
SEED                = 8128
OMP_THREADS         = 4                 # 0 for serial, 'auto' for all cores

########################################
# Feedback comparison
########################################
STUDY               = 'bpm'             # Or 'shared-correctors' for four IPs
N_TURNS             = 128               # Passages for the BPM comparison
SEPARATION          = 2.5e-6            # Initial e- minus e+ offset [m]
BPM_INCREMENT       = 200e-9            # Position-BPM quantization step [m]
GAIN                = 0.8               # Fraction of the forecast to correct
MAX_KICK            = 1e-4              # Maximum corrector angle [rad]

########################################
# Kick step and RMS window
########################################
STEP_TURN           = 64                # Completed turn before the step
STEP_KICK           = 10e-6             # Added positron dipole kick [rad]
RMS_START           = 97                # First passage in the RMS window

########################################
# Bare ring and injected emittances
########################################
P0C                 = 45e9              # Reference momentum times c [eV]
CIRCUMFERENCE       = 1000.0            # Ring circumference [m]
QX                  = 0.21              # Bare horizontal tune
QY                  = 0.31              # Bare vertical tune
BETX                = 0.2               # Horizontal beta at the IP [m]
BETY                = 0.1               # Vertical beta at the IP [m]
EMIT_X              = 50e-9             # Geometric emittance [m rad]
EMIT_Y              = 250e-12           # Geometric emittance [m rad]
SIGMA_Z             = 1e-3              # RMS bunch length [m]

########################################
# Optional shared-corrector study
########################################
SHARED_TURNS        = 16                # Turns in the four-IP comparison
SHARED_OFFSET       = 1e-6              # Opposing positron bunch offsets [m]
SHARED_POPULATION   = 1e10              # Physical particles per bunch

################################################################################
# Set the encounter order and predict the positions before collision
################################################################################

########################################
# Set the bare beam sizes and IP order
########################################
SIGMA_X = np.sqrt(BETX * EMIT_X)
SIGMA_Y = np.sqrt(BETY * EMIT_Y)

FOUR_BRANCH_ORDER       = {
    'p1': ('A', 'B', 'C', 'D'),
    'p2': ('C', 'D', 'A', 'B'),
    'e1': ('A', 'D', 'C', 'B'),
    'e2': ('C', 'B', 'A', 'D'),
}
FOUR_BRANCH_PARTNERS    = {
    'p1': dict(A = 'e1', B = 'e2', C = 'e1', D = 'e2'),
    'p2': dict(A = 'e2', B = 'e1', C = 'e2', D = 'e1'),
    'e1': dict(A = 'p1', B = 'p2', C = 'p1', D = 'p2'),
    'e2': dict(A = 'p2', B = 'p1', C = 'p2', D = 'p1'),
}


@dataclass
class Collider:
    """Colliding lines, readouts and the bound feedback pipeline."""

    env: xt.Environment
    lines: dict
    particles: dict
    monitors: dict
    collisions: dict
    manager: xt.PipelineManager
    boundary: xt.feedback.SharedFeedbackBoundary
    controller: xt.feedback.FeedbackController
    encounters: tuple
    response: np.ndarray
    nominal_xi_y: float


########################################
# Transport offsets to the next encounters
########################################
def _bare_prediction(branches, order, encounters, controls, bety, qy):
    """Predict the vertical separation with the bare transport maps.

    Parameters
    ----------
    branches : sequence of str
        Beam names in state-vector order.
    order, encounters, controls : mapping or sequence
        Element order, collision pairs and command indices.
    bety : float
        Vertical beta function at the IPs [m].
    qy : float
        Vertical tune of the bare ring.

    Returns
    -------
    ndarray, ndarray
        Maps from beam state and correction kicks to IP separation [m].
    """
    n_stations  = len(next(iter(order.values())))
    n_controls  = len(set(controls.values()))
    mu          = 2 * np.pi * qy / n_stations
    free        = np.zeros((len(encounters), 2 * len(branches)))
    response    = np.zeros((len(encounters), n_controls))

    for row, (stage, _, positron, electron) in enumerate(encounters):
        phase = (stage + 1) * mu
        for beam, sign in [(electron, 1), (positron, -1)]:
            column = 2 * branches.index(beam)

            free[row, column : column + 2] = sign * np.array(
                [np.cos(phase), bety * np.sin(phase)]
            )
            # A kick before arc j reaches stage s after (s - j + 1) arcs.
            for j, upstream_ip in enumerate(order[beam][:stage + 1]):
                key = (beam, upstream_ip)
                if key in controls:
                    phase_from_kick = (stage - j + 1) * mu
                    response[row, controls[key]] += (
                        sign * bety * np.sin(phase_from_kick)
                    )
    return free, response

################################################################################
# Machine and feedback construction
################################################################################
def build_collider(
    *,
    n_turns                = 16,
    n_particles            = N_PARTICLES,
    n_slices               = N_SLICES,
    topology               = 'two',
    feedback               = True,
    beam_beam_strength     = 1.0,
    gain                   = 1.0,
    seed                   = SEED,
    context                = None,
    initial_separation_m   = 1e-6,
    population             = POPULATION,
    branch_order           = None,
    antisymmetric_offset_m = 0.0,
    bpm_phase_rad          = None,
    controller_builder     = None,
):
    """Build electron and positron branches for head-on collisions.

    Use the same vertical sign in both frames and define separation as e− − e+.

    Parameters
    ----------
    n_turns : int
        Number of complete traversals to record.
    n_particles : int
        Macroparticles per beam.
    n_slices : int
        Number of longitudinal beam–beam slices per collision partner.
    topology : {'two', 'four'}
        Two beams at one IP, or four branches through four IPs.
    feedback : bool
        Whether to calculate and apply corrective requests.
    beam_beam_strength : float
        Multiplier of the nominal beam–beam kick.
    gain : float
        Dimensionless feedback gain.
    seed : int
        Seed for the matched particle ensemble.
    context : xobjects.Context
        Context used for the line and particle arrays.
    initial_separation_m : float
        Initial electron-minus-positron vertical offset [m].
    population : float
        Physical particles per beam.
    branch_order : sequence of str, optional
        Execution order of the pipeline branches.
    antisymmetric_offset_m : float
        Additional antisymmetric branch offset [m].
    bpm_phase_rad : float, optional
        Betatron phase from the collision to each side BPM [rad].
    controller_builder : callable, optional
        Construct the controller from the monitors and environment.

    Returns
    -------
    Collider
        Colliding particles, installed feedback and pipeline tracking state.

    """
    if topology not in ('two', 'four'):
        raise ValueError("topology must be 'two' or 'four'")
    if bpm_phase_rad is not None and (
        topology != 'two' or not 0 < bpm_phase_rad < 0.5
    ):
        raise ValueError('The BPM layout needs one IP and phase in (0, 0.5)')
    if topology == 'two' and antisymmetric_offset_m:
        raise ValueError('The antisymmetric mode needs four branches')
    context         = context or xo.ContextCpu()
    branches        = (
        ('p1', 'e1') if topology == 'two' else ('p1', 'p2', 'e1', 'e2')
    )
    tracking_order  = (
        branches if branch_order is None else tuple(branch_order)
    )
    if len(tracking_order) != len(branches) or set(tracking_order) != set(
        branches
    ):
        raise ValueError(
            'branch_order must be a permutation of the study branches'
        )

    ########################################
    # Assign encounter order and correctors
    ########################################
    order       = (
        {name: ('A',) for name in branches}
        if topology == 'two'
        else FOUR_BRANCH_ORDER
    )
    partners    = (
        {'p1': {'A': 'e1'}, 'e1': {'A': 'p1'}}
        if topology == 'two'
        else FOUR_BRANCH_PARTNERS
    )
    ips         = ('A',) if topology == 'two' else ('A', 'B', 'C', 'D')
    n_stations  = len(ips)

    electron_radius = physical_constants['classical electron radius'][0]
    gamma0          = np.hypot(1, P0C / xt.ELECTRON_MASS_EV)
    nominal_xi_y    = (
        electron_radius * population * BETY
        / (2 * np.pi * gamma0 * SIGMA_Y * (SIGMA_X + SIGMA_Y))
    )

    env     = xt.Environment()
    manager = xt.PipelineManager()

    particles, lines, monitors, collisions = {}, {}, {}, {}

    controls = {}
    for index, ip in enumerate(ips):
        env[f'feedback_{ip}_rad'] = 0.0
        if bpm_phase_rad is not None:
            env[f'disturbance_{ip}_rad'] = 0.0
        controlled_species = 'p' if ip in ['A', 'C'] else 'e'
        for name in branches:
            if name.startswith(controlled_species):
                controls[(name, ip)] = index
    encounters                  = tuple(
        (
            stage,
            order[name][stage],
            name,
            partners[name][order[name][stage]],
        )
        for stage in range(n_stations)
        for name in branches
        if name.startswith('p')
    )
    free_prediction, response   = _bare_prediction(
        branches, order, encounters, controls, BETY, QY
    )
    if bpm_phase_rad is not None:
        response[0, 0] = -BETY * np.sin(2 * np.pi * QY - bpm_phase_rad)

    ########################################
    # Sample the bunches and register collision partners
    ########################################
    for name in branches:
        # Bunch copies within a species deliberately share their initial
        # cloud.
        stream  = 0 if name.startswith('p') else 1
        rng     = np.random.default_rng(
            np.random.SeedSequence(seed, spawn_key = (stream,))
        )
        normal  = rng.normal(size = (6, n_particles))
        normal -= normal.mean(axis = 1, keepdims = True)
        charge      = 1.0 if name.startswith('p') else -1.0
        offset_y_m  = -charge * initial_separation_m / 2
        if name in ['p1', 'p2']:
            offset_y_m += antisymmetric_offset_m * (
                1.0 if name == 'p1' else -1.0
            )
        p = xt.Particles(
            _context    = context,
            mass0       = xt.ELECTRON_MASS_EV,
            q0          = charge,
            p0c         = P0C,
            x           = normal[0] * SIGMA_X,
            px          = normal[1] * SIGMA_X / BETX,
            y           = normal[2] * SIGMA_Y + offset_y_m,
            py          = normal[3] * SIGMA_Y / BETY,
            zeta        = normal[4] * SIGMA_Z,
            delta       = normal[5] * 1e-5,
            weight      = population / n_particles,
        )
        p.init_pipeline(name)
        particles[name] = p
        manager.add_particles(name, 0)
    for ip in ips:
        manager.add_element(f'IP.{ip}')
    for name in branches:
        components = []

        ########################################
        # Slice the opposing bunches
        ########################################
        slicer  = xf.TempSlicer(
            _context    = context,
            n_slices    = n_slices,
            sigma_z     = SIGMA_Z,
            mode        = 'shatilov',
        )
        indices = slicer.get_slice_indices(particles[name])
        xp      = context.nplike_lib
        counts  = xp.stack([xp.sum(indices == i) for i in range(n_slices)])
        if bool(xp.any(counts <= 20)) or not bool(
            xp.sum(counts) == n_particles
        ):
            raise ValueError(
                'Every slice must retain >20 macroparticles for Xfields '
                'moments'
            )

        ########################################
        # Place the corrector, transport and collision
        ########################################
        for ip in order[name]:
            prefix              = f'{name}_{ip}'
            kick_name, arc_name = prefix + '_corrector', prefix + '_arc'

            env.elements[kick_name] = xt.Multipole(
                _context = context, ksl = [0.0]
            )
            if (name, ip) in controls:
                env.ref[kick_name].ksl[0] = env.ref[f'feedback_{ip}_rad']
                if bpm_phase_rad is not None:
                    env.ref[kick_name].ksl[0] = (
                        env.ref[f'feedback_{ip}_rad']
                        + env.ref[f'disturbance_{ip}_rad']
                    )
            # Equal arcs set the encounter intervals for the shared clock.
            length  = CIRCUMFERENCE / n_stations
            tune_y  = QY / n_stations
            if bpm_phase_rad is not None:
                length  = 0.8 * CIRCUMFERENCE
                tune_y  = QY - bpm_phase_rad / np.pi
            env.elements[arc_name]  = xt.LineSegmentMap(
                _context    = context,
                length      = length,
                qx          = QX / n_stations,
                qy          = tune_y,
                betx        = BETX,
                bety        = BETY,
            )
            before_ip               = [kick_name, arc_name]
            after_ip                = []
            if bpm_phase_rad is not None:
                before_ip.extend([prefix + '_in', prefix + '_in_transport'])
                after_ip.extend([prefix + '_out_transport', prefix + '_out'])
                for side in ('in', 'out'):
                    env.elements[prefix + '_' + side + '_transport'] = (
                        xt.LineSegmentMap(
                            _context    = context,
                            length      = 0.1 * CIRCUMFERENCE,
                            qy          = bpm_phase_rad / (2 * np.pi),
                            betx        = BETX,
                            bety        = BETY,
                        )
                    )

            ########################################
            # Install the BPMs and beam–beam element
            ########################################
            sides = ['pre', 'post']
            if bpm_phase_rad is not None:
                sides.extend(['in', 'out'])
            for side in sides:
                monitor = xt.BeamStatsMonitor(
                    _context        = context,
                    stop_at_turn    = n_turns,
                    stats           = ['num_particles', 'mean_y', 'mean_py'],
                )

                monitors[(name, ip, side)]          = monitor
                env.elements[prefix + '_' + side]   = monitor

            update      = xf.ConfigForUpdateBeamBeamBiGaussian3D(
                pipeline_manager        = manager,
                element_name            = f'IP.{ip}',
                slicer                  = slicer,
                partner_particles_name  = partners[name][ip],
                update_every            = 1,
                quasistrongstrong       = False,
            )
            collision   = xf.BeamBeamBiGaussian3D(
                _context            = context,
                phi                 = 0.0,
                alpha               = 0.0,
                other_beam_q0       = particles[partners[name][ip]].q0,
                scale_strength      = beam_beam_strength,
                config_for_update   = update,
                flag_beamstrahlung  = 0,
                flag_bhabha         = 0,
            )

            env.elements[prefix + '_bb']    = collision
            collisions[(name, ip)]          = collision
            components.extend([
                *before_ip, prefix + '_pre', prefix + '_bb', prefix + '_post',
                *after_ip,
            ])
        lines[name] = env.new_line(components = components, name = name)

    ########################################
    # Wait for both beams' pickups before updating the correctors
    ########################################
    boundary = xt.feedback.SharedFeedbackBoundary(
        branches, reference_branch = 'e1',
    )
    boundary.install(lines)
    shared_buffer = context.new_buffer()
    for line in lines.values():
        line.build_tracker(_buffer = shared_buffer)
    if controller_builder is not None:
        controller = controller_builder(monitors, env)
    else:
        sources = {}
        for name in branches:
            last_ip = order[name][-1]
            for coordinate in ['y', 'py']:
                sources[f'{name}_{coordinate}'] = (
                    monitors[(name, last_ip, 'post')],
                    f'mean_{coordinate}',
                )

        ########################################
        # Predict and correct the following encounter
        ########################################
        def direct_separation_feedback(inputs, parameters, constants, xp):
            """Request kicks that cancel the predicted pre-collision offset.

            Parameters
            ----------
            inputs : mapping
                Last post-collision (y, py) for each bunch.
            parameters : mapping
                Dimensionless loop gain.
            constants : mapping
                Bare orbit and corrector response matrices.
            xp : module
                Array library for the tracking context.

            Returns
            -------
            dict
                Kick requests [rad] and predicted separations [m].
            """
            state                   = xp.stack(
                [
                    inputs[f'{name}_{coordinate}']
                    for name in branches
                    for coordinate in ['y', 'py']
                ]
            )
            predicted_separation_m  = constants['free_prediction'] @ state
            command_rad             = -parameters['gain'] * (
                constants['inverse_response'] @ predicted_separation_m
            )
            return {
                'outputs': {'command': command_rad},
                'diagnostics': {
                    'bare_prediction_m': predicted_separation_m
                },
            }

        controller = xt.feedback.FeedbackController(
            'direct_ip_reference',
            acquisition     = xt.feedback.Acquisition(sources = sources),
            program         = direct_separation_feedback,
            log_capacity    = n_turns,
            output          = xt.feedback.KnobOutput(
                env,
                [f'feedback_{ip}_rad' for ip in ips],
                lower   = -2e-4,
                upper   = 2e-4,
            ),
            parameters      = {'gain': gain},
            constants       = {
                'free_prediction': free_prediction,
                'inverse_response': np.linalg.pinv(response, rcond = 1e-10),
            },
        )
    if not feedback:
        controller.disable()

    ########################################
    # Assemble the tracked branches
    ########################################
    system          = xt.feedback.FeedbackSystem([controller], lines = lines)
    multitracker    = xt.PipelineMultiTracker(
        branches = [
            xt.PipelineBranch(line = lines[name], particles = particles[name])
            for name in tracking_order
        ]
    )
    boundary.bind(multitracker, system = system)
    return Collider(
        env,
        lines,
        particles,
        monitors,
        collisions,
        manager,
        boundary,
        controller,
        encounters,
        response,
        nominal_xi_y,
    )

################################################################################
# Tracked BPM and command calibration
################################################################################

########################################
# Store the measured responses
########################################
@dataclass
class DeflectionCalibration:
    weights: np.ndarray
    intercept_m: float
    slope: float
    impulse_m_per_rad: np.ndarray
    scan_separation_m: np.ndarray
    scan_signal_m: np.ndarray
    capture_m: float
    bpm_phase_rad: float
    plant_parameters: dict


########################################
# Reconstruct separation from four BPMs
########################################
def _bpm_signal(monitors, weights, bpm_step = 0.0):
    """Combine the four measured vertical positions into one IP signal.

    Parameters
    ----------
    monitors : mapping
        Position BPMs indexed by beam, IP and side.
    weights : array_like
        Coefficients for the four position BPM readings.
    bpm_step : float, optional
        Quantization interval on each reading [m].

    Returns
    -------
    ndarray
        Weighted position signal for each passage [m].
    """
    readings = np.stack(
        [
            np.asarray(monitors[(name, 'A', side)].mean_y)
            for name in ['e1', 'p1']
            for side in ['in', 'out']
        ],
        axis = 1,
    )
    if bpm_step:
        readings = np.round(readings / bpm_step) * bpm_step
    return readings @ weights


########################################
# Measure the BPM and kick responses
########################################
def calibrate_deflection(
    *,
    n_turns     = 128,
    context     = None,
    scan_m      = 0.2 * SIGMA_Y,
    capture_m   = SIGMA_Y,
    impulse_rad = 2e-7,
    **plant,
):
    """Tracked affine BPM calibration and absolute-command impulse response.

    The calibration is specific to the tracked optics and beam distribution.

    Parameters
    ----------
    n_turns : int
        Number of complete traversals to record.
    context : xobjects.Context
        Context used for the line and particle arrays.
    scan_m : float
        Half range of the tracked separation scan [m].
    capture_m : float
        Assumed linear capture half range [m].
    impulse_rad : float
        Applied command used to track the impulse response [rad].
    plant : dict
        Beam configuration overrides passed to the tracking model.

    Returns
    -------
    DeflectionCalibration
        Sensor response and per-turn impulse response of the tracked plant.

    """
    context = context or xo.ContextCpu()
    if not isinstance(context, xo.ContextCpu):
        raise NotImplementedError(
            'Tracked calibration is a CPU setup study'
        )
    if (
        not isinstance(n_turns, (int, np.integer))
        or isinstance(n_turns, bool)
        or n_turns < 1
    ):
        raise ValueError(
            'Calibration length must be a positive host integer'
        )
    if (
        not np.all(np.isfinite([scan_m, capture_m, impulse_rad]))
        or not 0 < scan_m <= capture_m
        or impulse_rad <= 0
    ):
        raise ValueError(
            'Require positive finite calibration scales and scan within '
            'capture'
        )
    defaults = dict(
        n_particles         = N_PARTICLES,
        n_slices            = N_SLICES,
        seed                = SEED,
        population          = POPULATION,
        beam_beam_strength  = 1.0,
    )
    if set(plant) - set(defaults):
        raise ValueError(
            'Calibration accepts only particle, slice, seed, intensity and '
            'BB-strength settings'
        )
    defaults.update(plant)
    plant = defaults

    ########################################
    # Scan relative position at the IP
    ########################################
    phi         = 0.25
    phase_to_ip = 2 * np.pi * QY - phi
    # Symmetric BPM phases cancel incoming angle in the bare transport.
    weights                 = (
        np.array([1.0, 1.0, -1.0, -1.0]) / (2 * np.cos(phi))
    )
    separations, signals    = [], []
    for separation in [-scan_m, 0.0, scan_m]:
        study = build_collider(
            n_turns         = 1,
            context         = context,
            feedback        = False,
            bpm_phase_rad   = phi,
            **plant,
        )
        # Launch a pure relative position offset and zero angle at the IP.
        for beam, sign in [('e1', 1), ('p1', -1)]:
            particles   = study.particles[beam]
            offset      = sign * separation / 2
            y           = offset * np.cos(phase_to_ip)
            py          = offset * np.sin(phase_to_ip) / BETY
            particles.y  += y - np.mean(particles.y)
            particles.py += py - np.mean(particles.py)
        study.boundary.track(num_turns = 1)
        y_p = study.monitors[('p1', 'A', 'pre')].mean_y[0]
        y_e = study.monitors[('e1', 'A', 'pre')].mean_y[0]
        separations.append(y_e - y_p)
        signals.append(_bpm_signal(study.monitors, weights)[0])
    slope, intercept = np.polyfit(separations, signals, 1)
    if not 0.5 < slope < 1.5:
        raise ValueError(
            'Unusable tracked four-BPM separation calibration'
        )

    ########################################
    # Track a single-turn corrector impulse
    ########################################
    runs = []
    for kick in [0.0, impulse_rad]:
        study = build_collider(
            n_turns                 = n_turns,
            context                 = context,
            feedback                = False,
            bpm_phase_rad           = phi,
            initial_separation_m    = 0.0,
            **plant,
        )

        study.env['disturbance_A_rad'] = kick
        study.boundary.track(num_turns = 1)
        study.env['disturbance_A_rad'] = 0.0
        if n_turns > 1:
            study.boundary.track(num_turns = n_turns - 1)
        runs.append(
            (_bpm_signal(study.monitors, weights) - intercept) / slope
        )
    impulse = (runs[1] - runs[0]) / impulse_rad
    if not np.isfinite(impulse).all() or abs(impulse[0]) < 1e-4:
        raise ValueError(
            'Unusable tracked first-passage command response'
        )
    return DeflectionCalibration(
        weights,
        intercept,
        slope,
        impulse,
        np.asarray(separations),
        np.asarray(signals),
        capture_m,
        phi,
        dict(plant),
    )

################################################################################
# Adaptive IP feedback with measured drive response
################################################################################
def adaptive_ip_feedback(inputs, parameters, previous, constants, xp):
    """Predict the next collision error and request a bounded correction.

    Parameters
    ----------
    inputs : mapping
        Published measurements and actuator readbacks.
    parameters : mapping
        Controller gains and numerical settings.
    previous : mapping
        Retained controller state from the preceding update.
    constants : mapping
        Fixed response, scales and reference values.
    xp : module
        Array operations for the tracking context.

    Returns
    -------
    dict
        Retained histories, the correction request and estimator diagnostics.
    """

    ########################################
    # Reconstruct the measured separation
    ########################################
    bpm = xp.stack(
        [
            inputs[f'{name}_{side}']
            for name in ['e1', 'p1']
            for side in ['in', 'out']
        ]
    )
    if parameters['bpm_lsb_m'] > 0:
        bpm = (
            xp.round(bpm / parameters['bpm_lsb_m'])
            * parameters['bpm_lsb_m']
        )
    measured = (
        constants['weights'] @ bpm - constants['intercept']
    ) / constants['slope']

    ########################################
    # Subtract earlier corrector kicks
    ########################################
    old_commands    = previous['commands']
    forecast_error  = measured - (
        previous['prediction_m']
        + constants['impulse'][0]
        * (old_commands[0] - previous['last_held_command_rad'])
    )
    self_motion     = constants['impulse'] @ old_commands
    disturbance     = measured - self_motion
    history         = xp.concatenate(
        (xp.atleast_1d(disturbance), previous['history'][:-1])
    )
    sensor_history  = xp.concatenate(
        (xp.atleast_1d(measured), previous['sensor_history'][:-1])
    )

    ########################################
    # Fit the free motion and forecast
    ########################################
    count       = previous['count'] + 1
    scale       = xp.maximum(xp.max(xp.abs(history)), 1e-30)
    normalised  = history / scale
    indices     = xp.arange(20)[:, None] + xp.arange(5)[None, :] + 1

    coefficients, _, rank, _ = xp.linalg.lstsq(
        normalised[indices], normalised[:20], rcond = 1e-5
    )

    predicted_disturbance   = coefficients @ history[:5]
    old_drive_next          = constants['impulse_next'] @ old_commands
    # Forecast the next passage if the present command remains held.
    prediction = (
        predicted_disturbance
        + old_drive_next
        + constants['impulse'][0] * old_commands[0]
    )

    ########################################
    # Screen the forecast and limit the kick
    ########################################
    accepted    = (
        (count >= 25)
        & (rank >= 2)
        & (xp.linalg.norm(coefficients) < 10.0)
        & xp.all(xp.abs(sensor_history) <= parameters['capture_m'])
        & (xp.abs(prediction) <= 3 * parameters['capture_m'])
        & xp.isfinite(prediction)
    )
    requested   = (
        old_commands[0]
        - parameters['gain'] * prediction / constants['impulse'][0]
    )
    bounded     = xp.clip(
        requested,
        -parameters['max_command_rad'],
        parameters['max_command_rad'],
    )
    command     = xp.where(accepted, bounded, previous['commands'][0])
    commands    = xp.concatenate((xp.atleast_1d(command), old_commands[:-1]))
    return {
        'memory': {
            'commands': commands,
            'history': history,
            'sensor_history': sensor_history,
            'count': count,
            'prediction_m': prediction,
            'last_held_command_rad': old_commands[0],
        },
        'outputs': {'command': xp.atleast_1d(command)},
        'diagnostics': {
            'offset_m': measured,
            'forecast_error_m': forecast_error,
            'accepted': accepted,
            'limited': accepted & (bounded != requested),
        },
    }


########################################
# Bind the calibrated controller
########################################
def build_bpm_collider(
    calibration,
    *,
    n_turns         = 128,
    context         = None,
    feedback        = True,
    gain            = GAIN,
    max_command_rad = MAX_KICK,
    bpm_lsb_m       = 0.0,
    **plant,
):
    """Forecast the separation after subtracting the earlier commands.

    The calibrated linear impulse response assumes an instantaneous actuator.

    Parameters
    ----------
    calibration : DeflectionCalibration
        Tracked sensor and command response for this collision model.
    n_turns : int
        Number of complete traversals to record.
    context : xobjects.Context
        Context used for the line and particle arrays.
    feedback : bool
        Whether to calculate and apply corrective requests.
    gain : float
        Dimensionless feedback gain.
    max_command_rad : float
        Bound on the differential command magnitude [rad].
    bpm_lsb_m : float
        Position BPM quantization increment [m]. Zero is ideal.
    plant : dict
        Beam configuration overrides passed to the tracking model.

    Returns
    -------
    Collider
        Matched two-beam tracking case with the calibrated feedback.

    """
    if n_turns > len(calibration.impulse_m_per_rad):
        raise ValueError(
            'Calibrate an impulse response covering the entire study'
        )
    for name, value in calibration.plant_parameters.items():
        if name in plant and plant[name] != value:
            raise ValueError(
                f'Plant setting {name} differs from the tracked calibration'
            )
        plant[name] = value
    if (
        not 0 <= gain <= 1
        or not np.isfinite(max_command_rad)
        or max_command_rad <= 0
    ):
        raise ValueError(
            'Use gain in [0,1] and a finite positive command limit'
        )
    if not np.isfinite(bpm_lsb_m) or bpm_lsb_m < 0:
        raise ValueError(
            'BPM quantisation step must be finite and nonnegative'
        )

    def build_controller(monitors, env):
        """Bind the four BPMs to the calibrated predictor.

        Parameters
        ----------
        monitors : mapping
            Pickups indexed by branch, IP and side.
        env : xtrack.Environment
            Environment that owns the vertical corrector knob.

        Returns
        -------
        xtrack.feedback.FeedbackController
            Single differential IP correction channel.
        """
        sources = {
            f'{name}_{side}': (monitors[(name, 'A', side)], 'mean_y')
            for name in ['e1', 'p1']
            for side in ['in', 'out']
        }

        impulse = np.asarray(calibration.impulse_m_per_rad[:n_turns])
        return xt.feedback.FeedbackController(
            'adaptive_bpm_ip',
            acquisition     = xt.feedback.Acquisition(sources = sources),
            program         = adaptive_ip_feedback,
            log_capacity    = n_turns,
            output          = xt.feedback.KnobOutput(
                env,
                ['feedback_A_rad'],
                lower   = -max_command_rad,
                upper   = max_command_rad,
            ),
            parameters      = {
                'gain': gain,
                'capture_m': calibration.capture_m,
                'max_command_rad': max_command_rad,
                'bpm_lsb_m': bpm_lsb_m,
            },
            constants       = {
                'weights': calibration.weights,
                'intercept': calibration.intercept_m,
                'slope': calibration.slope,
                'impulse': impulse,
                'impulse_next': np.concatenate((impulse[1:], [0.0])),
            },
            memory          = {
                'commands': np.zeros(n_turns),
                'history': np.zeros(25),
                'sensor_history': np.zeros(25),
                'count': 0,
                'prediction_m': 0.0,
                'last_held_command_rad': 0.0,
            },
        )

    study = build_collider(
        n_turns             = n_turns,
        context             = context,
        feedback            = feedback,
        bpm_phase_rad       = calibration.bpm_phase_rad,
        controller_builder  = build_controller,
        **plant,
    )
    return study

################################################################################
# Select the collision study
################################################################################
if STUDY not in ('bpm', 'shared-correctors'):
    raise ValueError('Choose bpm or shared-correctors')

context             = xo.ContextCpu(omp_num_threads = OMP_THREADS)
shared              = STUDY == 'shared-correctors'
population          = SHARED_POPULATION if shared else POPULATION
turns               = SHARED_TURNS if shared else N_TURNS
passages            = np.arange(1, turns + 1)
studies, ip_offsets = {}, {}

################################################################################
# Track the selected comparison
################################################################################
if STUDY == 'bpm':
    if not 25 < STEP_TURN < RMS_START <= turns:
        raise ValueError('Set the kick after the first fit and before the RMS '
                         'window')
    if not np.isfinite(BPM_INCREMENT) or BPM_INCREMENT <= 0:
        raise ValueError('Use a positive BPM increment for the comparison')

    ########################################
    # Calibrate the four-BPM estimator
    ########################################
    calibration = calibrate_deflection(
        n_turns     = turns,
        context     = context,
        population  = population,
        n_particles = N_PARTICLES,
        n_slices    = N_SLICES,
        seed        = SEED,
    )

    ########################################
    # Track the same positron kick step
    ########################################
    cases           = (
        ('Off', False, 0.0, '0.45'),
        ('Ideal BPMs', True, 0.0, 'tab:blue'),
        ('Quantized BPMs', True, BPM_INCREMENT, 'tab:orange'),
    )
    sensor_offsets  = {}
    for label, active, bpm_step, _ in cases:
        study = build_bpm_collider(
            calibration,
            n_turns                 = turns,
            context                 = context,
            feedback                = active,
            initial_separation_m    = SEPARATION,
            bpm_lsb_m               = bpm_step,
        )
        study.boundary.track(num_turns = STEP_TURN)
        study.env['disturbance_A_rad'] = STEP_KICK
        study.boundary.track(num_turns = turns - STEP_TURN)

        ########################################
        # Record the true and BPM separations
        ########################################
        studies[label] = study

        y_e = study.monitors[('e1', 'A', 'pre')].mean_y
        y_p = study.monitors[('p1', 'A', 'pre')].mean_y

        ip_offsets[label]       = (y_e - y_p)[:, None]
        pickup                  = _bpm_signal(
            study.monitors, calibration.weights, bpm_step,
        )
        sensor_offsets[label]   = (
            pickup - calibration.intercept_m
        ) / calibration.slope

else:
    ########################################
    # Share one setting per IP
    ########################################
    for label, active in [('Off', False), ('On', True)]:
        study = build_collider(
            n_turns                 = turns,
            context                 = context,
            topology                = 'four',
            feedback                = active,
            antisymmetric_offset_m  = SHARED_OFFSET,
            population              = population,
            n_particles             = N_PARTICLES,
            n_slices                = N_SLICES,
            seed                    = SEED,
            initial_separation_m    = SEPARATION,
        )
        study.boundary.track(num_turns = turns)
        studies[label]      = study
        ip_offsets[label]   = np.column_stack([
            study.monitors[(electron, ip, 'pre')].mean_y
            - study.monitors[(positron, ip, 'pre')].mean_y
            for _, ip, positron, electron in study.encounters
        ])

################################################################################
# Report the collision feedback results
################################################################################
if STUDY == 'bpm':
    ########################################
    # Summarize the matched BPM comparison
    ########################################
    print('Measuring the IP separation\n---------------------------')
    print(f'Nominal vertical beam–beam tune shift: {study.nominal_xi_y:.4f}')
    print('Each ring has one BPM before the collision and one after it.')
    print('The four readings estimate the electron-minus-positron separation.')
    print(f'BPM rounding step [nm]: {BPM_INCREMENT * 1e9:g}')
    print('The estimate is compared with the tracked separation at the IP.')

    ########################################
    # Identify when the first kick can act
    ########################################
    print('\nFeedback timing\n---------------')
    print('Each turn, the fit uses the latest 25 collision passages.')
    print('The first fit follows passage 25. Its kick can act on passage 26.')
    print('Electronics delay and corrector response time are both zero.')

    ########################################
    # Compare the late RMS and survival
    ########################################
    print('\nResponse to a vertical kick step\n'
          '--------------------------------')
    print(f'Positron kick step [µrad]: {STEP_KICK * 1e6:+.3f}')
    print(f'Kick changes after passage: {STEP_TURN}')
    print(f'RMS window [passages]: {RMS_START} to {turns}')
    print()
    print('Feedback        Separation RMS [nm]  Estimate error RMS [nm]'
          '  Survival e-/e+ [%]')
    late = passages >= RMS_START
    for label, _, _, _ in cases:
        truth                   = ip_offsets[label][late, 0]
        error                   = sensor_offsets[label][late] - truth
        rms_sep, rms_error      = 1e9 * np.sqrt(
            np.mean(np.stack((truth, error))**2, axis = 1)
        )
        survival_e, survival_p  = (
            100 * np.mean(studies[label].particles[name].state > 0)
            for name in ('e1', 'p1')
        )
        survival                = f'{survival_e:.2f} / {survival_p:.2f}'
        print(f'{label:15s} {rms_sep:19.3f} {rms_error:24.3f}  {survival:>16s}')

    ########################################
    # Plot rejection and the applied angle
    ########################################
    fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 5.5))
    for label, active, _, color in cases:
        y = ip_offsets[label][:, 0]
        axes[0].plot(passages, y * 1e9, label = label, color = color)
        if active:
            command             = [row['applied_output'][0]
                       for row in studies[label].controller.log]
            kick_at_collision   = np.r_[0.0, command[:-1]]
            axes[1].step(passages, kick_at_collision * 1e6,
                         where = 'post', color = color)
    axes[0].set_ylabel('IP separation [nm]')
    axes[1].set(xlabel = 'Collision passage', ylabel = 'Corrector kick [µrad]')
    for ax in axes:
        ax.axvline(STEP_TURN + 0.5, color = '0.35', ls = '--', lw = 1)
        ax.axvspan(RMS_START - 0.5, turns + 0.5, color = '0.5', alpha = 0.12)
    axes[0].legend(
        loc = 'lower center', bbox_to_anchor = (0.5, 1.02), ncols = 3,
    )
    fig.tight_layout()

    ########################################
    # Plot reconstruction against tracking
    ########################################
    fig, axes   = plt.subplots(2, 1, sharex = True, figsize = (8, 5.5))
    truth       = ip_offsets['Ideal BPMs'][:, 0]
    estimate    = sensor_offsets['Ideal BPMs']
    axes[0].plot(passages, truth * 1e9, color = '0.3',
                 label = 'Simulated IP separation')
    axes[0].plot(passages, estimate * 1e9, '--', color = 'tab:blue',
                 label = 'Reconstructed from four BPMs')
    axes[0].set_ylabel('IP separation [nm]')
    axes[0].legend(
        loc = 'lower center', bbox_to_anchor = (0.5, 1.02), ncols = 2,
    )
    for label, _, _, color in cases[1:]:
        error = sensor_offsets[label] - ip_offsets[label][:, 0]
        axes[1].plot(passages, error * 1e9, label = label, color = color)
    axes[1].set(
        xlabel = 'Collision passage', ylabel = 'Reconstruction error [nm]',
    )
    axes[1].legend(loc = 'upper right')
    fig.tight_layout()

else:
    ########################################
    # Separate the single-turn limit from the multi-turn result
    ########################################
    # The first command cancels the part in the response matrix column space.
    # The residual here is a prediction. The lower panel uses actual tracking.
    first       = study.controller.log[0]
    predicted   = first['diagnostics']['bare_prediction_m']
    remaining   = predicted + study.response @ first['applied_output']
    rank        = np.linalg.matrix_rank(study.response)

    print('One corrector setting per IP\n----------------------------')
    print('Each ring carries two bunches. Partners swap at IPs B and D.')
    print('Both collisions at an IP share the same corrector setting.')
    print('Use tracked bunch positions and angles to predict the next offsets.')
    print(f'Nominal vertical tune shift per IP: {study.nominal_xi_y:.4f}')
    print()
    print(f'{rank} independent settings for {len(predicted)} separations.')
    print(f'Turn 2 prediction before correction, RMS [µm]: '
          f'{1e6 * np.sqrt(np.mean(predicted**2)):.3f}')
    print(f'Turn 2 prediction after correction, RMS [µm]: '
          f'{1e6 * np.sqrt(np.mean(remaining**2)):.3f}')
    print()
    print('Feedback     Last tracked separation RMS [µm]')
    for label, offsets in ip_offsets.items():
        print(f'{label:10s} {1e6 * np.sqrt(np.mean(offsets[-1]**2)):34.4f}')

    ########################################
    # Plot the shared-corrector residuals
    ########################################
    fig, axes   = plt.subplots(2, 1, figsize = (8, 6.5))
    order       = sorted(range(len(study.encounters)),
                   key = lambda i: (study.encounters[i][1],
                                    study.encounters[i][3]))
    caption     = [
        f'IP {ip}\n' rf'$e^-_{{{electron[1:]}}},\ e^+_{{{positron[1:]}}}$'
        for _, ip, positron, electron in (study.encounters[i] for i in order)
    ]
    pairs       = np.arange(len(caption))
    axes[0].bar(pairs - 0.19, predicted[order] * 1e6, 0.38,
                label = 'Before correction', color = '0.5')
    axes[0].bar(pairs + 0.19, remaining[order] * 1e6, 0.38,
                label = 'After shared correction', color = 'tab:blue')
    axes[0].set(title = 'Predicted separation on turn 2',
                ylabel = 'Electron minus positron [µm]')
    axes[0].set_xticks(pairs, caption)
    axes[0].legend(
        loc = 'lower center', bbox_to_anchor = (0.5, 1.13), ncols = 2,
    )
    for label, offsets in ip_offsets.items():
        rms = np.sqrt(np.mean(offsets**2, axis = 1))
        axes[1].plot(
            passages, rms * 1e6, '.-', label = f'Feedback {label.lower()}',
        )
    axes[1].set(
        xlabel = 'Turn', ylabel = 'Tracked separation RMS [µm]',
    )
    axes[1].legend()
    fig.tight_layout()

plt.show()
