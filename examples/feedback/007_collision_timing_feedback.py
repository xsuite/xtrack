# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Align the arrival times of two beams at the collision point.
# Compare relative timing with feedback tied to an external master clock.

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as C_LIGHT
import xobjects as xo
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Fixed reference and ring optics
########################################
P0C                 = 45.6e9            # e-/e+ reference momentum times c [eV]
FREV                = 3300.0            # Revolution frequency [Hz]
RF_FREQUENCY        = 121212 * FREV     # RF frequency [Hz]
RF_VOLTAGE          = 20e6              # Peak voltage per beam [V]
MOMENTUM_COMPACTION = 1e-5
DAMPING             = 0.02              # Fractional pzeta damping per turn
BETATRON_BETA       = 50.0              # Transverse beta function [m]
CHICANE_MODEL       = 'bends'           # Four RBends, or 'affine' for a map
N_PARTICLES         = 16                # Macroparticles per beam
DURATION            = 8.0               # Tracking duration per case [s]

########################################
# Static response measurements
########################################
CALIBRATION_TURNS   = 2000              # Turns per response probe
SETTLING_TURNS      = 1600              # Initial transient omitted
PHASE_SCALE         = 0.01              # Differential RF phase scale [rad]
PATH_SCALE          = 50e-6             # Differential path length scale [m]
TIME_SCALE          = 5e-12             # Differential arrival time scale [s]
DELTA_SCALE         = 5e-5              # Relative momentum scale

########################################
# Feedback calculation and actuator response
########################################
UPDATE_TURNS        = 40                # Turns between calculations
KI                  = 2 * np.pi * 1.5   # Integral gain [s^-1]
PROCESSING_DELAY    = 0.0002            # Time to calculate a request [s]
COMMAND_DELAY       = 0.0001            # Command transmission time [s]
RF_RESPONSE_TIME    = 0.001             # RF response time [s]
PATH_RESPONSE_TIME  = 0.005             # Path response time [s]
PHASE_LIMIT         = 0.05              # Maximum RF phase trim [rad]
PATH_LIMIT          = 200e-6            # Maximum path trim [m]
PHASE_SLEW          = 0.2               # Maximum RF phase change [rad/s]
PATH_SLEW           = 0.001             # Maximum path change [m/s]

########################################
# Independent disturbances and bunch-length scale
########################################
PHASE_DISTURBANCE   = 0.008             # RF phase harmonic amplitude [rad]
PHASE_FREQUENCY     = 0.5               # RF phase harmonic frequency [Hz]
PATH_DISTURBANCE    = 30e-6             # Path harmonic amplitude [m]
PATH_FREQUENCY      = 0.25              # Path harmonic frequency [Hz]
PATH_STEP           = 40e-6             # Added path disturbance [m]
STEP_TIME           = 1.0               # Path step time [s]
COMMON_PHASE        = 0.04              # Shared phase disturbance [rad]
COMMON_FREQUENCY    = 0.25              # Shared phase disturbance [Hz]
BUNCH_LENGTH        = 5e-3              # RMS length for comparison [m]
CHICANE_ANGLE       = 0.05              # Nominal bend angle magnitude [rad]

################################################################################
# Model the chicane and the two rings
################################################################################

########################################
# Place the four rectangular bends
########################################
def make_chicane(prefix):
    """Construct the native rectangular-bend chicane in tracking order.

    Parameters
    ----------
    prefix : str
        Prefix for the bend and drift names.

    Returns
    -------
    dict
        Ordered elements with fixed geometry. The field trims change k0.
    """
    elements    = {}
    theta       = CHICANE_ANGLE
    for i, sign in enumerate((1, -1, -1, 1)):
        elements[f'{prefix}_b{i}'] = xt.RBend(
            length_straight = 1.0,
            angle           = sign * theta,
            k0              = sign * 2 * np.sin(theta / 2),
            rbend_model     = 'straight-body',
        )
        if i < 3:
            elements[f'{prefix}_d{i}'] = xt.Drift(length = 5 if i == 1 else 3)
    return elements


########################################
# Measure shear and path trim
########################################
def measure_chicane(context):
    """Measure the physical chicane response by central differences.

    Parameters
    ----------
    context : xobjects.ContextCpu
        Context for this example's tracks.

    Returns
    -------
    tuple of float
        Length [m], d(zeta)/d(pzeta) [m] and path per fractional field trim
        [m] at the reference momentum.
    """
    elements    = make_chicane('chicane')
    line        = xt.Line(elements = elements, element_names = list(elements))
    line.build_tracker(_context = context)
    probe_delta = 1e-4
    probe_trim  = 1e-3
    particles   = xt.Particles(
        _context    = context,
        mass0       = xt.ELECTRON_MASS_EV,
        q0          = -1,
        p0c         = P0C,
        delta       = [-probe_delta, probe_delta],
    )
    pzeta       = np.array(particles.pzeta)
    line.track(particles)
    shear = np.diff(particles.zeta)[0] / np.diff(pzeta)[0]

    bends           = [
        line[name] for name, element in elements.items()
        if isinstance(element, xt.RBend)
    ]
    nominal_fields  = [bend.k0 for bend in bends]
    zeta            = []
    for trim in (-probe_trim, probe_trim):
        for bend, nominal_field in zip(bends, nominal_fields):
            bend.k0 = nominal_field * (1 + trim)
        particle = xt.Particles(
            _context = context, mass0 = xt.ELECTRON_MASS_EV, q0 = -1, p0c = P0C,
        )
        line.track(particle)
        zeta.append(particle.zeta[0])
    path_per_trim = -(zeta[1] - zeta[0]) / (2 * probe_trim)
    return line.get_length(), shear, path_per_trim


########################################
# Build the two rings
########################################
def build_ip_rings(context, num_turns, chicane):
    """Build the same two rings for the response probes and feedback runs.

    Parameters
    ----------
    context : xobjects.ContextCpu
        Context for the two bunches.
    num_turns : int
        Number of complete passages to record at each IP.
    chicane : tuple of float or None
        Physical chicane measurements from measure_chicane. None selects
        the affine path map.

    Returns
    -------
    environment, dict, dict
        Shared variable registry, lines and bunches keyed by beam name.
    """
    harmonic = RF_FREQUENCY / FREV
    if not np.isclose(harmonic, round(harmonic), rtol = 0, atol = 1e-9):
        raise ValueError('RF frequency must be an integer revolution harmonic')
    if not 0 < DAMPING < 1:
        raise ValueError('Use a pzeta damping fraction between zero and one')

    beta            = P0C / np.hypot(P0C, xt.ELECTRON_MASS_EV)
    circumference   = beta * C_LIGHT / FREV
    eta             = MOMENTUM_COMPACTION - (1 - beta**2)
    if eta <= 0:
        raise ValueError('The phase-pi cavity requires a positive slip factor')

    ########################################
    # Preserve the ring slip factor
    ########################################
    if chicane is None:
        chicane_length, chicane_shear = 4.0, 0.02
    else:
        chicane_length, chicane_shear, path_per_trim = chicane

    ring_map = np.eye(6)
    for start, tune in ((0, 0.21), (2, 0.31)):
        mu = 2 * np.pi * tune

        ring_map[start:start + 2, start:start + 2] = [
            [np.cos(mu), BETATRON_BETA * np.sin(mu)],
            [-np.sin(mu) / BETATRON_BETA, np.cos(mu)],
        ]
    # Preserve d(zeta)/d(pzeta) = -eta*C including the device. The Taylor
    # map uses ptau and a beta0 factor in its zeta update.
    ring_map[4, 5] = (-eta * circumference - chicane_shear) / beta**2

    path_map        = np.eye(6)
    path_map[4, 5]  = chicane_shear / beta**2

    ########################################
    # Install the RF, chicane and IP pickups
    ########################################
    env = xt.Environment()
    for knob in (
        'phase_error', 'path_error', 'phase_trim', 'path_trim',
        'common_phase_error', 'common_phase_trim',
    ):
        env[knob] = 0.0
    lines, bunches = {}, {}
    for beam, sign in (('e', 1), ('p', -1)):
        elements = {
            f'{beam}_rf': xt.Cavity(
                voltage     = RF_VOLTAGE,
                frequency   = RF_FREQUENCY,
                phase       = np.pi,
            ),
            f'{beam}_damping': xt.LineSegmentMap(
                length              = 0,
                longitudinal_mode   = 'frozen',
                damping_rate_pzeta  = DAMPING,
            ),
            f'{beam}_ring': xt.FirstOrderTaylorMap(
                length  = circumference - chicane_length,
                m1      = ring_map,
            ),
        }
        if chicane is None:
            elements[f'{beam}_path'] = xt.FirstOrderTaylorMap(
                length = chicane_length, m1 = path_map,
            )
        else:
            elements.update(make_chicane(f'{beam}_path'))
        elements[f'{beam}_ip'] = xt.BeamStatsMonitor(
            stop_at_turn = num_turns, stats = ['mean_zeta', 'mean_delta'],
        )
        env.elements.update(elements)
        reference   = xt.Particles(
            mass0 = xt.ELECTRON_MASS_EV, q0 = -sign, p0c = P0C,
        )
        lines[beam] = xt.Line(
            env             = env,
            element_names   = list(elements),
            particle_ref    = reference,
        )

        ########################################
        # Wire the signed actuator trims
        ########################################
        # The same differential command sends +u/2 to e− and −u/2 to e+.
        phase   = env.ref['phase_error'] + env.ref['phase_trim']
        path    = env.ref['path_error'] + env.ref['path_trim']
        common  = env.ref['common_phase_error'] + env.ref['common_phase_trim']

        env.ref[f'{beam}_rf'].phase = np.pi + common + sign * phase / 2
        if chicane is None:
            env.ref[f'{beam}_path'].m0[4] = -sign * path / (2 * beta)
        else:
            field_trim = sign * path / (2 * path_per_trim)
            for name, element in elements.items():
                if isinstance(element, xt.RBend):
                    env.ref[name].k0 = element.k0 * (1 + field_trim)
        bunches[beam] = xt.Particles(
            _context    = context,
            mass0       = xt.ELECTRON_MASS_EV,
            q0          = -sign,
            p0c         = P0C,
            zeta        = np.linspace(-0.2e-3, 0.2e-3, N_PARTICLES),
            x           = np.linspace(-1e-6, 1e-6, N_PARTICLES),
            y           = np.linspace(1e-6, -1e-6, N_PARTICLES),
            weight      = np.full(N_PARTICLES, 1e10 / N_PARTICLES),
        )
    return env, lines, bunches


########################################
# Convert zeta into arrival time
########################################
def read_ip(lines, beta):
    """Read each beam's recorded arrival time and momentum history.

    Parameters
    ----------
    lines : dict
        Electron and positron lines keyed by beam name.
    beta : float
        Reference speed in units of c.

    Returns
    -------
    dict
        One array per beam, with time [s] and relative momentum columns.
    """
    readings = {}
    for beam, line in lines.items():
        pickup = line.get(f'{beam}_ip')
        # Positive zeta means earlier arrival at the local IP.
        time            = -pickup.get('mean_zeta') / (beta * C_LIGHT)
        delta           = pickup.get('mean_delta')
        readings[beam]  = np.column_stack((time, delta))
    return readings

################################################################################
# Calculate the correction from relative or master-clock measurements
################################################################################
def integrate_ip_error(inputs, previous, constants, event, xp):
    """Integrate the command inferred from the scaled static response.

    Parameters
    ----------
    inputs : mapping
        Per-beam mean zeta [m] and relative momentum at the IP.
    previous : mapping
        Previous dimensionless integral in actuator coordinates.
    constants : mapping
        Reference speed, response inverse, scales and master-clock setting.
    event : mapping
        Timing of the calculation, including the integration interval [s].
    xp : module
        Array operations for the tracking context.

    Returns
    -------
    dict
        Retained integral, differential phase [rad] and path [m] commands,
        and a common phase command [rad] if using the master clock.
    """
    time_e      = -inputs['e_zeta'] / constants['beta_c']
    time_p      = -inputs['p_zeta'] / constants['beta_c']
    delta_error = inputs['e_delta'] - inputs['p_delta']
    errors      = [time_e - time_p, delta_error]
    if constants['master_clock']:
        errors.append((time_e + time_p) / 2)
    error       = xp.stack(errors) / constants['read_scale']
    correction  = -constants['inverse'] @ error
    integral    = previous['integral'] + KI * event['dt_seconds'] * correction
    return {
        'memory': {'integral': integral},
        'outputs': {'command': constants['trim_scale'] * integral},
    }

################################################################################
# Measure the equilibrium response to RF phase and path trims
################################################################################

########################################
# Select the chicane model
########################################
context = xo.ContextCpu()
if CHICANE_MODEL == 'bends':
    chicane = measure_chicane(context)
elif CHICANE_MODEL == 'affine':
    chicane = None
else:
    raise ValueError('CHICANE_MODEL must be bends or affine')

########################################
# Probe the three longitudinal knobs
########################################
beta        = P0C / np.hypot(P0C, xt.ELECTRON_MASS_EV)
trim_knobs  = ('phase_trim', 'path_trim', 'common_phase_trim')
trim_scale  = np.array([PHASE_SCALE, PATH_SCALE, PHASE_SCALE])
read_scale  = np.array([TIME_SCALE, DELTA_SCALE, TIME_SCALE])
response    = np.zeros((3, 3))

for column, knob in enumerate(trim_knobs):
    probe = 0.2 * trim_scale[column]
    for sign in (-1, 1):
        env, lines, bunches = build_ip_rings(
            context, CALIBRATION_TURNS, chicane,
        )
        env[knob]           = sign * probe
        buffer              = context.new_buffer()
        for beam, line in lines.items():
            line.build_tracker(_buffer = buffer)
            line.track(bunches[beam], num_turns = CALIBRATION_TURNS)
        arrival = read_ip(lines, beta)
        e, p    = [
            arrival[beam][SETTLING_TURNS:].mean(axis = 0) for beam in 'ep'
        ]
        settled = np.r_[e - p, (e[0] + p[0]) / 2]
        response[:, column] += sign * settled / (2 * probe)

########################################
# Scale the response before inversion
########################################
# Scale seconds, momentum, radians and metres before comparing ranks.
scaled  = response * trim_scale[None, :] / read_scale[:, None]
rank    = np.linalg.matrix_rank(scaled)
if rank != 3 or np.linalg.cond(scaled) > 20:
    raise ValueError('The scaled longitudinal response is poorly conditioned')

print('RF and chicane calibration\n--------------------------')
if chicane is None:
    print('An affine map represents the chicane path change.')
else:
    print('Each chicane has four 1 m rectangular dipoles.')
    angle = CHICANE_ANGLE * 1e3
    print(f'Design angles [mrad]: '
          f'+{angle:g}, -{angle:g}, -{angle:g}, +{angle:g}')
    print('The trim changes all four dipole fields at fixed design angles.')
print('\nIP difference (e- minus e+)        Per rad of RF   Per mm of path')
print(f'Arrival time [ps] {response[0, 0] * 1e12:+29.3f}'
      f' {response[0, 1] * 1e9:+16.3f}')
print(f'Relative momentum [ppm] {response[1, 0] * 1e6:+23.3f}'
      f' {response[1, 1] * 1e3:+16.3f}')
print()
print(f'Mean arrival response to shared RF phase [ps/rad]: '
      f'{response[2, 2] * 1e12:+.3f}')

################################################################################
# Compare actuator choices and then add a master-clock measurement
################################################################################
num_turns       = round(DURATION * FREV)
time            = np.arange(1, num_turns + 1) / FREV
labels          = (
    'Off', 'RF phase', 'Path', 'RF phase and path', 'Master clock',
)
observed        = {}
clock_arrival   = {}

for label in labels:
    env, lines, bunches = build_ip_rings(context, num_turns, chicane)

    ########################################
    # Synchronize both beams at the end of each turn
    ########################################
    def set_next_turn_disturbance(completed_turn, time_seconds, updates):
        """Set the prescribed disturbances after both beams complete a turn.

        Parameters
        ----------
        completed_turn : int
            Number of completed turns for each beam.
        time_seconds : float
            Common reference time [s].
        updates : tuple of dict
            Boundary records returned by the feedback system.
        """
        env['phase_error']          = PHASE_DISTURBANCE * np.sin(
            2 * np.pi * PHASE_FREQUENCY * time_seconds
        )
        env['path_error']           = PATH_DISTURBANCE * np.sin(
            2 * np.pi * PATH_FREQUENCY * time_seconds
        ) + PATH_STEP * (time_seconds >= STEP_TIME)
        env['common_phase_error']   = COMMON_PHASE * np.sin(
            2 * np.pi * COMMON_FREQUENCY * time_seconds
        )

    boundary    = xt.feedback.SharedFeedbackBoundary(
        ('e', 'p'), after_feedback = set_next_turn_disturbance,
    ).install(lines)
    buffer      = context.new_buffer()
    for line in lines.values():
        line.build_tracker(_buffer = buffer)
    e_ip, p_ip = lines['e'].get('e_ip'), lines['p'].get('p_ip')

    ########################################
    # Bind differential trims and optionally the shared RF trim
    ########################################
    # A single actuator uses the observable it controls most directly.
    master_clock    = label == 'Master clock'
    channels        = 3 if master_clock else 2
    inverse         = np.linalg.inv(scaled[:channels, :channels])
    if label == 'RF phase':
        inverse = np.array([[1 / scaled[0, 0], 0], [0, 0]])
    elif label == 'Path':
        inverse = np.array([[0, 0], [0, 1 / scaled[1, 1]]])
    limits      = np.array([PHASE_LIMIT, PATH_LIMIT, PHASE_LIMIT])[:channels]
    controller  = xt.feedback.FeedbackController(
        'ip_timing',
        program                         = integrate_ip_error,
        output                          = xt.feedback.KnobOutput(
            env, trim_knobs[:channels], lower = -limits, upper = limits,
        ),
        constants                       = {
            'beta_c': beta * C_LIGHT, 'inverse': inverse,
            'read_scale': read_scale[:channels],
            'trim_scale': trim_scale[:channels], 'master_clock': master_clock,
        },
        memory                          = {'integral': np.zeros(channels)},
        acquisition                     = xt.feedback.Acquisition(
            sources = {
                'e_zeta': (e_ip, 'mean_zeta'), 'e_delta': (e_ip, 'mean_delta'),
                'p_zeta': (p_ip, 'mean_zeta'), 'p_delta': (p_ip, 'mean_delta'),
            },
        ),
        update_turns                    = UPDATE_TURNS,
        first_calculation_turn          = UPDATE_TURNS,
        processing_delay_seconds        = PROCESSING_DELAY,
        command_delay_seconds           = COMMAND_DELAY,
        response_time_constant_seconds  = [
            RF_RESPONSE_TIME, PATH_RESPONSE_TIME, RF_RESPONSE_TIME,
        ][:channels],
        slew_mode                       = 'independent',
        maximum_slew_per_second         = [
            PHASE_SLEW, PATH_SLEW, PHASE_SLEW,
        ][:channels],
        log_capacity                    = 1,
    )
    if label == 'Off':
        controller.disable()
    system = xt.feedback.FeedbackSystem([controller], lines = lines)

    ########################################
    # Track both beams together
    ########################################
    multitracker = xt.PipelineMultiTracker(branches = [
        xt.PipelineBranch(line = lines[beam], particles = bunches[beam])
        for beam in ('p', 'e')
    ])
    boundary.bind(multitracker, system = system)
    boundary.track(num_turns = num_turns)

    ########################################
    # Separate relative and clock errors
    ########################################
    readings = read_ip(lines, beta)

    observed[label]         = readings['e'] - readings['p']
    clock_arrival[label]    = np.column_stack(
        (readings['e'][:, 0], readings['p'][:, 0])
    )

################################################################################
# Compare relative control and arrival at the master clock
################################################################################

########################################
# Compare differential arrival and momentum
########################################
late = time >= DURATION / 2
print('\nRelative timing and momentum\n----------------------------')
print('The RF-only loop targets arrival. The path-only loop targets momentum.')
print('The combined loop corrects both observables.')
print()
print('Feedback           Arrival RMS [ps]    Relative momentum RMS [ppm]')
for label in labels[:4]:
    timing, momentum = np.sqrt(np.mean(observed[label][late]**2, axis = 0))
    print(f'{label:<18} {timing * 1e12:15.3f} {momentum * 1e6:27.3f}')
print()
print(f'Path step time [s]: {STEP_TIME:g}')
print(f'Differential path step [µm]: {PATH_STEP * 1e6:.1f}')
print('The path step excites the short synchrotron transient.')

print('\nTiming error scale\n------------------')

########################################
# Relate the error to the bunch length
########################################
sigma_time  = BUNCH_LENGTH / (beta * C_LIGHT)
off_time    = np.sqrt(np.mean(observed['Off'][late, 0]**2))
off_length  = beta * C_LIGHT * off_time
print(f'Comparison bunch RMS length [mm]: {BUNCH_LENGTH * 1e3:g}')
print(f'One bunch length in time [ps]: {sigma_time * 1e12:.2f}')
print(f'Off-loop timing RMS as a distance [mm]: {off_length * 1e3:.3f}')
print(f'Off-loop timing RMS in units of bunch length: '
      f'{off_length / BUNCH_LENGTH:.3f}')
print(f'Head-on collision-point RMS shift [mm]: {off_length * 0.5e3:.3f}')

print('\nArrival at the master clock\n---------------------------')
print(f'Shared RF phase disturbance [mrad]: {COMMON_PHASE * 1e3:g}')
print('The relative loop sees the difference in arrival time.')
print('The clock-based loop also measures their average arrival error')
print('and adjusts the shared RF phase.')
print()
print('Feedback      Mean arrival RMS [ps]   e- RMS [ps]   e+ RMS [ps]')

########################################
# Compare each beam to the master clock
########################################
for label, clock in [('Relative', 'RF phase and path'),
                     ('Master clock', 'Master clock')]:
    arrival         = clock_arrival[clock][late] * 1e12
    mean_rms        = np.sqrt(np.mean(arrival.mean(axis = 1)**2))
    e_rms, p_rms    = np.sqrt(np.mean(arrival**2, axis = 0))
    print(f'{label:14s} {mean_rms:20.3f} {e_rms:13.3f} {p_rms:13.3f}')
print('RMS values use the second half of each run.')

################################################################################
# Plot the arrival and momentum differences for the four actuator choices
################################################################################
fig, axes = plt.subplots(2, 1, sharex = True, figsize = (8, 5))
for label, color, style in zip(
    labels[:4], ('tab:blue', 'tab:orange', 'tab:purple', 'tab:green'),
    ('-', '--', '-.', '-'),
):
    timing, momentum = observed[label].T
    axes[0].plot(
        time, 1e12 * timing, label = label, color = color, ls = style,
        zorder = 3 if style != '-' else 2,
    )
    axes[1].plot(
        time, 1e6 * momentum, color = color, ls = style,
        zorder = 3 if style != '-' else 2,
    )
axes[0].set_ylabel('Arrival difference [ps]')
axes[1].set(xlabel = 'Time [s]', ylabel = 'Momentum difference [ppm]')
axes[0].legend(
    loc = 'lower center', bbox_to_anchor = (0.5, 1.02), ncols = 4,
)
fig.tight_layout()

################################################################################
# Show each beam against the master clock with both feedback strategies
################################################################################
clock_fig, clock_axes = plt.subplots(
    2, 1, sharex = True, sharey = True, figsize = (8, 5),
)
for ax, label, title in zip(
    clock_axes, ('RF phase and path', 'Master clock'),
    ('Relative timing feedback', 'Feedback tied to the master clock'),
):
    e, p = clock_arrival[label].T
    ax.plot(time, 1e12 * e, label = 'Electron')
    ax.plot(time, 1e12 * p, '--', label = 'Positron')
    ax.set(title = title, ylabel = 'Clock error [ps]')
clock_axes[0].legend()
clock_axes[-1].set_xlabel('Time [s]')
clock_fig.tight_layout()
plt.show()
