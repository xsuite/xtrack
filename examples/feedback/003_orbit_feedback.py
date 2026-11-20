# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Follow a correction from the BPM through command delivery to the field.
# A scalar vertical loop and a prescribed sine plus step isolate the delays.

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from scipy.constants import c

################################################################################
# User parameters
################################################################################

########################################
# Damped vertical ring and disturbance
########################################
P0C                 = 45.6e9            # Electron momentum times c [eV]
FREV                = 3300.0            # Revolution frequency [Hz]
QY                  = 0.31              # Vertical tune
BETA_Y              = 10.0              # Vertical beta function [m]
DAMPING             = 0.02              # Fractional amplitude loss per turn
DURATION            = 10.0              # Tracking duration per case [s]
FREQUENCY           = 0.2               # Disturbance frequency [Hz]
SINE_KICK           = 3e-6              # Disturbance amplitude [rad]
STEP_KICK           = 1e-6              # Added disturbance [rad]
STEP_TIME           = 2.0               # Step time [s]

########################################
# Measurement, controller and hardware
########################################
WINDOW_TURNS        = 7                 # Samples per rolling mean
UPDATE_TURNS        = 40                # Turns between calculations
KP                  = 0.1               # Proportional gain
KI                  = 2 * np.pi         # Integral gain [s^-1]
PROCESSING_DELAY    = 0.005             # Time to calculate a request [s]
COMMAND_DELAY       = 0.005             # Command transmission time [s]
RESPONSE_TIME       = 0.005             # Field response time constant [s]
SLEW                = 10e-6             # Maximum field change [rad/s]
KICK_LIMIT          = 20e-6             # Correction limit [rad]

################################################################################
# Prepare the excitation and reference response
################################################################################

########################################
# Get the static kick-to-BPM gain
########################################
# A positive Multipole ksl[0] adds to py. The BPM reads after the damped map.
phase       = 2 * np.pi * QY
matrix      = (1 - DAMPING) * np.array([
    [np.cos(phase), BETA_Y * np.sin(phase)],
    [-np.sin(phase) / BETA_Y, np.cos(phase)],
])
response    = np.linalg.solve(np.eye(2) - matrix, matrix[:, 1])[0]

########################################
# Generate the harmonic and step kicks
########################################
reference   = xt.Particles(mass0 = xt.ELECTRON_MASS_EV, q0 = -1, p0c = P0C)
length      = float(reference.beta0[0]) * c / FREV
n_turns     = round(DURATION * FREV)
kick_time   = np.arange(n_turns) / FREV
time        = (np.arange(n_turns) + 1) / FREV
forcing     = SINE_KICK * np.sin(2 * np.pi * FREQUENCY * kick_time)
forcing  += STEP_KICK * (kick_time >= STEP_TIME)

################################################################################
# Track the same kick waveform with the loop off and on
################################################################################
orbit       = {}
feedbacks   = {}
for active, label in [(False, 'Off'), (True, 'On')]:
    ########################################
    # Build the ring and Exciter
    ########################################
    env = xt.Environment()

    env.elements['disturbance'] = xt.Exciter(
        samples             = forcing,
        sampling_frequency  = FREV,
        frev                = FREV,
        ksl                 = [1.0],
    )
    env.new('corrector', xt.Multipole, ksl = [0.0])
    env['corrector_rad']        = 0.0
    env.ref['corrector'].ksl[0] = env.ref['corrector_rad']

    env.elements['arc'] = xt.LineSegmentMap(
        length          = length,
        qy              = QY,
        bety            = BETA_Y,
        damping_rate_y  = DAMPING,
        damping_rate_py = DAMPING,
    )
    env.elements['bpm'] = xt.BeamStatsMonitor(
        stop_at_turn = n_turns, stats = ['mean_y'],
    )

    ########################################
    # Bind the pickup and the corrector
    ########################################
    # Delivery delays belong to the controller. Field dynamics belong to the
    # actuator bound to the corrector knob.
    actuator    = xt.feedback.PhysicalActuator(
        target                          = 'corrector_rad',
        lower                           = -KICK_LIMIT,
        upper                           = KICK_LIMIT,
        response_time_constant_seconds  = RESPONSE_TIME,
        maximum_slew_per_second         = SLEW,
    )
    fb          = xt.feedback.Feedback(
        name                        = 'orbit',
        active                      = active,
        history_length              = WINDOW_TURNS,
        update_turns                = UPDATE_TURNS,
        processing_delay_seconds    = PROCESSING_DELAY,
        command_delay_seconds       = COMMAND_DELAY,
        log_capacity                = n_turns if active else 1,
    )
    fb.inputs.new(name = 'y', monitor = 'bpm', quantity = 'mean_y', unit = 'm')
    fb.outputs.new(name = 'kick', actuator = actuator, unit = 'rad')

    ########################################
    # Calculate the PI correction
    ########################################
    fb.parameters.new(name = 'response', value = response, unit = 'm/rad')
    fb.parameters.new(name = 'kp', value = KP)
    fb.parameters.new(name = 'ki', value = KI, unit = '1/s')
    fb.state.new(name = 'integral', initial_value = 0.0, unit = 'rad')
    fb.program = '''
y = mean(acquired.y_history)
correction = -y / parameters.response
increment = parameters.ki * event.dt_seconds * correction
state.integral = state.integral + increment
outputs.kick = parameters.kp * correction + state.integral
'''

    env.elements['feedback'] = fb

    line                = env.new_line(components = [
        'disturbance', 'corrector', 'arc', 'bpm', 'feedback',
    ])
    line.particle_ref   = reference

    ########################################
    # Track the whole waveform in one call
    ########################################
    # At zeta = -1 mm, arrival is just inside the turn's Exciter sample.
    # The map leaves zeta fixed at delta = 0.
    particles           = reference.copy()
    particles.zeta[:]   = -1e-3
    line.build_tracker()
    line.feedback.enable()
    line.track(particles, num_turns = n_turns)
    orbit[label]        = line.get('bpm').mean_y.copy()
    feedbacks[label]    = fb

################################################################################
# Measure rejection of the harmonic disturbance
################################################################################

########################################
# Fit the late BPM signal
########################################
fit_start   = STEP_TIME + 1.0
late        = time >= fit_start
phase       = 2 * np.pi * FREQUENCY * time[late]
design      = np.column_stack((
    np.sin(phase), np.cos(phase), np.ones(phase.size),
))
fits        = {
    mode: np.linalg.lstsq(design, y[late], rcond = None)[0]
    for mode, y in orbit.items()
}
amplitude   = {mode: np.hypot(*fit[:2]) for mode, fit in fits.items()}
print('Applied disturbance\n-------------------')
print(f'Sinusoidal kick amplitude [µrad]: {SINE_KICK * 1e6:.1f}')
print(f'Sinusoidal kick frequency [Hz]: {FREQUENCY:g}')
print(f'Added kick at {STEP_TIME:g} s [µrad]: {STEP_KICK * 1e6:+.1f}')
print(f'The step at {STEP_TIME:g} s excites a damped betatron oscillation.')

print('\nVertical orbit rejection\n------------------------')
print(f'Closed-orbit response [m/rad]: {response:+.4f}')
print(f'Fit start [s]: {fit_start:.1f}')
print()
print('Feedback    BPM RMS [µm]    Harmonic amplitude [µm]    Offset [µm]')
for mode, y in orbit.items():
    rms = np.sqrt(np.mean(y[late]**2))
    print(
        f'{mode:8s} {rms * 1e6:15.3f} {amplitude[mode] * 1e6:26.3f} '
        f'{fits[mode][2] * 1e6:+14.3f}'
    )
print(f'Harmonic rejection factor: {amplitude["Off"] / amplitude["On"]:.2f}')

########################################
# Read the command and field times
########################################
# Convert the turn boundary in the log to the time when the beam next sees
# the corrector. It is at the origin of the ring, just before the arc.
log             = feedbacks['On'].log
requests        = [entry for entry in log if entry['executed']]
request_turn    = np.array([entry['completed_turn'] for entry in requests])
requested       = np.array([entry['requested_output'][0] for entry in requests])
target          = np.r_[0.0, [entry['target_output'][0] for entry in log]]
applied         = np.r_[0.0, [entry['applied_output'][0] for entry in log]]
request_time    = np.array([entry['time_seconds'] for entry in requests])
command_time    = time[np.flatnonzero(target)[0] - 1]
first_kick      = time[np.flatnonzero(applied)[0] - 1]

print('\nFirst feedback kick\n-------------------')
print(f'Calculation delay [ms]: {PROCESSING_DELAY * 1e3:g}')
print(f'Transmission delay [ms]: {COMMAND_DELAY * 1e3:g}')
print(f'Corrector field time constant [ms]: {RESPONSE_TIME * 1e3:g}')
print(f'Corrector slew limit [µrad/s]: {SLEW * 1e6:g}')
print('\nEvent'.ljust(39) + 'Time from calculation start [ms]')
for event, event_time in [
    ('Controller starts calculation', request_time[0]),
    ('Command reaches the corrector', command_time),
    ('Beam first sees a changed kick', first_kick),
]:
    print(f'{event:<38} {(event_time - request_time[0]) * 1e3:31.3f}')
print('\nThe corrector starts moving towards the received command at the')
print('next turn boundary. The plot timestamps each command at the start')
print('of its calculation.')

########################################
# Plot the orbit and timing detail
########################################
fig, axes = plt.subplots(2, 1, figsize = (8, 6))
for mode, y in orbit.items():
    axes[0].plot(time, y * 1e6, label = f'Feedback {mode.lower()}')
axes[0].set(xlabel = 'Time [s]', ylabel = 'BPM centroid [µm]')

start, stop = STEP_TIME - 0.02, STEP_TIME + 0.12
show        = (request_time >= start - 0.02) & (request_time <= stop + 0.02)
detail      = (kick_time >= start - 0.001) & (kick_time <= stop + 0.001)
request_ms  = (request_time[show] - STEP_TIME) * 1e3
elapsed_ms  = (kick_time[detail] - STEP_TIME) * 1e3
axes[1].step(
    request_ms, requested[show] * 1e6,
    where = 'post', label = 'Controller command',
)
axes[1].step(
    elapsed_ms, target[:-1][detail] * 1e6,
    where = 'post', label = 'Command at corrector',
)
axes[1].plot(
    elapsed_ms, applied[:-1][detail] * 1e6, label = 'Kick seen by beam',
)
axes[1].axvline(0.0, color = '0.5', ls = ':', label = 'Disturbance step')
axes[1].set(
    xlabel = 'Time from disturbance step [ms]', ylabel = 'Correction [µrad]',
    xlim    = (np.array([start, stop]) - STEP_TIME) * 1e3,
    title   = 'Corrector response at the disturbance step',
)
for axis in axes:
    axis.legend()
fig.tight_layout()
plt.show()
