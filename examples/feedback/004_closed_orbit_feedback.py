# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Reject a time-varying orbit disturbance with limited corrector kicks.
# Use the response modes to identify which BPM patterns the loop can correct.

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
# Damped linear electron ring
########################################
P0C                     = 45.6e9        # Reference momentum times c [eV]
FREV                    = 3300.0        # Revolution frequency [Hz]
QX                      = 0.21          # Horizontal tune
QY                      = 0.31          # Vertical tune
BETA                    = 10.0          # Beta function in both planes [m]
DAMPING                 = 0.02          # Fractional amplitude loss per turn
DURATION                = 10.0          # Tracking duration per case [s]

# Locations as fractions of circumference in the tracking direction.
BPM_POSITIONS           = (0.125, 0.375, 0.625, 0.875)
CORRECTOR_POSITIONS     = (0.0, 0.25, 0.5, 0.5001)
DISTURBANCE_POSITIONS   = (0.1, 0.35, 0.6, 0.85)

########################################
# Response calibration and feedback
########################################
PROBE_TURNS             = 1400          # Turns per signed corrector probe
SETTLE_TURNS            = 1200          # Turns omitted before averaging
PROBE                   = 2e-6          # Probe kick [rad]
SVD_CUTOFF              = 0.01          # Relative singular value threshold
WINDOW_TURNS            = 7             # Turns in the BPM average
UPDATE_TURNS            = 40            # Turns between calculations
KP                      = 0.1           # Proportional gain
KI                      = 2 * np.pi     # Integral gain [s^-1]

PROCESSING_DELAY        = 0.005         # Measurement to request [s]
COMMAND_DELAY           = 0.005         # Request to delivered target [s]
RESPONSE_TIME           = 0.005         # Corrector field time constant [s]
SLEW                    = 10e-6         # Rate limit per corrector [rad/s]
KICK_LIMIT              = 20e-6         # Field limit per corrector [rad]

########################################
# Independent dipole excitation
########################################
FREQUENCY               = 0.2           # Harmonic frequency [Hz]
AMPLITUDES              = np.array(     # Kicks [rad]
    [2.0, -1.5, 0.8, -1.2],
) * 1e-6
PHASES                  = np.array(     # Phases [rad]
    [0.0, 0.7, 1.4, 2.1],
)
STEP_TIME               = 2.0           # Step time [s]
STEP_KICKS              = np.array(     # Kicks [rad]
    [1.0, 0.4, -0.3, 0.8],
) * 1e-6

################################################################################
# Build the same ring for calibration and correction
################################################################################
def build_distributed_ring(num_turns, waveform = None):
    """Place thin vertical correctors and BPMs in a damped linear ring.

    Parameters
    ----------
    num_turns : int
        Number of turns recorded at each BPM.
    waveform : ndarray of shape (num_turns, 4), optional
        Independent vertical kicks [rad] for the four disturbance stations.

    Returns
    -------
    xtrack.Line
        Ring with four named corrector knobs and four named BPMs.
    """
    reference     = xt.Particles(
        mass0 = xt.ELECTRON_MASS_EV, q0 = -1, p0c = P0C,
    )

    ########################################
    # Place the BPMs and dipoles
    ########################################
    circumference   = c * float(reference.beta0[0]) / FREV
    stations        = [(1.0, 'end', None)]
    for i, position in enumerate(CORRECTOR_POSITIONS, 1):
        stations.append((position, f'cor{i}', xt.Multipole(ksl = [0.0])))
    for i, position in enumerate(BPM_POSITIONS, 1):
        bpm = xt.BeamStatsMonitor(stop_at_turn = num_turns, stats = ['mean_y'])
        stations.append((position, f'bpm{i}', bpm))
    if waveform is not None:
        for i, position in enumerate(DISTURBANCE_POSITIONS):
            exciter = xt.Exciter(
                ksl                 = [1.0],
                samples             = waveform[:, i],
                sampling_frequency  = FREV,
                frev                = FREV,
            )
            stations.append((position, f'drive{i + 1}', exciter))

    ########################################
    # Connect stations with damped maps
    ########################################
    elements    = {}
    previous    = 0.0
    for position, name, element in sorted(stations):
        fraction = position - previous
        if fraction > 0.0:
            damping = 1.0 - (1.0 - DAMPING) ** fraction

            elements[f'to_{name}'] = xt.LineSegmentMap(
                length          = circumference * fraction,
                qx              = QX * fraction,
                qy              = QY * fraction,
                betx            = BETA,
                bety            = BETA,
                damping_rate_y  = damping,
                damping_rate_py = damping,
            )
        if element is not None:
            elements[name] = element
        previous = position

    line = xt.Line(
        elements        = elements,
        element_names   = list(elements),
        particle_ref    = reference,
    )
    for i in range(1, 5):
        line[f'kick{i}']            = 0.0
        line.ref[f'cor{i}'].ksl[0]  = line.ref[f'kick{i}']
    return line

################################################################################
# Measure each response column and omit the weak direction
################################################################################

########################################
# Measure the signed response
########################################
# A positive Multipole ksl[0] adds to py. The reference orbit is zero.
knobs       = [f'kick{i}' for i in range(1, 5)]
bpms        = [f'bpm{i}' for i in range(1, 5)]
response    = np.empty((4, 4))
for column, knob in enumerate(knobs):
    means = []
    for sign in (-1, 1):
        line        = build_distributed_ring(PROBE_TURNS)
        line[knob]  = sign * PROBE
        line.track(line.particle_ref.copy(), num_turns = PROBE_TURNS)
        means.append([
            line.get(bpm).mean_y[SETTLE_TURNS:].mean() for bpm in bpms
        ])
    minus, plus         = np.array(means)
    response[:, column] = (plus - minus) / (2 * PROBE)

########################################
# Invert the retained directions
########################################
u, singular, vt = np.linalg.svd(response)
keep            = singular > SVD_CUTOFF * singular[0]
inverse         = (vt[keep].T / singular[keep]) @ u[:, keep].T

################################################################################
# Apply the same independent kicks with feedback off and on
################################################################################

########################################
# Generate the independent disturbances
########################################
num_turns   = round(DURATION * FREV)
turns       = np.arange(num_turns)
kick_time   = turns / FREV
waveform    = AMPLITUDES * np.sin(
    2 * np.pi * FREQUENCY * kick_time[:, None] + PHASES
)
waveform += (kick_time[:, None] >= STEP_TIME) * STEP_KICKS
orbits    = {}
for mode in ('Off', 'On'):
    ########################################
    # Bind the BPMs and correctors
    ########################################
    line    = build_distributed_ring(num_turns, waveform)
    fb      = xt.feedback.Feedback(
        name                        = 'vertical_orbit',
        active                      = mode == 'On',
        history_length              = WINDOW_TURNS,
        update_turns                = UPDATE_TURNS,
        first_calculation_turn      = UPDATE_TURNS,
        processing_delay_seconds    = PROCESSING_DELAY,
        command_delay_seconds       = COMMAND_DELAY,
    )
    for i, bpm in enumerate(bpms, 1):
        fb.inputs.new(
            name = f'y{i}', monitor = bpm, quantity = 'mean_y', unit = 'm',
        )

    # A common slew scale preserves the retained corrector subspace.
    actuator = xt.feedback.PhysicalActuator(
        target                          = knobs,
        lower                           = -KICK_LIMIT,
        upper                           = KICK_LIMIT,
        response_time_constant_seconds  = RESPONSE_TIME,
        maximum_slew_per_second         = SLEW,
        slew_mode                       = 'coordinated',
    )
    fb.outputs.new(name = 'kicks', actuator = actuator, unit = 'rad')
    fb.parameters.new(name = 'inverse', value = inverse, unit = 'rad/m')
    fb.parameters.new(name = 'kp', value = KP)
    fb.parameters.new(name = 'ki', value = KI, unit = '1/s')
    fb.state.new(name = 'integral', initial_value = np.zeros(4), unit = 'rad')

    ########################################
    # Calculate the retained correction
    ########################################
    # The single probe particle has the same weight on every passage.
    fb.program = '''
measured = stack((mean(acquired.y1_history), mean(acquired.y2_history),
                  mean(acquired.y3_history), mean(acquired.y4_history)))
correction = -(parameters.inverse @ measured)
state.integral = state.integral + parameters.ki * event.dt_seconds * correction
outputs.kicks = parameters.kp * correction + state.integral
'''
    line.append('cofb', fb)
    line.build_tracker()
    line.feedback.enable()

    ########################################
    # Track the matched launch
    ########################################
    # Linear transport and common dipole kicks preserve centroid dynamics.
    particles = line.particle_ref.copy()
    # A frozen 1 mm lag places the particle inside each Exciter sample bin.
    particles.zeta[:] = -1e-3
    line.track(particles, num_turns = num_turns)
    orbits[mode] = np.column_stack([line.get(bpm).mean_y for bpm in bpms])

################################################################################
# Compare retained and omitted modes
################################################################################
# Project on the left singular vectors and fit sine, cosine and DC terms.
########################################
# Measure the residual harmonic motion
########################################
time        = (turns + 1) / FREV
late        = time >= STEP_TIME + 1.0
phase       = 2 * np.pi * FREQUENCY * time[late]
design      = np.column_stack((
    np.sin(phase), np.cos(phase), np.ones(phase.size),
))
results     = {mode: orbit @ u for mode, orbit in orbits.items()}
fits        = {
    mode: np.linalg.lstsq(design, orbit[late], rcond = None)[0]
    for mode, orbit in results.items()
}
amplitudes  = {mode: np.hypot(*fit[:2]) for mode, fit in fits.items()}

print('Vertical response modes\n-----------------------')
print('A mode pairs a pattern of corrector kicks with a pattern of BPM')
print('displacements. The response is the gain from one pattern to the other.')
print(f'The loop uses modes above {SVD_CUTOFF:.0%} of the strongest response.')
print()
print(f'Corrector 3 at s/C = {CORRECTOR_POSITIONS[2]:.4f}, '
      f'corrector 4 at s/C = {CORRECTOR_POSITIONS[3]:.4f}.')
print(f'Mode 4 kick weights on those correctors: {vt[-1, 2]:+.3f}, '
      f'{vt[-1, 3]:+.3f}.')
print('Opposite kicks at nearly the same phase move the BPMs very little.')

print('\nRejection of the changing disturbance\n'
      '-------------------------------------')
print(f'Disturbance frequency [Hz]: {FREQUENCY:g}')
print(f'Additional kick step at all four disturbance sites [s]: {STEP_TIME:g}')
print('The loop updates from live BPM readings throughout both disturbances.')
print()
print('Mode     Response [m/rad]   Used by loop    Harmonic amplitude [µm]')
print('                                                  Off            On')
for i, value in enumerate(singular):
    kept    = 'Yes' if keep[i] else 'No'
    off     = 1e6 * amplitudes['Off'][i]
    on      = 1e6 * amplitudes['On'][i]
    print(f'{i + 1:4d} {value:20.5g}   {kept:12s} {off:13.3f} {on:13.3f}')
weak_kick_norm  = amplitudes['Off'][-1] / singular[-1]
max_kick_norm   = np.sqrt(len(knobs)) * KICK_LIMIT
print(f'\nKick norm needed for mode 4 harmonic [µrad]: '
      f'{1e6 * weak_kick_norm:.0f}')
print(f'Maximum available kick norm [µrad]: {1e6 * max_kick_norm:.0f}')

########################################
# Plot the BPM and corrector weights
########################################
mode_fig, mode_axes = plt.subplots(
    1, 2, figsize = (9, 4), layout = 'constrained',
)
mode_indices        = np.arange(len(singular))
for ax, weights, devices, title in [
    (mode_axes[0], vt, [f'cor{i}' for i in range(1, 5)], 'Corrector pattern'),
    (mode_axes[1], u.T, bpms, 'BPM pattern'),
]:
    colors = ax.imshow(weights, cmap = 'RdBu_r', vmin = -1, vmax = 1)
    ax.set(
        title   = title,
        xticks  = np.arange(len(devices)), xticklabels = devices,
        yticks = mode_indices, yticklabels = mode_indices + 1,
    )
    for (row, column), weight in np.ndenumerate(weights):
        color = 'white' if abs(weight) > 0.58 else 'black'
        ax.text(column, row, f'{weight:+.2f}', ha = 'center', va = 'center',
                color = color)
mode_axes[0].set_ylabel('Mode')
mode_fig.colorbar(colors, ax = mode_axes, label = 'Normalized weight')

########################################
# Plot the measured mode amplitudes
########################################
fig, axes = plt.subplots(2, 2, sharex = True, figsize = (9, 6))
for i, ax in enumerate(axes.flat):
    ax.plot(time, 1e6 * results['Off'][:, i], label = 'Feedback off')
    ax.plot(time, 1e6 * results['On'][:, i], '--', label = 'Feedback on')
    ax.axvline(STEP_TIME, color = '0.45', ls = ':', label = 'Disturbance step')
    ax.set_title(f'Mode {i + 1} · {"retained" if keep[i] else "omitted"}')
for ax in axes[:, 0]:
    ax.set_ylabel('BPM projection [µm]')
for ax in axes[-1]:
    ax.set_xlabel('Time [s]')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc = 'upper center', ncols = 3, frameon = False)
fig.tight_layout(rect = (0, 0, 1, 0.93))
plt.show()
