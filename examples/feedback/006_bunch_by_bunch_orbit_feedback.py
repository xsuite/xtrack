# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Compare one train-wide orbit kick with one kick per bunch.
# A sparse fill shows why waveform samples follow physical slot numbers.

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
# Fixed-energy electron ring
########################################
P0C             = 45.6e9                # Reference momentum times c [eV]
FREV            = 3300.0                # Revolution frequency [Hz]
QY              = 0.31                  # Vertical tune
BETA_Y          = 10.0                  # Vertical beta function [m]
DAMPING         = 0.02                  # Fractional amplitude loss per turn
N_TURNS         = 3300                  # Turns per comparison

########################################
# Filling pattern and independent kicks
########################################
FILL            = [1, 0, 1, 0, 0, 1, 0, 0]
DISTURBANCES    = [1.0e-6, -2.0e-6, 0.5e-6]     # Kicks at filled slots [rad]
N_PER_BUNCH     = 16                            # Macroparticles per bunch

########################################
# Controller and kicker
########################################
UPDATE_TURNS    = 8                     # Measurements per correction
INTEGRAL_GAIN   = 0.05                  # Fraction of inferred correction
DELAY_TURNS     = 1                     # Command transport delay [turns]
RESPONSE_TURNS  = 2                     # Kicker response time [turns]

################################################################################
# Place bunches on the physical arrival-time grid
################################################################################

########################################
# Convert slot spacing to zeta
########################################
slots           = np.flatnonzero(FILL)
n_slots         = len(FILL)
reference       = xt.Particles(mass0 = xt.ELECTRON_MASS_EV, q0 = -1, p0c = P0C)
length          = float(reference.beta0[0]) * c / FREV
slot_spacing    = length / n_slots

########################################
# Sample the occupied slots
########################################
# Negative zeta arrives later. Start one quarter into each occupied sample
# so the launch is away from slot boundaries. The map holds zeta fixed.
rng     = np.random.default_rng(2026)
offsets = rng.normal(size = (len(slots), N_PER_BUNCH, 2))
offsets -= offsets.mean(axis = 1, keepdims = True)
initial = xt.Particles(
    mass0   = xt.ELECTRON_MASS_EV,
    q0      = -1,
    p0c     = P0C,
    y       = offsets[..., 0].ravel() * 1e-6,
    py      = offsets[..., 1].ravel() * 1e-6 / BETA_Y,
    zeta    = np.repeat(-(slots + 0.25) * slot_spacing, N_PER_BUNCH),
    weight  = np.full(len(slots) * N_PER_BUNCH, 1e10 / N_PER_BUNCH),
)

disturbance_waveform        = np.zeros(n_slots)
disturbance_waveform[slots] = DISTURBANCES

################################################################################
# Calculate the signed response at the cycle origin
################################################################################
# A positive Exciter sample adds to py when ksl[0] is +1. For the BPM
# upstream of the kick, the periodic response is (I-M)^-1 M e_py.
phase       = 2 * np.pi * QY
matrix      = (1 - DAMPING) * np.array([
    [np.cos(phase), BETA_Y * np.sin(phase)],
    [-np.sin(phase) / BETA_Y, np.cos(phase)],
])
response    = np.linalg.solve(np.eye(2) - matrix, matrix[:, 1])[0]

################################################################################
# Track the same train without, with common and with bunch-by-bunch feedback
################################################################################
runs = {}
for mode in ('off', 'common', 'bunch'):
    ########################################
    # Place the pickups and kickers
    ########################################
    pickup          = xt.BeamStatsMonitor(
        stop_at_turn        = N_TURNS,
        filling_pattern     = FILL,
        selected_slots      = slots,
        bunch_spacing_zeta  = slot_spacing,
        stats               = ['num_particles', 'mean_y', 'mean_py'],
    )
    after_corrector = xt.BeamStatsMonitor(
        stop_at_turn        = N_TURNS,
        filling_pattern     = FILL,
        selected_slots      = slots,
        bunch_spacing_zeta  = slot_spacing,
        stats               = ['mean_py'],
    )
    corrector       = xt.Exciter(
        nsamples            = n_slots,
        ksl                 = [1.0],
        frev                = FREV,
        sampling_frequency  = n_slots * FREV,
        duration            = (N_TURNS + 1) / FREV,
    )
    disturbance     = xt.Exciter(
        samples             = disturbance_waveform,
        ksl                 = [1.0],
        frev                = FREV,
        sampling_frequency  = n_slots * FREV,
        duration            = (N_TURNS + 1) / FREV,
    )
    ring            = xt.LineSegmentMap(
        length          = length,
        qy              = QY,
        bety            = BETA_Y,
        damping_rate_y  = DAMPING,
        damping_rate_py = DAMPING,
    )

    ########################################
    # Set up the three feedback cases
    ########################################
    fb          = xt.feedback.Feedback(
        name                    = mode,
        active                  = mode != 'off',
        history_length          = UPDATE_TURNS,
        update_turns            = UPDATE_TURNS,
        command_delay_seconds   = DELAY_TURNS / FREV,
        log_capacity            = N_TURNS,
    )
    elements    = dict(
        pickup          = pickup,
        corrector       = corrector,
        after_corrector = after_corrector,
        disturbance     = disturbance,
        ring            = ring,
        feedback        = fb,
    )
    line        = xt.Line(
        elements        = elements,
        element_names   = list(elements),
        particle_ref    = reference,
    )

    ########################################
    # Bind the filled waveform slots
    ########################################
    # The BPM array is compact. The Exciter waveform includes the empty slots.
    targets = ['common_rad'] if mode == 'common' else [
        f'slot_{slot}_rad' for slot in slots
    ]
    for target in targets:
        line[target] = 0.0
    slot_targets = targets * len(slots) if mode == 'common' else targets
    for slot, target in zip(slots, slot_targets):
        line.ref['corrector'].samples[int(slot)] = line.ref[target]
    actuator = xt.feedback.PhysicalActuator(
        target                          = targets,
        response_time_constant_seconds  = RESPONSE_TURNS / FREV,
    )
    fb.outputs.new(name = 'kick', actuator = actuator, unit = 'rad')

    ########################################
    # Pool charge before normalizing the position
    ########################################
    fb.inputs.new(
        name = 'sum_y', monitor = 'pickup', quantity = 'sum_y', unit = 'm',
    )
    fb.inputs.new(
        name = 'weight', monitor = 'pickup', quantity = 'num_particles',
    )
    fb.parameters.new(
        name = 'gain', value = INTEGRAL_GAIN / response, unit = 'rad/m',
    )
    fb.state.new(
        name = 'kick', initial_value = np.zeros(len(targets)), unit = 'rad',
    )

    ########################################
    # Choose the correction resolution
    ########################################
    if mode == 'common':
        fb.program = '''
orbit = sum(acquired.sum_y_history) / sum(acquired.weight_history)
state.kick = state.kick - parameters.gain * orbit
outputs.kick = state.kick
'''
    else:
        fb.program = '''
weight = sum(acquired.weight_history, axis = 0)
orbit = sum(acquired.sum_y_history, axis = 0) / weight
state.kick = state.kick - parameters.gain * orbit
outputs.kick = state.kick
'''

    ########################################
    # Track the same train
    ########################################
    line.build_tracker()
    line.feedback.enable()
    particles = initial.copy()
    line.track(particles, num_turns = N_TURNS)
    runs[mode] = line, particles

################################################################################
# Compare the common orbit and the spread about the train centroid
################################################################################
print('Physical slots and feedback modes\n---------------------------------')
print(f'Occupied slots: {slots.tolist()}')
empty_slots = np.flatnonzero(np.asarray(FILL) == 0)
print(f'Empty slots: {empty_slots.tolist()}')
print('Feedback           Common RMS [µm]    Differential RMS [µm]')

labels      = {
    'off'   : 'Off',
    'common': 'Common kick',
    'bunch' : 'One kick per bunch',
}
late        = slice(N_TURNS // 2, None)
centroid    = {}
spread      = {}
for mode, (line, _) in runs.items():
    pickup          = line.get('pickup')
    centroid[mode]  = pickup.get('mean_y', level = 'beam')
    difference      = pickup.get('mean_y') - centroid[mode][:, None]
    spread[mode]    = np.sqrt(np.mean(difference**2, axis = 1))
    common_rms      = np.sqrt(np.mean(centroid[mode][late]**2))
    bunch_rms       = np.sqrt(np.mean(spread[mode][late]**2))
    print(f'{labels[mode]:<18} {common_rms * 1e6:15.4g} '
          f'{bunch_rms * 1e6:24.4g}')
print(f'RMS uses BPM records {late.start} to {N_TURNS - 1}.')
print('Both loops correct the train centroid. Separate kicks also reduce')
print('the orbit differences between bunches.')

################################################################################
# Measure the first kick at the particles
################################################################################
line            = runs['bunch'][0]
fb              = line.feedback['bunch']
request_turns   = (row['completed_turn'] for row in fb.log if row['executed'])
before          = line.get('pickup').get('mean_py')
after           = line.get('after_corrector').get('mean_py')
affected        = np.flatnonzero(
    np.any(np.abs(after - before) > 1e-15, axis = 1)
)
first_request   = next(request_turns, 'none')
first_kick      = affected[0] if affected.size else 'none'
print('\nMeasurement to correction\n-------------------------')
print(f'First calculation [completed turn]: {first_request}')
print(f'First affected kick [record turn]: {first_kick}')
print('Pickup record n is available after completed turn n + 1.')
print('The field starts changing one boundary after the target arrives.')

################################################################################
# Plot the train motion and final waveform on the physical slot axis
################################################################################
turns       = np.arange(N_TURNS)
fig, axes   = plt.subplots(3, 1, figsize = (8, 7))
for mode, (line, _) in runs.items():
    style   = '--' if mode == 'common' else '-'
    layer   = 3 if mode == 'common' else 2
    (artist,) = axes[0].plot(
        turns, centroid[mode] * 1e6, style,
        label = labels[mode], zorder = layer,
    )
    axes[1].plot(
        turns, spread[mode] * 1e6, style,
        color = artist.get_color(), zorder = layer,
    )
    if mode != 'off':
        waveform = line.get('corrector').samples * 1e6
        axes[2].step(
            np.arange(n_slots + 1), np.r_[waveform, waveform[-1]],
            where = 'post', ls = style, color = artist.get_color(),
            label = labels[mode],
        )

axes[0].set_ylabel('Train centroid [µm]')
axes[0].tick_params(labelbottom = False)
axes[0].legend()
axes[1].sharex(axes[0])
axes[1].set(xlabel = 'BPM record turn', ylabel = 'Bunch spread [µm]')
axes[2].set(
    xlabel      = 'Physical slot',
    ylabel      = 'Final kick [µrad]',
    xticks      = np.arange(n_slots) + 0.5,
    xticklabels = np.arange(n_slots),
    xlim        = (0, n_slots),
)
for slot in empty_slots:
    axes[2].axvspan(slot, slot + 1, color = '0.93', zorder = -1)
axes[2].legend()
fig.tight_layout()
plt.show()
