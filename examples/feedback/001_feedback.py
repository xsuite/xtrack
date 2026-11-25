# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Use a BPM and a corrector to compensate a persistent vertical kick.
# Compare the same launch and disturbance with feedback off and on.

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Ring and independent kick
########################################
P0C             = 45.6e9                # Fixed electron momentum times c [eV]
CIRCUMFERENCE   = 1000.0                # [m]
BETA_Y          = 10.0                  # [m]
QY              = 0.25                  # Vertical tune
DAMPING         = 0.1                   # Fractional amplitude loss per turn
DISTURBANCE     = 1e-6                  # Vertical kick on every passage [rad]

########################################
# Controller
########################################
N_TURNS         = 600
FIRST_UPDATE    = 100                   # Let the uncorrected orbit settle
UPDATE_TURNS    = 20                    # Turns between corrections
GAIN            = 0.05                  # Kick increment per BPM error [rad/m]

################################################################################
# Build the ring and wire a corrector knob
################################################################################

########################################
# Set the disturbance and corrector
########################################
# The thin Multipole gives the same kick every time the particle passes it,
# starting on the first turn. Positive ksl[0] raises py. The colocated
# corrector can cancel that kick at the same point in the ring.
env = xt.Environment()

env['correction_rad'] = 0.0
env.new('disturbance', xt.Multipole, ksl = [DISTURBANCE])
env.new('corrector', xt.Multipole, ksl = [0.0])
env.ref['corrector'].ksl[0] = env.ref['correction_rad']

env.elements['arc'] = xt.LineSegmentMap(
    length          = CIRCUMFERENCE,
    qy              = QY,
    bety            = BETA_Y,
    damping_rate_y  = DAMPING,
    damping_rate_py = DAMPING,
)

########################################
# Record the vertical orbit
########################################
env.elements['bpm'] = xt.BeamStatsMonitor(
    stop_at_turn    = N_TURNS,
    stats           = ['mean_y'],
)

line = env.new_line(components = ['disturbance', 'corrector', 'arc', 'bpm'])
line.set_particle_ref('electron', p0c = P0C)

################################################################################
# Track the displaced orbit with the corrector held at zero
################################################################################
# Both cases start at y = py = 0. Damping removes the launch oscillation
# about the displaced closed orbit under the constant kick.
line_off        = line.copy()
particles_off   = line_off.particle_ref.copy()
line_off.build_tracker()
line_off.track(particles_off, num_turns = N_TURNS)
y_off = line_off.get('bpm').mean_y

################################################################################
# Connect the BPM and the corrector to the feedback
################################################################################

########################################
# Bind the input, output and state
########################################
fb = xt.feedback.Feedback(
    name                    = 'orbit',
    update_turns            = UPDATE_TURNS,
    first_calculation_turn  = FIRST_UPDATE,
)
fb.inputs.new(name = 'y', monitor = 'bpm', quantity = 'mean_y', unit = 'm')
fb.outputs.new(name = 'kick', target = 'correction_rad', unit = 'rad')
fb.parameters.new(name = 'gain', value = GAIN, unit = 'rad/m')
fb.state.new(name = 'kick', initial_value = 0.0, unit = 'rad')

########################################
# Calculate the next held kick
########################################
# The damped ring has a positive static response at the BPM. A positive
# measured y therefore calls for a negative change in the held kick.
fb.program = '''
state.kick = state.kick - parameters.gain * acquired.y
outputs.kick = state.kick
'''

line.append('orbit_feedback', fb)
particles = line.particle_ref.copy()

################################################################################
# Track the same disturbance with feedback on
################################################################################
# The terminal feedback reads a completed BPM passage. The first new kick
# acts when the particle next encounters the corrector.
line.build_tracker()
line.feedback.enable()
line.track(particles, num_turns = N_TURNS)

################################################################################
# Compare the measured orbits
################################################################################
y_on = line.get('bpm').mean_y

########################################
# Identify the first affected passage
########################################
print('Vertical disturbance and feedback timing\n'
      '----------------------------------------')
print(f'Disturbance kick per passage [µrad]: {1e6 * DISTURBANCE:+.3f}')
print('The same kick acts on every passage, starting on the first turn.')
print(f'First calculation [completed turn]: {FIRST_UPDATE}')
print(f'First traversal with a corrector kick: {FIRST_UPDATE + 1}')

########################################
# Compare the final BPM and kick
########################################
print('\nMatched orbit tracking\n----------------------')
print('Feedback    BPM before correction [µm]    Final BPM [µm]'
      '    Final corrector [µrad]')
for label, y, kick in [
    ('Off', y_off, line_off['correction_rad']),
    ('On', y_on, line['correction_rad']),
]:
    print(f'{label:<8} {1e6 * y[FIRST_UPDATE - 1]:30.3f}'
          f' {1e6 * y[-1]:17.3f} {1e6 * kick:+25.3f}')
print('Damping alone leaves a displaced orbit. With feedback, the corrector')
print('opposes the fixed kick and brings the orbit back towards zero.')

########################################
# Plot the two BPM traces
########################################
turns   = np.arange(1, N_TURNS + 1)
fig, ax = plt.subplots(figsize = (8, 4))
ax.plot(turns, y_off * 1e6, '--', color = '0.35', label = 'Feedback off')
ax.plot(
    turns, y_on * 1e6, color = 'tab:blue', label = 'Feedback on', zorder = 1,
)
ax.axvline(FIRST_UPDATE, ls = ':', color = '0.3', label = 'First calculation')
ax.set(xlabel = 'Completed turn', ylabel = 'BPM y [µm]')
ax.legend()
fig.tight_layout()
plt.show()
