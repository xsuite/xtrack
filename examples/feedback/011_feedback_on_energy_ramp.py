# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Compare transverse feedback at two working points during a momentum ramp.
# A fixed delay in seconds adds a turn as the beam speeds up.

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

import xtrack as xt

################################################################################
# User parameters
################################################################################

########################################
# Proton ring and reference momentum ramp
########################################
CIRCUMFERENCE   = 120.0                 # Ring circumference [m]
P0C             = 0.6e9                 # Initial momentum times c [eV]
DP0C_DT         = 1.2e13                # Rate of change of p0c [eV/s]
RAMP_END        = 150e-6                # End of the momentum table [s]
N_TURNS         = 120                   # Complete traversals per run
QY_VALUES       = (0.14, 0.25)          # Vertical tunes
BETY            = 12.0                  # Vertical beta at pickup [m]

########################################
# Bunch launch and feedback
########################################
Y_OFFSET        = 1e-3                  # Initial pickup displacement [m]
GAIN            = 0.022                 # Integrated field per offset [T m/m]
COMMAND_DELAY   = 0.52e-6               # Command transport delay [s]
RMS_TURNS       = 20                    # Turns in the final RMS window

################################################################################
# Build the accelerating ring and vertical feedback
################################################################################

########################################
# Program the reference momentum and corrector rigidity
########################################
reference           = xt.Particles(p0c = P0C, mass0 = xt.PROTON_MASS_EV)
env                 = xt.Environment(particle_ref = reference)
env['trim_tm']      = 0.0
env['p0c_ramp']     = P0C + DP0C_DT * env.ref['t_turn_s']
# For a proton, B rho [T m] = p0c [eV] / c [m/s].
env.new(
    'corrector', xt.Multipole,
    ksl = [c * env.ref['trim_tm'] / env.ref['p0c_ramp']],
)
env.elements['bpm'] = xt.BeamStatsMonitor(
    stop_at_turn    = N_TURNS,
    stats           = ['mean_y'],
)
env.elements['arc'] = xt.LineSegmentMap(
    length              = CIRCUMFERENCE,
    qy                  = QY_VALUES[0],
    bety                = BETY,
    longitudinal_mode   = 'frozen',
)

########################################
# Add the ideal accelerating kick
########################################
# For a linear p0c ramp, dE/dN = dp0c/dt * circumference / c.
# A zero-frequency Cavity at phase pi/2 supplies this constant energy gain.
energy_step = DP0C_DT * CIRCUMFERENCE / c
env.new(
    'accelerator', xt.Cavity,
    voltage     = energy_step,
    frequency   = 0.0,
    phase       = np.pi / 2,
)

########################################
# Bind the BPM and corrector to feedback
########################################
fb = xt.feedback.Feedback(
    name            = 'vertical_damper',
    update_turns    = 1,
    log_capacity    = N_TURNS,
)
fb.inputs.new(name = 'y', monitor = 'bpm', quantity = 'mean_y', unit = 'm')
fb.parameters.new(name = 'gain', value = GAIN, unit = 'T m/m')
fb.outputs.new(name = 'trim', target = 'trim_tm', unit = 'T m')
# Positive ksl[0] kicks the reference proton upwards. The phase advance
# from pickup to applied kick determines whether the oscillation damps.
fb.program = 'outputs.trim = parameters.gain * acquired.y'
env.elements['feedback'] = fb

########################################
# Build the line and attach the reference ramp
########################################
line = env.new_line(components = [
    'bpm', 'corrector', 'arc', 'accelerator', 'feedback',
])
line.energy_program = xt.EnergyProgram(
    t_s = [0.0, RAMP_END],
    p0c = [P0C, P0C + DP0C_DT * RAMP_END],
)
line.enable_time_dependent_vars = True
line.build_tracker()

################################################################################
# Compare the command-delivery settings at both working points
################################################################################
# Copy the initial bunch, including its injection reference momentum.
initial_particles = line.build_particles(y = Y_OFFSET)
cases = (
    ('Feedback off', False, 0.0),
    ('Immediate delivery', True, 0.0),
    (f'{1e6 * COMMAND_DELAY:g} µs command delay', True, COMMAND_DELAY),
)
results = {}
for qy in QY_VALUES:
    results[qy] = {}
    for label, active, delay in cases:
        line.feedback.reset()
        line['arc'].qy = qy
        fb.command_delay_seconds = delay
        line.feedback.enabled = active
        particles = initial_particles.copy()
        line.track(particles, num_turns = N_TURNS)
        results[qy][label] = line['bpm'].mean_y.copy()

################################################################################
# Relate the physical delay to the changing turn duration
################################################################################

########################################
# Determine the delivery lag in turns
########################################
turns       = np.arange(N_TURNS + 1)
time_s      = np.r_[0.0, [row['time_seconds'] for row in fb.log]]
period_us   = 1e6 * np.diff(time_s)
momentum    = line.energy_program.get_p0c_at_t_s(time_s)

# A request at boundary n can first be delivered at a later full-turn boundary.
delivery    = np.searchsorted(time_s, time_s[1:] + COMMAND_DELAY)
inside      = delivery <= N_TURNS
calculated  = turns[1:][inside]
lag_turns   = delivery[inside] - calculated

########################################
# Report the ramp and final oscillation amplitude
########################################
title = 'Vertical bunch feedback on a proton energy ramp'
print(f'\n{title}\n{"-" * len(title)}')
print(f'Reference momentum [GeV/c]: {momentum[0] / 1e9:.3f}'
      f' to {momentum[-1] / 1e9:.3f}')
print(f'Turn duration [µs]: {period_us[0]:.3f} to {period_us[-1]:.3f}')
if lag_turns.size:
    print(f'Request-to-delivery lag [turns]: {lag_turns[0]} to {lag_turns[-1]}')
else:
    print('The run ends before the first delayed command is delivered.')
title = f'Final {RMS_TURNS} turns: BPM RMS [µm]'
print(f'\n{title}\n{"-" * len(title)}')
print(f'{"Case":28s}' + ''.join(f'{f"Qy = {qy:g}":>16s}' for qy in QY_VALUES))
for label, _, _ in cases:
    rms = [np.sqrt(np.mean(results[qy][label][-RMS_TURNS:]**2))
           for qy in QY_VALUES]
    print(f'{label:28s}' + ''.join(f'{1e6 * value:16.2f}' for value in rms))
print('\nThe off case includes adiabatic damping during acceleration.')
print('The immediate kick uses the preceding pickup.')
print('The command delay adds turns to the pickup-to-kick lag.')

################################################################################
# Plot the bunch oscillation and time to the next boundary
################################################################################
fig, axes = plt.subplots(
    len(QY_VALUES) + 1, 1, sharex = True,
    figsize = (8, 7.8), layout = 'constrained',
)
for ax, (qy, case_results) in zip(axes[:-1], results.items()):
    for label, bpm_y in case_results.items():
        ax.plot(turns[:-1], 1e3 * bpm_y, label = label)
    ax.set(title = fr'Vertical tune $Q_y = {qy:g}$', ylabel = 'BPM [mm]')
fig.legend(*axes[0].get_legend_handles_labels(), loc = 'outside upper center',
           ncols = 3)

axes[-1].plot(turns[:-1], period_us, label = 'Time to next turn boundary')
axes[-1].axhline(1e6 * COMMAND_DELAY, color = 'tab:red', ls = '--',
                 label = 'Command delay')
axes[-1].set(xlabel = 'Completed reference turns', ylabel = 'Interval [µs]')
axes[-1].legend()
changes = np.flatnonzero(np.diff(lag_turns)) + 1
if changes.size:
    first_change = calculated[changes[0]]
    for ax in axes:
        ax.axvline(first_change, color = '0.45', ls = ':')
    axes[-1].annotate(
        f'Request delivered after {lag_turns[changes[0]]} turns',
        xy = (first_change, period_us[first_change]),
        xytext = (7, 22), textcoords = 'offset points',
        arrowprops = {'arrowstyle': '->', 'color': '0.45'},
    )
plt.show()
