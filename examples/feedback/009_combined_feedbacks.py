# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Control the horizontal orbit, horizontal tune and vertical head–tail
# motion of the same bunch during one line.track call.

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
# Two-particle ring
########################################
P0C         = 7e12                      # Proton momentum times c [eV]
N_TURNS     = 256
QX_BARE     = 0.28                      # Horizontal tune before correction
QY          = 0.25                      # Phase for the head–tail correction
QS          = 1 / 40                    # Synchrotron tune
BETX        = 10.0                      # Horizontal beta at the pickup [m]
BETY        = 10.0                      # Vertical beta at the pickup [m]
BETS        = 120.0                     # Synchrotron beta [m]

########################################
# Launch and perturbations
########################################
X_ERROR     = 2e-6                      # Fixed horizontal dipole kick [rad]
X_PROBE     = 10e-6                     # Horizontal oscillation amplitude [m]
Y_OFFSET    = 0.1e-6                    # Initial head/tail offset magnitude [m]
ZETA_MAX    = 0.10                      # Synchrotron amplitude [m]
WAKE_SLOPE  = 0.025                     # Tail kick per head offset [rad/m]

########################################
# Orbit and tune loops
########################################
QX_TARGET   = 0.31                      # Horizontal tune setpoint
ORBIT_TURNS = 32                        # Turns per orbit average and correction
TUNE_TURNS  = 64                        # Turns between tune corrections
TUNE_WINDOW = 24                        # BPM samples in each tune fit
ORBIT_GAIN  = 0.5                       # Fraction of orbit error to correct
TUNE_GAIN   = 0.8                       # Fraction of tune error to correct
MIN_SIGNAL  = 0.1e-6                    # Minimum RMS BPM difference for fit [m]

########################################
# Vertical feedback
########################################
COMMON_GAIN = 0.015                     # Uniform vertical kick gain [rad/m]
SLICE_GAIN  = 0.015                     # Odd vertical kick gain [rad/m]

################################################################################
# Define the head-to-tail wake
################################################################################
class TwoParticleWake:
    """Apply a short-range dipole wake from the leading charge to the tail."""

    iscollective = True

    def copy(self, **kwargs):
        """Return a new wake for an independent line.

        Parameters
        ----------
        **kwargs
            Context and buffer options from Line.copy. The wake has no arrays.
        """
        return type(self)()

    def track(self, particles):
        """Kick the trailing macroparticle at this passage.

        Parameters
        ----------
        particles : xtrack.Particles
            Two equal charges at opposite synchrotron phases.
        """
        xp      = particles._context.nplike_lib
        head    = particles.zeta > 0
        head_y  = xp.sum(xp.where(head, particles.y, 0.0))
        particles.py[:] += xp.where(head, 0.0, WAKE_SLOPE * head_y)


################################################################################
# Build the ring, pickup and actuators
################################################################################
if (
    not 4 <= TUNE_WINDOW <= ORBIT_TURNS <= TUNE_TURNS
    or TUNE_TURNS % ORBIT_TURNS
):
    raise ValueError('Fit the tune inside an orbit interval before a trim')

########################################
# Compute the orbit and RF settings
########################################
reference   = xt.Particles(p0c = P0C)
k_rf        = np.pi / (2 * ZETA_MAX)
frequency   = k_rf * float(reference.beta0[0]) * c / (2 * np.pi)

# For the pickup upstream of the kick, the closed orbit is
# x = beta_x * kick / (2 * tan(pi * Qx)) and px = -kick / 2.
x_closed            = BETX * X_ERROR / (2 * np.tan(np.pi * QX_BARE))
orbit_response      = BETX / (2 * np.tan(np.pi * QX_TARGET))
longitudinal_phase  = 2 * np.pi * QS / 4
side                = np.array([1.0, -1.0])

########################################
# Define the actuator knobs
########################################
env = xt.Environment(particle_ref = reference)

env['x_error']      = X_ERROR
env['orbit_trim']   = 0.0
env['qx_bare']      = QX_BARE
env['tune_trim']    = 0.0
env['common_kick']  = 0.0
env['slice_kick']   = 0.0

########################################
# Place the pickup, wake and kickers
########################################
env.elements['pickup']  = xt.BeamStatsMonitor(
    stop_at_turn    = N_TURNS,
    zeta_range      = (-1.1 * ZETA_MAX, 1.1 * ZETA_MAX),
    num_slices      = 2,
    stats           = ['mean_x', 'mean_y', 'mean_py', 'mean_zeta'],
)
env.elements['wake']    = TwoParticleWake()
# A negative knl[0] gives a positive horizontal kick.
env.new('orbit_corrector', xt.Multipole, knl = ['-x_error - orbit_trim'])
env.new('common_corrector', xt.Multipole, ksl = ['common_kick'])
env.new(
    'slice_corrector', xt.RFMultipole,
    frequency = frequency, ksl = ['slice_kick'], phase_s = [np.pi / 2],
)

########################################
# Bind the horizontal tune trim
########################################
env.elements['ring']    = xt.LineSegmentMap(
    length = 26000.0, betx = BETX, bety = BETY, qs = QS, bets = BETS,
    qx     = QX_BARE, qy = QY,
)
env.ref['ring'].qx      = env.ref['qx_bare'] + env.ref['tune_trim']

################################################################################
# Configure the three feedbacks
################################################################################

########################################
# Correct the horizontal orbit
########################################
orbit = xt.feedback.Feedback(
    name            = 'orbit',
    history_length  = ORBIT_TURNS,
    update_turns    = ORBIT_TURNS,
    log_capacity    = N_TURNS,
)
orbit.inputs.new(
    name = 'x', monitor = 'pickup', quantity = 'mean_x', bunch_slot = 0,
    unit = 'm',
)
orbit.parameters.new(
    name = 'gain', value = ORBIT_GAIN / orbit_response, unit = 'rad/m',
)
orbit.state.new(name = 'trim', initial_value = 0.0, unit = 'rad')
orbit.outputs.new(name = 'trim', target = 'orbit_trim', unit = 'rad')
orbit.program = '''
error = mean(acquired.x_history)
state.trim = state.trim - parameters.gain * error
outputs.trim = state.trim
'''

########################################
# Measure and correct the tune
########################################
# Over the fit window the horizontal kick and tune are held. Successive
# BPM differences remove the closed orbit, leaving the recurrence
# d[n+1] + d[n-1] = 2 cos(2 pi Qx) d[n].
tune = xt.feedback.Feedback(
    name            = 'tune',
    history_length  = TUNE_WINDOW,
    update_turns    = TUNE_TURNS,
    log_capacity    = N_TURNS,
)
tune.inputs.new(
    name = 'x', monitor = 'pickup', quantity = 'mean_x', bunch_slot = 0,
    unit = 'm',
)
tune.parameters.new(name = 'target', value = np.cos(2 * np.pi * QX_TARGET))
cosine_slope = -2 * np.pi * np.sin(2 * np.pi * QX_TARGET)
tune.parameters.new(
    name = 'gain', value = TUNE_GAIN / cosine_slope,
)
tune.parameters.new(name = 'min_power', value = MIN_SIGNAL**2)
tune.state.new(name = 'trim', initial_value = 0.0)
tune.outputs.new(name = 'trim', target = 'tune_trim')
tune.program = '''
d = diff(mean(acquired.x_history, axis=1))
power = mean(d[1:-1] ** 2)
projection = mean(d[1:-1] * (d[:-2] + d[2:]))
cosine = projection / (2 * maximum(power, parameters.min_power))
trim = state.trim - parameters.gain * (cosine - parameters.target)
state.trim = where(power > parameters.min_power, trim, state.trim)
outputs.trim = state.trim
'''

########################################
# Correct centroid and slice motion
########################################
head_tail = xt.feedback.Feedback(
    name = 'head_tail', update_turns = 1,
)
for coordinate in ('y', 'zeta'):
    head_tail.inputs.new(
        name     = coordinate, monitor = 'pickup',
        quantity = f'mean_{coordinate}', bunch_slot = 0, unit = 'm',
    )
head_tail.parameters.new(
    name = 'common', value = COMMON_GAIN, unit = 'rad/m',
)
head_tail.parameters.new(name = 'odd', value = SLICE_GAIN, unit = 'rad/m')
head_tail.parameters.new(name = 'k_rf', value = k_rf, unit = '1/m')
head_tail.outputs.new(
    name = 'kick', target = ['common_kick', 'slice_kick'], unit = 'rad',
)
head_tail.program = '''
common = parameters.common * mean(acquired.y)
odd = parameters.odd * mean(acquired.y * sin(parameters.k_rf * acquired.zeta))
outputs.kick = stack((common, odd))
'''

################################################################################
# Place the feedbacks and prepare the launch
################################################################################

########################################
# Place the feedbacks after the ring
########################################
for fb in (orbit, tune, head_tail):
    env.elements[fb.name] = fb
line = env.new_line(components = [
    'pickup', 'wake', 'orbit_corrector', 'common_corrector',
    'slice_corrector', 'ring', orbit.name, tune.name, head_tail.name,
])

########################################
# Launch just off the horizontal orbit
########################################
particles = xt.Particles(
    p0c     = P0C,
    x       = x_closed + X_PROBE,
    px      = -X_ERROR / 2,
    y       = side * Y_OFFSET,
    zeta    = side * ZETA_MAX * np.cos(longitudinal_phase),
    pzeta   = side * ZETA_MAX * np.sin(longitudinal_phase) / BETS,
)

################################################################################
# Track two independent runs
################################################################################

########################################
# Inspect the feedback table
########################################
print('Feedbacks on the line\n---------------------')
feedback_table = line.feedback.get_table()
print(feedback_table.show(
    cols    = ['name', 'update_turns', 'enabled'],
    output  = str,
))
print()
print(f'Tune fits use {TUNE_WINDOW} BPM samples while both horizontal '
      'settings are held.')
print(f'Horizontal tune setpoint: {QX_TARGET:.5f}')

########################################
# Track with all feedbacks off
########################################
# A prepared line keeps its pickup records and binds to one Particles object.
# Copy the line before tracking so both runs start with fresh records and state.
line_off        = line.copy()
particles_off   = particles.copy()

line_off.feedback.enabled = False
line_off.track(particles_off, num_turns = N_TURNS)

########################################
# Track with all feedbacks on
########################################
line.feedback.enabled = True
line.track(particles, num_turns = N_TURNS)

################################################################################
# Compare the measured orbit, tune and head–tail motion
################################################################################

########################################
# Extract the BPM records and updates
########################################
results = {}
for label, tracked_line in [
    ('Feedback off', line_off),
    ('All three feedbacks', line),
]:
    pickup      = tracked_line.get('pickup')
    x           = pickup.get('mean_x', slot = 0).mean(axis = 1)
    y           = pickup.get('mean_y', slot = 0)
    py          = pickup.get('mean_py', slot = 0)
    amplitude   = 0.5 * np.hypot(
        y[:, 1] - y[:, 0], BETY * (py[:, 1] - py[:, 0]),
    )

    feedback        = tracked_line.feedback
    updates         = {
        name: [
            row['completed_turn'] for row in feedback[name].log
            if row['executed']
        ]
        for name in ('orbit', 'tune', 'head_tail')
    }
    trim            = np.array([
        row['applied_output'][0] for row in feedback['tune'].log
    ])
    results[label]  = dict(
        x = x, head_tail = amplitude, updates = updates, tune_trim = trim,
    )

########################################
# Compare the three controlled quantities
########################################
orbit_turns = np.arange(ORBIT_TURNS, N_TURNS + 1, ORBIT_TURNS)
tune_turns  = np.arange(TUNE_TURNS, N_TURNS + 1, TUNE_TURNS)
period      = round(1 / QS)
fit_turns   = np.arange(2 * period, N_TURNS, period)

print('\nMeasured effect of feedback\n---------------------------')
print('Case                 Orbit mean [µm]   Measured Qx'
      '   Head–tail rate [1/turn]')
fig, axes = plt.subplots(3, 1, sharex = True, figsize = (8, 7))
for label, result in results.items():
    x           = result['x']
    orbit_mean  = np.array([
        x[turn - ORBIT_TURNS : turn].mean() for turn in orbit_turns
    ])
    measured_qx = []
    for turn in tune_turns:
        d       = np.diff(x[turn - TUNE_WINDOW : turn])
        cosine  = d[1:-1] @ (d[:-2] + d[2:]) / (2 * (d[1:-1] @ d[1:-1]))
        measured_qx.append(np.arccos(np.clip(cosine, -1, 1)) / (2 * np.pi))

    amplitude   = result['head_tail']
    rate        = np.polyfit(fit_turns, np.log(amplitude[fit_turns]), 1)[0]
    result.update(orbit_mean = orbit_mean, measured_qx = measured_qx,
                  rate = rate)
    print(f'{label:19s} {orbit_mean[-1] * 1e6:+17.3f} {measured_qx[-1]:13.5f} '
          f'{rate:+23.5f}')

    axes[0].plot(orbit_turns, orbit_mean * 1e6, '.-', label = label)
    axes[1].plot(tune_turns, measured_qx, '.-')
    axes[2].semilogy(np.arange(1, N_TURNS + 1), amplitude * 1e6)

axes[0].set_ylabel('Orbit mean [µm]')
axes[1].set_ylabel('Measured $Q_x$')
axes[1].axhline(QX_TARGET, color = '0.4', ls = ':', label = 'Tune setpoint')
axes[1].legend(loc = 'center left')
axes[2].set(xlabel = 'Completed turn', ylabel = 'Head–tail amplitude [µm]')
fig.legend(
    *axes[0].get_legend_handles_labels(), loc = 'upper center', ncols = 2,
)
fig.tight_layout(rect = (0, 0, 1, 0.94))
plt.show()
