# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

# Resolve and damp head–tail motion using longitudinal slice centroids.
# Synchrotron exchange and a causal wake isolate an intrabunch mode.

################################################################################
# Required packages
################################################################################
import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt
from scipy.constants import c

################################################################################
# User parameters
################################################################################

########################################
# Ring and launch
########################################
P0C             = 7e12                  # Proton momentum times c [eV]
N_TURNS         = 240                   # Turns per comparison
QY              = 0.25                  # Vertical tune; sets feedback phase
QS              = 1 / 40                # Integer-turn synchrotron period
BETA_Y          = 10.0                  # Vertical beta function [m]
BETA_S          = 120.0                 # Synchrotron beta function [m]
Y_OFFSET        = 0.1e-6                # Initial head/tail offset magnitude [m]
ZETA_MAX        = 0.10                  # Synchrotron amplitude [m]

########################################
# Wake and feedback
########################################
WAKE_STRENGTH   = 0.25                  # β_y × dipole wake slope
COMMON_GAIN     = 0.15                  # β_y × centroid feedback gain
SLICE_GAIN      = 0.15                  # β_y × odd-profile feedback gain

################################################################################
# Two-particle ring and slice pickup
################################################################################
class TwoParticleWake:
    """Effective short-range dipole wake for two particles at opposite zeta."""

    iscollective = True

    def __init__(self, strength_per_m, context):
        """Set the dipole wake slope.

        Parameters
        ----------
        strength_per_m : float
            Tail kick per metre of vertical head offset [rad/m].
        context : xobjects.Context
            Context of the two particles traversing the wake.
        """
        self.strength_per_m = strength_per_m
        self.xp             = context.nplike_lib

    def track(self, particles):
        """Apply the head dipole wake only to the tail.

        Parameters
        ----------
        particles : xtrack.Particles
            Two equal charges at opposite longitudinal phases.
        """
        xp = self.xp
        # Positive zeta arrives first. Only the trailing charge receives a
        # kick.
        head        = particles.zeta > 0
        source_y    = xp.sum(xp.where(head, particles.y, 0.0))
        particles.py[:] += xp.where(
            head, 0.0, self.strength_per_m * source_y
        )


################################################################################
# Track a wake-off control and the four matched feedback cases
################################################################################

########################################
# Set the synchrotron phase
########################################
# Offset the launch phase to keep the slice exchange between sampled turns.
context         = xo.ContextCpu()
reference       = xt.Particles(p0c = P0C)
phase_s         = 2 * np.pi * QS / 4
k_rf            = np.pi / (2 * ZETA_MAX)
frequency_hz    = k_rf * float(reference.beta0[0]) * c / (2 * np.pi)
launch_zeta     = ZETA_MAX * np.cos(phase_s) * np.array([1.0, -1.0])
launch_pzeta    = ZETA_MAX * np.sin(phase_s) / BETA_S * np.array([1.0, -1.0])

if QY != 0.25 or QS <= 0:
    raise ValueError('Use QY = 1/4 and a positive synchrotron tune')
period = round(1 / QS)
if not np.isclose(period, 1 / QS) or N_TURNS < 4 * period:
    raise ValueError('Use at least four whole-turn synchrotron periods')
sample  = np.arange(0, N_TURNS, period)
sample  = sample[sample >= N_TURNS / 2 - period]

########################################
# Choose the wake and feedback cases
########################################
cases   = {
    # Label, wake slope times beta, common gain, odd-profile gain.
    'no_wake': ('Wake off', 0.0, 0.0, 0.0),
    'off': ('Feedback off', WAKE_STRENGTH, 0.0, 0.0),
    'common': ('Centroid feedback', WAKE_STRENGTH, COMMON_GAIN, 0.0),
    'slice_only': ('Slice feedback', WAKE_STRENGTH, 0.0, SLICE_GAIN),
    'slice': (
        'Centroid and slice feedback', WAKE_STRENGTH, COMMON_GAIN, SLICE_GAIN,
    ),
}
results = {}
for name, (_, wake, common_gain, slice_gain) in cases.items():
    ########################################
    # Place the wake and pickup
    ########################################
    env = xt.Environment()

    env.elements['pickup']  = xt.BeamStatsMonitor(
        start_at_turn   = 0,
        stop_at_turn    = N_TURNS,
        zeta_range      = (-1.1 * ZETA_MAX, 1.1 * ZETA_MAX),
        num_slices      = 2,
        stats           = ['num_particles', 'mean_y', 'mean_py', 'mean_zeta'],
    )
    env.elements['wake']    = TwoParticleWake(wake / BETA_Y, context)
    env.new('common_kicker', xt.Multipole, ksl = [0.0])
    env.new(
        'slice_kicker',
        xt.RFMultipole,
        frequency   = frequency_hz,
        ksl         = [0.0],
        phase_s     = [np.pi / 2],
    )
    env.elements['ring'] = xt.LineSegmentMap(
        length  = 26000.0,
        qx      = 0.31,
        qy      = QY,
        betx    = 10.0,
        bety    = BETA_Y,
        qs      = QS,
        bets    = BETA_S,
    )

    ########################################
    # Common and odd RF kick profiles
    ########################################
    for profile in ('common', 'slice'):
        env[f'{profile}_rad']               = 0.0
        env.ref[f'{profile}_kicker'].ksl[0] = env.ref[f'{profile}_rad']

    ########################################
    # Bind the pickup to both kick profiles
    ########################################
    fb = xt.feedback.Feedback(
        name            = 'head_tail',
        active          = bool(common_gain or slice_gain),
        update_turns    = 1,
        log_capacity    = N_TURNS,
    )
    fb.inputs.new(
        name        = 'y',
        monitor     = 'pickup',
        quantity    = 'mean_y',
        bunch_slot  = 0,
        unit        = 'm',
    )
    fb.inputs.new(
        name        = 'zeta',
        monitor     = 'pickup',
        quantity    = 'mean_zeta',
        bunch_slot  = 0,
        unit        = 'm',
    )
    fb.outputs.new(
        name = 'kick', target = ['common_rad', 'slice_rad'], unit = 'rad'
    )
    fb.parameters.new(
        name = 'common_gain', value = common_gain / BETA_Y, unit = 'rad/m'
    )
    fb.parameters.new(
        name = 'slice_gain', value = slice_gain / BETA_Y, unit = 'rad/m'
    )
    fb.parameters.new(name = 'k_rf', value = k_rf, unit = '1/m')

    ########################################
    # Calculate the two kick components
    ########################################
    # The pickup sees a command two turns later. The bare motion has by
    # then advanced by half a vertical period, so a positive gain damps.
    fb.program = '''
common = parameters.common_gain * mean(acquired.y)
profile = sin(parameters.k_rf * acquired.zeta)
odd = parameters.slice_gain * mean(acquired.y * profile)
outputs.kick = stack((common, odd))
'''

    ########################################
    # Track the matched two-particle launch
    ########################################
    env.elements['head_tail'] = fb

    line                = env.new_line(
        components = [
            'pickup',
            'wake',
            'common_kicker',
            'slice_kicker',
            'ring',
            'head_tail',
        ]
    )
    line.particle_ref   = reference
    line.build_tracker(_context = context)
    line.feedback.enable()
    particles = xt.Particles(
        _context    = context,
        p0c         = P0C,
        y           = [Y_OFFSET, -Y_OFFSET],
        zeta        = launch_zeta,
        pzeta       = launch_pzeta,
    )
    # The terminal calculation changes the field only for the following pass.
    line.track(particles, num_turns = N_TURNS)

    ########################################
    # Resolve the centroid and head–tail motion
    ########################################
    pickup      = line.get('pickup')
    y           = pickup.get('mean_y', slot = 0)
    py          = pickup.get('mean_py', slot = 0)
    phase_space = np.stack((y, BETA_Y * py), axis = -1)
    centroid    = np.linalg.norm(phase_space.mean(axis = 1), axis = 1)
    difference  = (phase_space[:, 1] - phase_space[:, 0]) / 2
    head_tail   = np.linalg.norm(difference, axis = 1)
    amplitude   = np.hypot(centroid, head_tail)
    # Fit at the same synchrotron phase to remove within-period modulation.
    fit_amplitude   = amplitude[sample] / amplitude[sample[0]]
    growth          = np.polyfit(sample, np.log(fit_amplitude), 1)[0]
    results[name]   = dict(
        y_m             = y,
        applied_rad     = np.array([row['applied_output'] for row in fb.log]),
        zeta_m          = pickup.get('mean_zeta', slot = 0),
        centroid_m      = centroid,
        head_tail_m     = head_tail,
        growth_per_turn = growth,
        recorded_weight = pickup.get('num_particles', slot = 0),
    )

################################################################################
# Compare the modal growth and inspect the command delay
################################################################################
print('Head–tail growth and damping\n----------------------------')
print('Case                            Log-amplitude rate [1/turn]')
for name, result in results.items():
    print(f'{cases[name][0]:30s} {result["growth_per_turn"]:+24.5f}')
print()
print('With the wake off, the launch oscillates at constant amplitude.')
print(f'Fit uses pickup turns {sample[0]} to {sample[-1]} '
      f'in steps of {period}.')
print('The fit samples the total transverse amplitude '
      'at one synchrotron phase.')

print('\nFrom the pickup to the kick\n---------------------------')

########################################
# Locate the first affected kick
########################################
first_kick = results['slice']['applied_rad'][0] * 1e9
print(f'Initial vertical offsets, head / tail [µm]: '
      f'{Y_OFFSET * 1e6:+.3f} / {-Y_OFFSET * 1e6:+.3f}')
print('The uniform dipole gives both slices the same vertical kick.')
print('The RF dipole gives an odd vertical kick proportional to sin(k·ζ).')
print(f'Uniform kick [nrad]: {first_kick[0]:+.3f}')
print(f'Odd RF kick amplitude [nrad]: {first_kick[1]:+.3f}')
print('Read at the pickup on turn 0. Apply at the kicker on turn 1.')
print('The upstream pickup first sees the response on turn 2.')

################################################################################
# Compare head–tail and centroid motion across the cases
################################################################################
fig, axes   = plt.subplots(2, 1, sharex = True, figsize = (8, 6))
colors      = ('tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red')
for (name, result), color in zip(results.items(), colors):
    axes[0].semilogy(1e6 * result['head_tail_m'],
                     color = color, label = cases[name][0])
    axes[1].plot(1e6 * result['centroid_m'], color = color)
axes[0].set(title = 'Half the head–tail difference',
            ylabel = 'Amplitude [µm]')
axes[1].set(title = 'Bunch centroid', ylabel = 'Amplitude [µm]',
            xlabel = 'Pickup turn')
# A linear range near zero keeps the wake-off centroid visible.
axes[1].set_yscale('symlog', linthresh = 1e-4)
fig.legend(
    *axes[0].get_legend_handles_labels(), loc = 'upper center', ncols = 3,
)
fig.tight_layout(rect = (0, 0, 1, 0.91))

################################################################################
# Plot how the measured slices drive the next-pass RF waveform
################################################################################
result          = results['slice']
zeta            = np.linspace(-ZETA_MAX, ZETA_MAX, 201)
measured_zeta   = result['zeta_m'][0]
next_zeta       = result['zeta_m'][1]
waveform        = first_kick[0] + first_kick[1] * np.sin(k_rf * zeta)
sampled_kick    = first_kick[0] + first_kick[1] * np.sin(
    k_rf * next_zeta
)

kick_fig, axes = plt.subplots(1, 2, figsize = (9, 3.6), sharex = True)
axes[0].scatter(
    measured_zeta * 100,
    result['y_m'][0] * 1e6,
    color = ['tab:orange', 'tab:blue'],
)
axes[0].axhline(
    result['y_m'][0].mean() * 1e6,
    color   = '0.4',
    ls      = ':',
    label   = 'Bunch centroid',
)
axes[0].set(title = 'Pickup turn 0', ylabel = 'Slice centroid [µm]')
axes[0].legend(loc = 'upper left')
axes[1].plot(
    zeta * 100, waveform, color = '0.35', label = 'Kicker waveform'
)
axes[1].scatter(
    next_zeta * 100,
    sampled_kick,
    color   = ['tab:orange', 'tab:blue'],
    zorder  = 3,
)
axes[1].set(title = 'Kicker turn 1', ylabel = 'Feedback kick [nrad]')
axes[1].legend(loc = 'upper left')
for axis in axes:
    axis.set_xlabel('Longitudinal coordinate ζ [cm]')
    axis.grid(alpha = 0.2)
    axis.margins(x = 0.08, y = 0.2)
kick_fig.tight_layout()
plt.show()
