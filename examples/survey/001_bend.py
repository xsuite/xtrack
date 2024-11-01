"""
Test the survey on Bends
"""
################################################################################
# Packages
################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _helpers import madpoint_twiss_survey, summary_plot, zero_small_values
import xtrack as xt

################################################################################
# User variables
################################################################################
PLOT_COMPARISONS    = True
TOLERANCE           = 1E-12

# Small angle needed to paraxial approximation tests
BEND_ANGLE          = 1E-3
BEND_LENGTH         = 0.5

################################################################################
# Create line
################################################################################
env = xt.Environment(particle_ref = xt.Particles(p0c = 1E9))

line    = env.new_line(
    name        = 'line',
    components  = [
        env.new('bend', xt.Bend, k0 = 0, h = 0, length = BEND_LENGTH, at = 1),
        env.new('end', xt.Marker, at = 2)])

########################################
# Configure Bend and Drift Model
########################################
line.configure_bend_model(edge = 'suppressed')
line.config.XTRACK_USE_EXACT_DRIFTS = True

################################################################################
# Horizontal Bend
################################################################################

########################################
# Plot setup
########################################
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(5, 3, hspace = 0, wspace=0.5)
axs = gs.subplots(sharex=True)

def add_to_plot(axes, survey, twiss, index, tol = 1E-12):
    axes[index, 0].plot(
        zero_small_values(survey.Z, tol),
        zero_small_values(survey.X, tol),
        c = 'k')
    axes[index, 0].plot(
        zero_small_values(survey.Z, tol),
        zero_small_values(survey.Y, tol),
        c = 'r')

    axes[index, 1].plot(
        zero_small_values(twiss.s, tol),
        zero_small_values(twiss.x, tol),
        c = 'k')
    axes[index, 1].plot(
        zero_small_values(twiss.s, tol),
        zero_small_values(twiss.y, tol),
        c = 'r')

    axes[index, 2].plot(
        zero_small_values(survey.s, tol),
        zero_small_values(survey.xx, tol),
        c = 'k')
    axes[index, 2].plot(
        zero_small_values(survey.s, tol),
        zero_small_values(survey.yy, tol),
        c = 'r')


########################################
# h = k0 = 0
########################################
line['bend'].k0         = 0
line['bend'].h          = 0
line['bend'].rot_s_rad  = 0

sv, tw = madpoint_twiss_survey(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Horizontal Bend: h = 0, k0 = 0')
    add_to_plot(axs, sv, tw, 0)

# Everything zero here
assert np.allclose(sv.X[-1],    0, atol = TOLERANCE)
assert np.allclose(sv.xx[-1],   0, atol = TOLERANCE)
assert np.allclose(tw.x[-1],    0, atol = TOLERANCE)
# All y related quantities are zero
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol = TOLERANCE)

########################################
# h = 0, k0 != 0
########################################
line['bend'].k0         = BEND_ANGLE / BEND_LENGTH
line['bend'].h          = 0
line['bend'].rot_s_rad  = 0

sv, tw = madpoint_twiss_survey(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Horizontal Bend: h = 0, k0 != 0')
    add_to_plot(axs, sv, tw, 1)

# No survey with h = 0
assert np.allclose(sv.X[-1], 0, atol = TOLERANCE)
# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.X[-1] + tw.x[-1], sv.xx[-1], rtol = TOLERANCE)
# All y related quantities are zero
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol = TOLERANCE)

########################################
# h != 0, k0 = 0
########################################
line['bend'].k0         = 0
line['bend'].h          = BEND_ANGLE / BEND_LENGTH
line['bend'].rot_s_rad  = 0

sv, tw = madpoint_twiss_survey(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Horizontal Bend: h != 0, k0 = 0')
    add_to_plot(axs, sv, tw, 2)

# Survey negative of Twiss for h != 0, k0 = 0
assert np.allclose(sv.X[-1], -tw.x[-1], rtol = TOLERANCE)
# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.X[-1] + tw.x[-1], sv.xx[-1], rtol = TOLERANCE)
# All y related quantities are zero
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol = TOLERANCE)

########################################
# h != k0 !=0
########################################
line['bend'].k0         = BEND_ANGLE / BEND_LENGTH * 2
line['bend'].h          = BEND_ANGLE / BEND_LENGTH
line['bend'].rot_s_rad  = 0

sv, tw = madpoint_twiss_survey(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Horizontal Bend: h = k0 / 2')
    add_to_plot(axs, sv, tw, 3)

# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.X[-1] + tw.x[-1], sv.xx[-1], rtol = TOLERANCE)
# All y related quantities are zero
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol = TOLERANCE)

########################################
# h = k0 != 0
########################################
line['bend'].k0         = BEND_ANGLE / BEND_LENGTH
line['bend'].h          = BEND_ANGLE / BEND_LENGTH
line['bend'].rot_s_rad  = 0

sv, tw = madpoint_twiss_survey(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Horizontal Bend: h = k0 != 0')
    add_to_plot(axs, sv, tw, 4)

# No orbit with h = k0
assert np.allclose(tw.x[-1], 0, atol = TOLERANCE)
# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.X[-1] + tw.x[-1], sv.xx[-1], rtol = TOLERANCE)
# All y related quantities are zero
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol = TOLERANCE)

########################################
# Plot
########################################
# Titles
axs[0, 0].set_title('Survey')
axs[0, 1].set_title('Twiss')
axs[0, 2].set_title('MadPoint')

# x labels
axs[4, 0].set_xlabel('Z [m]')
axs[4, 1].set_xlabel('s [m]')
axs[4, 2].set_xlabel('s [m]')

# y labels
axs[2, 0].set_ylabel('X [m]')
axs[2, 1].set_ylabel('x [m]')
axs[2, 2].set_ylabel('x [m]')

# Row labels
for j, label in enumerate([
    'h = 0, k0 = 0',
    'h = 0, k0 != 0',
    'h != 0, k0 = 0',
    'h = k0 / 2 != 0',
    'h = k0 != 0']):
    fig.text(0.05, 0.2 * j, label, va='center', ha='center', fontsize=12)

legend_elements = [
    Line2D([0], [0], color='black', lw=2, label='x'),
    Line2D([0], [0], color='red', lw=2, label='y')]

# Place the custom legend below the x-axis labels
fig.legend(handles=legend_elements, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))

fig.suptitle('Horizontal Bend')

plt.tight_layout()
plt.show()

################################################################################
# Vertical Bend
################################################################################

########################################
# h = k0 = 0
########################################
line['bend'].k0         = 0
line['bend'].h          = 0
line['bend'].rot_s_rad  = np.pi / 2

sv, tw = madpoint_twiss_survey(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Vertical Bend: h = 0, k0 = 0')

# Everything zero here
assert np.allclose(sv.Y[-1],    0, atol = TOLERANCE)
assert np.allclose(sv.yy[-1],   0, atol = TOLERANCE)
assert np.allclose(tw.y[-1],    0, atol = TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol = TOLERANCE)

########################################
# h = 0, k0 != 0
########################################
line['bend'].k0         = BEND_ANGLE / BEND_LENGTH
line['bend'].h          = 0
line['bend'].rot_s_rad  = np.pi / 2

sv, tw = madpoint_twiss_survey(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Vertical Bend: h = 0, k0 != 0')

assert np.allclose(sv.Y[-1], 0, atol = TOLERANCE)
assert np.allclose(sv.yy[-1], tw.y[-1], rtol = TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol = TOLERANCE)

########################################
# h != 0, k0 = 0
########################################
line['bend'].k0         = 0
line['bend'].h          = BEND_ANGLE / BEND_LENGTH
line['bend'].rot_s_rad  = np.pi / 2

sv, tw = madpoint_twiss_survey(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Vertical Bend: h != 0, k0 = 0')

# Survey negative of Twiss
assert np.allclose(sv.Y[-1], -tw.y[-1], rtol = TOLERANCE)
# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.Y[-1] + tw.y[-1], sv.yy[-1], rtol = TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol = TOLERANCE)

########################################
# h != k0 !=0 
########################################
line['bend'].k0         = BEND_ANGLE / BEND_LENGTH * 2
line['bend'].h          = BEND_ANGLE / BEND_LENGTH
line['bend'].rot_s_rad  = np.pi / 2

sv, tw = madpoint_twiss_survey(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Vertical Bend: h = k0 / 2')

# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.Y[-1] + tw.y[-1], sv.yy[-1], rtol = TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol = TOLERANCE)

########################################
# h = k0 != 0
########################################
line['bend'].k0         = BEND_ANGLE / BEND_LENGTH
line['bend'].h          = BEND_ANGLE / BEND_LENGTH
line['bend'].rot_s_rad  = np.pi / 2

sv, tw = madpoint_twiss_survey(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Vertical Bend: h = k0 != 0')

assert np.allclose(tw.y[-1], 0, atol = TOLERANCE)
# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.Y[-1] + tw.y[-1], sv.yy[-1], rtol = TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol = TOLERANCE)

########################################
# Show plots
########################################
if PLOT_COMPARISONS:
    plt.show()
