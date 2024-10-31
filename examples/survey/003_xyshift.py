"""
Test the survey on XY Shifts
"""
################################################################################
# Packages
################################################################################
import numpy as np
import matplotlib.pyplot as plt

from _helpers import survey_test, summary_plot
import xtrack as xt

################################################################################
# User variables
################################################################################
PLOT_COMPARISONS    = True
TOLERANCE           = 1E-12
REL_TOLERANCE       = 1E-3

################################################################################
# Create line
################################################################################
env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
line    = env.new_line(name = 'line', components=[
    env.new('bend', xt.Bend, k0 = 0, h = 0, length = 0.5, at = 1),
    env.new('xyshift', xt.XYShift, dx = 0, dy = 0, at = 2),
    env.new('end', xt.Marker, at = 3)])
line.configure_bend_model(edge = 'suppressed')
line.config.XTRACK_USE_EXACT_DRIFTS = True

################################################################################
# X Shift
################################################################################

########################################
# Shift only
########################################
line['bend'].k0         = 0
line['bend'].h          = 0
line['bend'].rot_s_rad  = 0

line['xyshift'].dx      = 0.1
line['xyshift'].dy      = 0

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Shift X, bend off')

# Survey negative of Twiss for h != 0, k0 = 0
assert np.allclose(sv.X[-1], -tw.x[-1], rtol=REL_TOLERANCE)
# MadPoint sum of twiss and survey
assert np.allclose(sv.X[-1] + tw.x[-1], sv.xx[-1], rtol=REL_TOLERANCE)
# All y related quantities are zero
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

########################################
# Shift + horizontal bend
########################################
line['bend'].k0         = 1
line['bend'].h          = 1
line['bend'].rot_s_rad  = 0

line['xyshift'].dx      = 0.1
line['xyshift'].dy      = 0

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Shift X, bend off')

# MadPoint sum of twiss and survey
# TODO: This is broken?
# assert np.allclose(sv.X[-1] + tw.x[-1], sv.xx[-1], rtol=REL_TOLERANCE)
# All y related quantities are zero
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

########################################
# Shift + vertical bend
########################################
line['bend'].k0         = 1
line['bend'].h          = 1
line['bend'].rot_s_rad  = np.pi / 2

line['xyshift'].dx      = 0.1
line['xyshift'].dy      = 0

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Shift X, bend off')

# MadPoint sum of twiss and survey
# TODO: This is broken?
# assert np.allclose(sv.X[-1] + tw.x[-1], sv.xx[-1], rtol=REL_TOLERANCE)

################################################################################
# Y Shift
################################################################################

########################################
# Shift only
########################################
line['bend'].k0         = 0
line['bend'].h          = 0
line['bend'].rot_s_rad  = 0

line['xyshift'].dx      = 0
line['xyshift'].dy      = 0.1

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Shift X, bend off')

# Survey negative of Twiss for h != 0, k0 = 0
assert np.allclose(sv.Y[-1], -tw.y[-1], rtol=REL_TOLERANCE)
# MadPoint sum of twiss and survey
assert np.allclose(sv.Y[-1] + tw.y[-1], sv.yy[-1], rtol=REL_TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)

########################################
# Shift + horizontal bend
########################################
line['bend'].k0         = 1
line['bend'].h          = 1
line['bend'].rot_s_rad  = 0

line['xyshift'].dx      = 0
line['xyshift'].dy      = 0.1

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Shift X, bend off')

# MadPoint sum of twiss and survey
# TODO: This is broken?
# assert np.allclose(sv.Y[-1] + tw.y[-1], sv.yy[-1], rtol=REL_TOLERANCE)

########################################
# Shift + vertical bend
########################################
line['bend'].k0         = 1
line['bend'].h          = 1
line['bend'].rot_s_rad  = np.pi / 2

line['xyshift'].dx      = 0
line['xyshift'].dy      = 0.1

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Shift X, bend off')

# MadPoint sum of twiss and survey
# TODO: This is broken?
# assert np.allclose(sv.Y[-1] + tw.y[-1], sv.yy[-1], rtol=REL_TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)

########################################
# Show plots
########################################
if PLOT_COMPARISONS:
    plt.show()
