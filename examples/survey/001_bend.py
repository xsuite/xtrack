"""
Test the survey on Bends
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

bend_h              = 1E-3

################################################################################
# Create line
################################################################################
env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
line    = env.new_line(name = 'line', components=[
    env.new('bend', xt.Bend, k0 = 0, h = 0, length = 0.5, at = 1),
    env.new('end', xt.Marker, at = 2)])
line.configure_bend_model(edge = 'suppressed')
line.config.XTRACK_USE_EXACT_DRIFTS = True

################################################################################
# Horizontal Bend
################################################################################

########################################
# h = k0 = 0
########################################
line['bend'].k0         = 0
line['bend'].h          = 0
line['bend'].rot_s_rad  = 0

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Horizontal Bend: h = k0 = 0')

# Everything zero here
assert np.allclose(sv.X[-1], 0, atol=TOLERANCE)
assert np.allclose(sv.xx[-1], 0, atol=TOLERANCE)
assert np.allclose(tw.x[-1], 0, atol=TOLERANCE)
# All y related quantities are zero
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

########################################
# h = 0, k0 != 0
########################################
line['bend'].k0         = bend_h
line['bend'].h          = 0
line['bend'].rot_s_rad  = 0

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Horizontal Bend: h = 0, k0 != 0')

# No survey with h = 0
assert np.allclose(sv.X[-1], 0, atol=TOLERANCE)
# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.X[-1] + tw.x[-1], sv.xx[-1], rtol=REL_TOLERANCE)
# All y related quantities are zero
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

########################################
# h != 0, k0 = 0
########################################
line['bend'].k0         = 0
line['bend'].h          = bend_h
line['bend'].rot_s_rad  = 0

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Horizontal Bend: h != 0, k0 = 0')

# Survey negative of Twiss for h != 0, k0 = 0
assert np.allclose(sv.X[-1], -tw.x[-1], rtol=REL_TOLERANCE)
# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.X[-1] + tw.x[-1], sv.xx[-1], rtol=REL_TOLERANCE)
# All y related quantities are zero
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

########################################
# h != k0 !=0 
########################################
line['bend'].k0         = bend_h * 2
line['bend'].h          = bend_h
line['bend'].rot_s_rad  = 0

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Horizontal Bend: h = k0 /2')

# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.X[-1] + tw.x[-1], sv.xx[-1], rtol=REL_TOLERANCE)
# All y related quantities are zero
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

########################################
# h = k0 != 0
########################################
line['bend'].k0         = bend_h
line['bend'].h          = bend_h
line['bend'].rot_s_rad  = 0

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Horizontal Bend: h = k0 != 0')

# No orbit with h=k0
assert np.allclose(tw.x[-1], 0, atol=TOLERANCE)
# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.X[-1] + tw.x[-1], sv.xx[-1], rtol=REL_TOLERANCE)
# All y related quantities are zero
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

################################################################################
# Vertical Bend
################################################################################

########################################
# h = k0 = 0
########################################
line['bend'].k0         = 0
line['bend'].h          = 0
line['bend'].rot_s_rad  = np.pi / 2

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Vertical Bend: h = k0 = 0')

# Everything zero here
assert np.allclose(sv.Y[-1], 0, atol=TOLERANCE)
assert np.allclose(sv.yy[-1], 0, atol=TOLERANCE)
assert np.allclose(tw.y[-1], 0, atol=TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)

########################################
# h = 0, k0 != 0
########################################
line['bend'].k0         = bend_h
line['bend'].h          = 0
line['bend'].rot_s_rad  = np.pi / 2

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Vertical Bend: h = 0, k0 != 0')

assert np.allclose(sv.Y[-1], 0, atol=TOLERANCE)
assert np.allclose(sv.yy[-1], tw.y[-1], rtol=REL_TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)

########################################
# h != 0, k0 = 0
########################################
line['bend'].k0         = 0
line['bend'].h          = bend_h
line['bend'].rot_s_rad  = np.pi / 2

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Vertical Bend: h != 0, k0 = 0')

# Survey negative of Twiss
assert np.allclose(sv.Y[-1], -tw.y[-1], rtol=REL_TOLERANCE)
# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.Y[-1] + tw.y[-1], sv.yy[-1], rtol=REL_TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)

########################################
# h != k0 !=0 
########################################
line['bend'].k0         = bend_h * 2
line['bend'].h          = bend_h
line['bend'].rot_s_rad  = np.pi / 2

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Vertical Bend: h = k0 /2')

# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.Y[-1] + tw.y[-1], sv.yy[-1], rtol=REL_TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)

########################################
# h = k0 != 0
########################################
line['bend'].k0         = bend_h
line['bend'].h          = bend_h
line['bend'].rot_s_rad  = np.pi / 2

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Vertical Bend: h = k0 != 0')

assert np.allclose(tw.y[-1], 0, atol=TOLERANCE)
# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.Y[-1] + tw.y[-1], sv.yy[-1], rtol=REL_TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)

########################################
# Show plots
########################################
if PLOT_COMPARISONS:
    plt.show()
