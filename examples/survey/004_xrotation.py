"""
Test the survey on XRotation
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

hxl                 = 1E-3

################################################################################
# Create line
################################################################################
env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
line    = env.new_line(name = 'line', components=[
    env.new('mult', xt.Multipole, hxl = 0, length = 0.5, at = 1),
    env.new('xrotation', xt.XRotation, angle = 0, at = 2),
    env.new('end', xt.Marker, at = 3)])
line.configure_bend_model(edge = 'suppressed')
line.config.XTRACK_USE_EXACT_DRIFTS = True

################################################################################
# Tests
################################################################################

########################################
# Rotation only
########################################
line['mult'].hxl        = 0
line['mult'].rot_s_rad  = 0

line['xrotation'].angle = np.rad2deg(hxl)

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Rotation X, bend off')

########################################
# Shift + horizontal bend
########################################
line['mult'].hxl        = hxl
line['mult'].rot_s_rad  = 0

line['xrotation'].angle = np.rad2deg(hxl)

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Rotation X, Horiztonal Mult')

########################################
# Shift + vertical bend
########################################
line['mult'].hxl        = hxl
line['mult'].rot_s_rad  = np.pi / 2

line['xrotation'].angle = np.rad2deg(hxl)

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Rotation X, Vertical Mult')

########################################
# Show plots
########################################
if PLOT_COMPARISONS:
    plt.show()
