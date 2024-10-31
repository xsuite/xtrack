"""
Test the survey on YRotation
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
    env.new('yrotation', xt.YRotation, angle = 0, at = 2),
    env.new('end', xt.Marker, at = 3)])
line.configure_bend_model(edge = 'suppressed')
line.config.XTRACK_USE_EXACT_DRIFTS = True

################################################################################
# Tests
################################################################################

########################################
# Rotation only
########################################
line['bend'].k0         = 0
line['bend'].h          = 0
line['bend'].rot_s_rad  = 0

line['yrotation'].angle = 10

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Rotation Y, bend off')

########################################
# Shift + horizontal bend
########################################
line['bend'].k0         = 1
line['bend'].h          = 1
line['bend'].rot_s_rad  = 0

line['yrotation'].angle = 10

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Rotation Y, Horiztonal Bend')

########################################
# Shift + vertical bend
########################################
line['bend'].k0         = 1
line['bend'].h          = 1
line['bend'].rot_s_rad  = np.pi / 2

line['yrotation'].angle = 10

sv, tw = survey_test(line)
if PLOT_COMPARISONS:
    summary_plot(sv, tw, 'Rotation Y, Vertical Bend')

########################################
# Show plots
########################################
if PLOT_COMPARISONS:
    plt.show()
