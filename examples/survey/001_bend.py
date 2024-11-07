"""
Test the survey on Bends
"""
################################################################################
# Packages
################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _helpers import madpoint_twiss_survey, add_to_plot
import xtrack as xt

################################################################################
# User variables
################################################################################
TOLERANCE           = 1E-12

# Small angle needed for paraxial approximation tests
BEND_ANGLE          = 1E-3
BEND_LENGTH         = 0.5

################################################################################
# Plot setup
################################################################################
fig_h   = plt.figure(figsize = (16, 8))
gs_h    = fig_h.add_gridspec(3, 5, hspace = 0.3, wspace = 0)
axs_h   = gs_h.subplots(sharex = 'row', sharey = True)

fig_v   = plt.figure(figsize = (16, 8))
gs_v    = fig_v.add_gridspec(3, 5, hspace = 0.3, wspace = 0)
axs_v   = gs_v.subplots(sharex = 'row', sharey = True)

################################################################################
# Create line
################################################################################
env     = xt.Environment(particle_ref = xt.Particles(p0c = 1E9))
line    = env.new_line(
    name        = 'line',
    components  = [
        env.new('bend', xt.Bend, k0 = 0, h = 0, length = BEND_LENGTH, at = 1),
        env.new('end', xt.Marker, at = 2)])

########################################
# Configure Bend Model
########################################
line.configure_bend_model(edge = 'suppressed')

################################################################################
# Horizontal Bend
################################################################################

########################################
# h = k0 = 0
########################################
line['bend'].k0         = 0
line['bend'].h          = 0
line['bend'].rot_s_rad  = 0

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_h, sv, tw, 0)

####################
# Tests
####################
# Element off must have no effect
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

########################################
# h = 0, k0 != 0
########################################
line['bend'].k0         = BEND_ANGLE / BEND_LENGTH
line['bend'].h          = 0
line['bend'].rot_s_rad  = 0

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_h, sv, tw, 1)

####################
# Tests
####################
# No bending with h = 0, so the survey must be zero
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

add_to_plot(axs_h, sv, tw, 2)

####################
# Tests
####################
# With h!=0, k0=0, the survey must be the negative of the Twiss
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

add_to_plot(axs_h, sv, tw, 3)

####################
# Tests
####################
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

add_to_plot(axs_h, sv, tw, 4)

####################
# Tests
####################
# With h=k0, there should be no residual orbit on the twiss
assert np.allclose(tw.x[-1], 0, atol = TOLERANCE)
# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.X[-1] + tw.x[-1], sv.xx[-1], rtol = TOLERANCE)
# All y related quantities are zero
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol = TOLERANCE)

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

add_to_plot(axs_v, sv, tw, 0)

####################
# Tests
####################
# Element off must have no effect
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

########################################
# h = 0, k0 != 0
########################################
line['bend'].k0         = BEND_ANGLE / BEND_LENGTH
line['bend'].h          = 0
line['bend'].rot_s_rad  = np.pi / 2

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_v, sv, tw, 1)

####################
# Tests
####################
# No bending with h = 0, so the survey must be zero
assert np.allclose(sv.Y[-1], 0, atol = TOLERANCE)
# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.Y[-1] + tw.y[-1], sv.yy[-1], rtol = TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol = TOLERANCE)

########################################
# h != 0, k0 = 0
########################################
line['bend'].k0         = 0
line['bend'].h          = BEND_ANGLE / BEND_LENGTH
line['bend'].rot_s_rad  = np.pi / 2

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_v, sv, tw, 2)

####################
# Tests
####################
# With h!=0, k0=0, the survey must be the negative of the Twiss
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

add_to_plot(axs_v, sv, tw, 3)

####################
# Tests
####################
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

add_to_plot(axs_v, sv, tw, 4)

####################
# Tests
####################
# With h=k0, there should be no residual orbit on the twiss
assert np.allclose(tw.y[-1], 0, atol = TOLERANCE)
# MadPoint sum of twiss and survey (paraxial approximation)
assert np.allclose(sv.Y[-1] + tw.y[-1], sv.yy[-1], rtol = TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol = TOLERANCE)

################################################################################
# Show Plots
################################################################################

########################################
# Horizontal Bends
########################################
# y labels
axs_h[0, 0].set_ylabel('Survey x,y [m]')
axs_h[1, 0].set_ylabel('Twiss x,y [m]')
axs_h[2, 0].set_ylabel('MadPoint xx,yy [m]')

# x labels
axs_h[0, 2].set_xlabel('Z [m]')
axs_h[1, 2].set_xlabel('s [m]')
axs_h[2, 2].set_xlabel('s [m]')

# Titles
axs_h[0, 0].set_title('h = 0, k0 = 0')
axs_h[0, 1].set_title('h = 0, k0 != 0')
axs_h[0, 2].set_title('h != 0, k0 = 0')
axs_h[0, 3].set_title('h = k0 / 2 != 0')
axs_h[0, 4].set_title('h = k0 != 0')

legend_elements = [
    Line2D([0], [0], color = 'black', lw = 2, label = 'x'),
    Line2D([0], [0], color = 'red',   lw = 2, label = 'y')]
fig_h.legend(
    handles         = legend_elements,
    loc             = 'lower center',
    ncol            = 2,
    frameon         = False,
    bbox_to_anchor  = (0.5, -0.05))
fig_h.suptitle('Horizontal Bend')

########################################
# Vertical Bends
########################################
# y labels
axs_v[0, 0].set_ylabel('Survey x,y [m]')
axs_v[1, 0].set_ylabel('Twiss x,y [m]')
axs_v[2, 0].set_ylabel('MadPoint xx,yy [m]')

# x labels
axs_v[0, 2].set_xlabel('Z [m]')
axs_v[1, 2].set_xlabel('s [m]')
axs_v[2, 2].set_xlabel('s [m]')

# Titles
axs_v[0, 0].set_title('h = 0, k0 = 0')
axs_v[0, 1].set_title('h = 0, k0 != 0')
axs_v[0, 2].set_title('h != 0, k0 = 0')
axs_v[0, 3].set_title('h = k0 / 2 != 0')
axs_v[0, 4].set_title('h = k0 != 0')

legend_elements = [
    Line2D([0], [0], color = 'black', lw = 2, label = 'x'),
    Line2D([0], [0], color = 'red',   lw = 2, label = 'y')]
fig_v.legend(
    handles         = legend_elements,
    loc             = 'lower center',
    ncol            = 2,
    frameon         = False,
    bbox_to_anchor  = (0.5, -0.05))
fig_v.suptitle('Vertical Bend')

########################################
# Show
########################################
plt.tight_layout()
plt.show()
