"""
Test the survey on XY Shifts
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

################################################################################
# Plot setup
################################################################################
fig     = plt.figure(figsize = (16, 8))
gs      = fig.add_gridspec(3, 4, hspace = 0.3, wspace = 0)
axs     = gs.subplots(sharex = 'row', sharey = True)

################################################################################
# Create line
################################################################################
env     = xt.Environment(particle_ref = xt.Particles(p0c = 1E9))
line    = env.new_line(name = 'line', components=[
    env.new('xyshift', xt.XYShift, dx = 0, dy = 0, at = 1),
    env.new('end', xt.Marker, at = 2)])

################################################################################
# No Shift
################################################################################
line['xyshift'].dx      = 0
line['xyshift'].dy      = 0

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs, sv, tw, 0, ylims = (-0.15, 0.15))

####################
# Tests
####################
# Element off must have no effect
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

################################################################################
# X Shift
################################################################################
line['xyshift'].dx      = 0.1
line['xyshift'].dy      = 0

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs, sv, tw, 1, ylims = (-0.15, 0.15))

####################
# Tests
####################
# No bending of particles, so must remain zero
assert np.allclose( sv.xx[-1], 0, atol=TOLERANCE)
# Survey negative of Twiss for h != 0, k0 = 0
assert np.allclose(sv.X[-1], -tw.x[-1], rtol=TOLERANCE)
# MadPoint sum of twiss and survey
assert np.allclose(sv.X[-1] + tw.x[-1], sv.xx[-1], rtol=TOLERANCE)
# All y related quantities are zero
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

################################################################################
# Y Shift
################################################################################
line['xyshift'].dx      = 0
line['xyshift'].dy      = 0.1

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs, sv, tw, 2, ylims = (-0.15, 0.15))

####################
# Tests
####################
# No bending of particles, so must remain zero
assert np.allclose(sv.yy[-1], 0, atol=TOLERANCE)
# Survey negative of Twiss for h != 0, k0 = 0
assert np.allclose(sv.Y[-1], -tw.y[-1], rtol=TOLERANCE)
# MadPoint sum of twiss and survey
assert np.allclose(sv.Y[-1] + tw.y[-1], sv.yy[-1], rtol=TOLERANCE)
# All x related quantities are zero
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)

################################################################################
# X and Y Shift
################################################################################
line['xyshift'].dx      = 0.1
line['xyshift'].dy      = 0.1

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs, sv, tw, 3, ylims = (-0.15, 0.15))

####################
# Tests
####################
# No bending of particles, so must remain zero
assert np.allclose( sv.xx[-1], 0, atol=TOLERANCE)
assert np.allclose(sv.yy[-1], 0, atol=TOLERANCE)
# Survey negative of Twiss for h != 0, k0 = 0
assert np.allclose(sv.X[-1], -tw.x[-1], rtol=TOLERANCE)
assert np.allclose(sv.Y[-1], -tw.y[-1], rtol=TOLERANCE)
# MadPoint sum of twiss and survey
assert np.allclose(sv.X[-1] + tw.x[-1], sv.xx[-1], rtol=TOLERANCE)
assert np.allclose(sv.Y[-1] + tw.y[-1], sv.yy[-1], rtol=TOLERANCE)

################################################################################
# Show Plots
################################################################################
# y labels
axs[0, 0].set_ylabel('Survey x,y [m]')
axs[1, 0].set_ylabel('Twiss x,y [m]')
axs[2, 0].set_ylabel('MadPoint xx,yy [m]')

# x labels
axs[0, 2].set_xlabel('Z [m]')
axs[1, 2].set_xlabel('s [m]')
axs[2, 2].set_xlabel('s [m]')

# Titles
axs[0, 0].set_title('DX = 0, DY = 0')
axs[0, 1].set_title('DX != 0, DY = 0')
axs[0, 2].set_title('DX = 0, DY != 0')
axs[0, 3].set_title('DX != 0, DY != 0')

legend_elements = [
    Line2D([0], [0], color = 'black', lw = 2, label = 'x'),
    Line2D([0], [0], color = 'red',   lw = 2, label = 'y')]
fig.legend(
    handles         = legend_elements,
    loc             = 'lower center',
    ncol            = 2,
    frameon         = False,
    bbox_to_anchor  = (0.5, -0.05))
fig.suptitle('XYShift')

########################################
# Show
########################################
plt.tight_layout()
plt.show()
