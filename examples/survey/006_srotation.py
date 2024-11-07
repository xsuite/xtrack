"""
Test the survey on SRotation
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
ROT_DEG             = 30
HXL                 = 0.1
DX                  = 0.5
PLOT_X_LIMS         = (-0.1, 4.1)
PLOT_Y_LIMS         = (-1, 1)

################################################################################
# Plot setup
################################################################################
fig_mult_h  = plt.figure(figsize = (16, 8))
gs_mult_h   = fig_mult_h.add_gridspec(3, 5, hspace = 0.3, wspace = 0)
axs_mult_h  = gs_mult_h.subplots(sharex = 'row', sharey = True)

fig_mult_v  = plt.figure(figsize = (16, 8))
gs_mult_v   = fig_mult_v.add_gridspec(3, 5, hspace = 0.3, wspace = 0)
axs_mult_v  = gs_mult_v.subplots(sharex = 'row', sharey = True)

fig_shift_h = plt.figure(figsize = (16, 8))
gs_shift_h  = fig_shift_h.add_gridspec(3, 5, hspace = 0.3, wspace = 0)
axs_shift_h = gs_shift_h.subplots(sharex = 'row', sharey = True)

fig_shift_v = plt.figure(figsize = (16, 8))
gs_shift_v  = fig_shift_v.add_gridspec(3, 5, hspace = 0.3, wspace = 0)
axs_shift_v = gs_shift_v.subplots(sharex = 'row', sharey = True)

################################################################################
# Horizontal Multipole Bends
################################################################################

########################################
# Create line
########################################
env     = xt.Environment(particle_ref = xt.Particles(p0c = 1E9))
line    = env.new_line(name = 'line', components=[
    env.new('mult1', xt.Multipole, hxl = 0, length = 0.5, at = 1),
    env.new('srotation', xt.SRotation, angle = 0, at = 2),
    env.new('mult2', xt.Multipole, hxl = 0, length = 0.5, at = 3),
    env.new('end', xt.Marker, at = 4)])

########################################
# Off
########################################
line['mult1'].hxl       = 0
line['mult2'].hxl       = 0
line['srotation'].angle = np.rad2deg(0)

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_mult_h, sv, tw, 0, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Element off must have no effect
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

########################################
# Rotation Only
########################################
line['mult1'].hxl       = 0
line['mult2'].hxl       = 0
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_mult_h, sv, tw, 1, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

########################################
# Mult then rot
########################################
line['mult1'].hxl       = HXL
line['mult2'].hxl       = 0
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_mult_h, sv, tw, 2, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

########################################
# Rot then mult
########################################
line['mult1'].hxl       = 0
line['mult2'].hxl       = HXL
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_mult_h, sv, tw, 3, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

########################################
# Mult then rot then mult
########################################
line['mult1'].hxl       = HXL
line['mult2'].hxl       = HXL
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_mult_h, sv, tw, 4, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

################################################################################
# Vertical Multipole Bends
################################################################################

########################################
# Create line
########################################
env     = xt.Environment(particle_ref = xt.Particles(p0c = 1E9))
line    = env.new_line(name = 'line', components=[
    env.new('mult1', xt.Multipole, hxl = 0, length = 0.5, at = 1, rot_s_rad = np.pi / 2),
    env.new('srotation', xt.SRotation, angle = 0, at = 2),
    env.new('mult2', xt.Multipole, hxl = 0, length = 0.5, at = 3, rot_s_rad = np.pi / 2),
    env.new('end', xt.Marker, at = 4)])

########################################
# Off
########################################
line['mult1'].hxl       = 0
line['mult2'].hxl       = 0
line['srotation'].angle = np.rad2deg(0)

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_mult_v, sv, tw, 0, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Element off must have no effect
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

########################################
# Rotation Only
########################################
line['mult1'].hxl       = 0
line['mult2'].hxl       = 0
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_mult_v, sv, tw, 1, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

########################################
# Mult then rot
########################################
line['mult1'].hxl       = HXL
line['mult2'].hxl       = 0
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_mult_v, sv, tw, 2, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

########################################
# Rot then mult
########################################
line['mult1'].hxl       = 0
line['mult2'].hxl       = HXL
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_mult_v, sv, tw, 3, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

########################################
# Mult then rot then mult
########################################
line['mult1'].hxl       = HXL
line['mult2'].hxl       = HXL
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_mult_v, sv, tw, 4, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

################################################################################
# Horizontal Shift
################################################################################

########################################
# Create line
########################################
env     = xt.Environment(particle_ref = xt.Particles(p0c = 1E9))
line    = env.new_line(name = 'line', components=[
    env.new('shift1', xt.XYShift, dx = 0, dy = 0, at = 1),
    env.new('srotation', xt.SRotation, angle = 0, at = 2),
    env.new('shift2', xt.XYShift, dx = 0, dy = 0, at = 3),
    env.new('end', xt.Marker, at = 4)])

########################################
# Off
########################################
line['shift1'].dx       = 0
line['shift2'].dx       = 0
line['srotation'].angle = np.rad2deg(0)

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_shift_h, sv, tw, 0, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Element off must have no effect
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

########################################
# Rotation Only
########################################
line['shift1'].dx       = 0
line['shift2'].dx       = 0
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_shift_h, sv, tw, 1, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

########################################
# Mult then rot
########################################
line['shift1'].dx       = DX
line['shift2'].dx       = 0
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_shift_h, sv, tw, 2, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

########################################
# Rot then mult
########################################
line['shift1'].dx       = 0
line['shift2'].dx       = DX
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_shift_h, sv, tw, 3, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

########################################
# Mult then rot then mult
########################################
line['shift1'].dx       = DX
line['shift2'].dx       = DX
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_shift_h, sv, tw, 4, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

################################################################################
# Vertical Shift
################################################################################

########################################
# Create line
########################################
env     = xt.Environment(particle_ref = xt.Particles(p0c = 1E9))
line    = env.new_line(name = 'line', components=[
    env.new('shift1', xt.XYShift, dx = 0, dy = 0, at = 1),
    env.new('srotation', xt.SRotation, angle = 0, at = 2),
    env.new('shift2', xt.XYShift, dx = 0, dy = 0, at = 3),
    env.new('end', xt.Marker, at = 4)])

########################################
# Off
########################################
line['shift1'].dy       = 0
line['shift2'].dy       = 0
line['srotation'].angle = np.rad2deg(0)

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_shift_v, sv, tw, 0, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Element off must have no effect
assert np.allclose(np.array([sv.X[-1], sv.xx[-1], tw.x[-1]]), 0, atol=TOLERANCE)
assert np.allclose(np.array([sv.Y[-1], sv.yy[-1], tw.y[-1]]), 0, atol=TOLERANCE)

########################################
# Rotation Only
########################################
line['shift1'].dy       = 0
line['shift2'].dy       = 0
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_shift_v, sv, tw, 1, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

########################################
# Mult then rot
########################################
line['shift1'].dy       = DX
line['shift2'].dy       = 0
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_shift_v, sv, tw, 2, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

########################################
# Rot then mult
########################################
line['shift1'].dy       = 0
line['shift2'].dy       = DX
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_shift_v, sv, tw, 3, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

########################################
# Mult then rot then mult
########################################
line['shift1'].dy       = DX
line['shift2'].dy       = DX
line['srotation'].angle = ROT_DEG

sv, tw = madpoint_twiss_survey(line)

add_to_plot(axs_shift_v, sv, tw, 4, xlims = PLOT_X_LIMS, ylims = PLOT_Y_LIMS)

####################
# Tests
####################
# Only rotation must have no effect
# TODO

################################################################################
# Show Plots
################################################################################

########################################
# Horizontal Mult
########################################
# y labels
axs_mult_h[0, 0].set_ylabel('Survey x,y [m]')
axs_mult_h[1, 0].set_ylabel('Twiss x,y [m]')
axs_mult_h[2, 0].set_ylabel('MadPoint xx,yy [m]')

# x labels
axs_mult_h[0, 2].set_xlabel('Z [m]')
axs_mult_h[1, 2].set_xlabel('s [m]')
axs_mult_h[2, 2].set_xlabel('s [m]')

# Titles
axs_mult_h[0, 0].set_title('Rot  = 0,\n HXL1 = 0, HXL2 = 0')
axs_mult_h[0, 1].set_title('Rot != 0,\n HXL1 = 0, HXL2 = 0')
axs_mult_h[0, 2].set_title('Rot != 0,\n HXL1 != 0, HXL2 = 0')
axs_mult_h[0, 3].set_title('Rot != 0,\n HXL1 = 0, HXL2 != 0')
axs_mult_h[0, 4].set_title('Rot != 0,\n HXL1 != 0, HXL2 != 0')

legend_elements = [
    Line2D([0], [0], color = 'black', lw = 2, label = 'x'),
    Line2D([0], [0], color = 'red',   lw = 2, label = 'y')]
fig_mult_h.legend(
    handles         = legend_elements,
    loc             = 'lower center',
    ncol            = 2,
    frameon         = False,
    bbox_to_anchor  = (0.5, -0.05))
fig_mult_h.suptitle('Horizontal Multipole and SRotation')

########################################
# Vertical Mult
########################################
# y labels
axs_mult_v[0, 0].set_ylabel('Survey x,y [m]')
axs_mult_v[1, 0].set_ylabel('Twiss x,y [m]')
axs_mult_v[2, 0].set_ylabel('MadPoint xx,yy [m]')

# x labels
axs_mult_v[0, 2].set_xlabel('Z [m]')
axs_mult_v[1, 2].set_xlabel('s [m]')
axs_mult_v[2, 2].set_xlabel('s [m]')

# Titles
axs_mult_v[0, 0].set_title('Rot  = 0,\n HYL1 = 0, HYL2 = 0')
axs_mult_v[0, 1].set_title('Rot != 0,\n HYL1 = 0, HYL2 = 0')
axs_mult_v[0, 2].set_title('Rot != 0,\n HYL1 != 0, HYL2 = 0')
axs_mult_v[0, 3].set_title('Rot != 0,\n HYL1 = 0, HYL2 != 0')
axs_mult_v[0, 4].set_title('Rot != 0,\n HYL1 != 0, HYL2 != 0')

legend_elements = [
    Line2D([0], [0], color = 'black', lw = 2, label = 'x'),
    Line2D([0], [0], color = 'red',   lw = 2, label = 'y')]
fig_mult_v.legend(
    handles         = legend_elements,
    loc             = 'lower center',
    ncol            = 2,
    frameon         = False,
    bbox_to_anchor  = (0.5, -0.05))
fig_mult_v.suptitle('Vertical Multipole and SRotation')

########################################
# Horizontal XYShift
########################################
# y labels
axs_shift_h[0, 0].set_ylabel('Survey x,y [m]')
axs_shift_h[1, 0].set_ylabel('Twiss x,y [m]')
axs_shift_h[2, 0].set_ylabel('MadPoint xx,yy [m]')

# x labels
axs_shift_h[0, 2].set_xlabel('Z [m]')
axs_shift_h[1, 2].set_xlabel('s [m]')
axs_shift_h[2, 2].set_xlabel('s [m]')

# Titles
axs_shift_h[0, 0].set_title('Rot  = 0,\n DX1 = 0, DX2 = 0')
axs_shift_h[0, 1].set_title('Rot != 0,\n DX1 = 0, DX2 = 0')
axs_shift_h[0, 2].set_title('Rot != 0,\n DX1 != 0, DX2 = 0')
axs_shift_h[0, 3].set_title('Rot != 0,\n DX1 = 0, DX2 != 0')
axs_shift_h[0, 4].set_title('Rot != 0,\n DX1 != 0, DX2 != 0')

legend_elements = [
    Line2D([0], [0], color = 'black', lw = 2, label = 'x'),
    Line2D([0], [0], color = 'red',   lw = 2, label = 'y')]
fig_shift_h.legend(
    handles         = legend_elements,
    loc             = 'lower center',
    ncol            = 2,
    frameon         = False,
    bbox_to_anchor  = (0.5, -0.05))
fig_shift_h.suptitle('Horizontal XYShift and SRotation')

########################################
# Vertical XYShift
########################################
# y labels
axs_shift_v[0, 0].set_ylabel('Survey x,y [m]')
axs_shift_v[1, 0].set_ylabel('Twiss x,y [m]')
axs_shift_v[2, 0].set_ylabel('MadPoint xx,yy [m]')

# x labels
axs_shift_v[0, 2].set_xlabel('Z [m]')
axs_shift_v[1, 2].set_xlabel('s [m]')
axs_shift_v[2, 2].set_xlabel('s [m]')

# Titles
axs_shift_v[0, 0].set_title('Rot  = 0,\n DY1 = 0, DY2 = 0')
axs_shift_v[0, 1].set_title('Rot != 0,\n DY1 = 0, DY2 = 0')
axs_shift_v[0, 2].set_title('Rot != 0,\n DY1 != 0, DY2 = 0')
axs_shift_v[0, 3].set_title('Rot != 0,\n DY1 = 0, DY2 != 0')
axs_shift_v[0, 4].set_title('Rot != 0,\n DY1 != 0, DY2 != 0')

legend_elements = [
    Line2D([0], [0], color = 'black', lw = 2, label = 'x'),
    Line2D([0], [0], color = 'red',   lw = 2, label = 'y')]
fig_shift_v.legend(
    handles         = legend_elements,
    loc             = 'lower center',
    ncol            = 2,
    frameon         = False,
    bbox_to_anchor  = (0.5, -0.05))
fig_shift_v.suptitle('Vertical XYShift and SRotation')

########################################
# Show
########################################
plt.tight_layout()
plt.show()
