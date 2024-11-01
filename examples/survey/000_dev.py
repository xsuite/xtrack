"""
Test the survey on Bends
"""
################################################################################
# Packages
################################################################################
import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt

from _helpers import madpoint_twiss_survey, summary_plot

################################################################################
# Setup
################################################################################

env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
env['k0']           = 1E-3
env['h']            = 1E-3
env['dx']           = 2E-6
env['dy']           = 3E-6

# TODO: These don't work with the env.new function
env['angle_x']      = 2E-4
env['angle_y']      = 3E-4

line    = env.new_line(name = 'line', components=[
    env.new('bend1', xt.Bend, k0 = 0, h = 0, length = 0.5, at=2),
    env.new('xyshift', xt.XYShift, dx = 'dx', dy = 'dy', at = 3,),
    env.new('xyshift2', 'xyshift', mode = 'replica', at = 4),
    env.new('yrotation', xt.YRotation, angle = env['angle_x'], at = 5),
    env.new('xrotation', xt.XRotation, angle = env['angle_y'], at = 6),
    env.new('xyshift3', 'xyshift', mode = 'replica', at = 7),
    # env.new('bend2', xt.Bend, k0 = 'k0', h = 'h', length = 0.5, at=8),
    # env.new('xyshift3', 'xyshift', mode = 'replica', at = 9),
    env.new('end', xt.Marker, at = 10)])

# line.cut_at_s(np.linspace(0, line.get_length(), 1001))

################################################################################
# Horizontal Mult vs YRotation
################################################################################

########################################
# Rotation
########################################
env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
line    = env.new_line(name = 'line', components=[
    env.new('yrotation', xt.YRotation, angle = 0, at = 1),
    env.new('xyshift', xt.XYShift, dx = 1E-6, dy = 1E-6, at = 2),
    env.new('end', xt.Marker, at = 3)])
line.configure_bend_model(edge = 'suppressed')
# line.config.XTRACK_USE_EXACT_DRIFTS = True

# NB Minus sign here as rotation works opposite direction
line['yrotation'].angle = -1E-3

sv, tw = madpoint_twiss_survey(line)
summary_plot(sv, tw, 'Rotation Y')

########################################
# Bend
########################################
env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
line    = env.new_line(name = 'line', components=[
    env.new('mult', xt.Multipole, hxl = 0, at = 1),
    env.new('xyshift', xt.XYShift, dx = 1E-6, dy = 1E-6, at = 2),
    env.new('end', xt.Marker, at = 3)])
line.configure_bend_model(edge = 'suppressed')
# line.config.XTRACK_USE_EXACT_DRIFTS = True

line['mult'].hxl = np.deg2rad(1E-3)

sv, tw = madpoint_twiss_survey(line)
summary_plot(sv, tw, 'Horiztonal Mult')

################################################################################
# Vertical Bend vs XRotation
################################################################################

########################################
# Rotation
########################################
env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
line    = env.new_line(name = 'line', components=[
    env.new('xrotation', xt.XRotation, angle = 0, at = 1),
    env.new('xyshift', xt.XYShift, dx = 1E-6, dy = 1E-6, at = 2),
    env.new('end', xt.Marker, at = 3)])
line.configure_bend_model(edge = 'suppressed')
# line.config.XTRACK_USE_EXACT_DRIFTS = True

line['xrotation'].angle = 1E-3

sv, tw = madpoint_twiss_survey(line)
summary_plot(sv, tw, 'Rotation X')

########################################
# Bend
########################################
env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
line    = env.new_line(name = 'line', components=[
    env.new('mult', xt.Multipole, hxl = 0, at = 1, rot_s_rad = np.pi / 2),
    env.new('xyshift', xt.XYShift, dx = 1E-6, dy = 1E-6, at = 2),
    env.new('end', xt.Marker, at = 3)])
line.configure_bend_model(edge = 'suppressed')
# line.config.XTRACK_USE_EXACT_DRIFTS = True

line['mult'].hxl = np.deg2rad(1E-3)

sv, tw = madpoint_twiss_survey(line)
summary_plot(sv, tw, 'Vertical Mult')

################################################################################
# Vertical shift vs SRotation + Horizontal shift
################################################################################

########################################
# Vertical Shift
########################################
env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
line    = env.new_line(name = 'line', components=[
    env.new('xyshift', xt.XYShift, dy = -1E-6, at = 2),
    env.new('end', xt.Marker, at = 3)])
line.configure_bend_model(edge = 'suppressed')
# line.config.XTRACK_USE_EXACT_DRIFTS = True

sv, tw = madpoint_twiss_survey(line)
summary_plot(sv, tw, 'Vertical Shift')

########################################
# Bend
########################################
env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
line    = env.new_line(name = 'line', components=[
    env.new('srot', xt.SRotation, angle = 90, at=1),
    env.new('xyshift', xt.XYShift, dx = 1E-6, at = 2),
    env.new('end', xt.Marker, at = 3)])
line.configure_bend_model(edge = 'suppressed')
# line.config.XTRACK_USE_EXACT_DRIFTS = True

sv, tw = madpoint_twiss_survey(line)
summary_plot(sv, tw, 'SRot + Horizontal Shift')

################################################################################
# SRotation then Horizontal Bend
################################################################################

########################################
# Vertical Shift
########################################
env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
line    = env.new_line(name = 'line', components=[
    env.new('bend', xt.Bend, k0 = 1E-3, h = 1E-3, length = 0.5, at=2),
    env.new('end', xt.Marker, at = 3)])
line.configure_bend_model(edge = 'suppressed')
# line.config.XTRACK_USE_EXACT_DRIFTS = True

sv, tw = madpoint_twiss_survey(line)
summary_plot(sv, tw, 'Horizontal Bend')

########################################
# Bend
########################################
env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
line    = env.new_line(name = 'line', components=[
    env.new('srot', xt.SRotation, angle=90, at=1),
    env.new('bend', xt.Bend, k0 = 1E-3, h = 1E-3, length = 0.5, at=2),
    env.new('end', xt.Marker, at = 3)])
line.configure_bend_model(edge = 'suppressed')
# line.config.XTRACK_USE_EXACT_DRIFTS = True

sv, tw = madpoint_twiss_survey(line)
summary_plot(sv, tw, 'SRot + Horizontal Bend')

################################################################################
# Multiple
################################################################################

########################################
# Rotation
########################################
env = xt.Environment(particle_ref=xt.Particles(p0c=1e9))
line    = env.new_line(
        name = 'line',
        components= [env.new('drift', xt.Drift, length = 1E-3)] +\
                    [env.new('xrotation', xt.XRotation, angle = 1E-2),
                    env.new('drift', xt.Drift, length = 1E-3)]*1000 +\
                    [env.new('xyshift', xt.XYShift, dy = 1E-2)] +\
                    [env.new('xrotation', xt.XRotation, angle = 1E-2),
                    env.new('drift', xt.Drift, length = 1E-3)]*1000 +\
                    [env.new('end', xt.Marker)])
line.configure_bend_model(edge = 'suppressed')
sv, tw = madpoint_twiss_survey(line)
summary_plot(sv, tw, 'Check that XYShift is suitably rotated')

# TODO: Replica
# TODO: Reverse


plt.show()