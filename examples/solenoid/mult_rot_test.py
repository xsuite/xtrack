"""
Test implementation of the multipole shifts in solenoid slices
=============================================
Author(s): John P T Salvesen
Email:  john.salvesen@cern.ch
Date:   18-11-2024
"""
################################################################################
# Required Modules
################################################################################
import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# User Parameters
################################################################################
N_SLICES    = int(1E2)
K0          = 1E-3
L_SOL       = 1
XING_RAD    = 1E-3

BETX        = 100E-3
BETY        = 1E-3

################################################################################
# Build Test Lines
################################################################################

########################################
# Build Environment
########################################
env     = xt.Environment(particle_ref = xt.Particles(p0c = 1E9))

########################################
# Line (beamline frame)
########################################
bl_components_in    = [env.new('bl_drift0', xt.Drift, length  = 1)]
bl_components_out   = [env.new('bl_drift1', xt.Drift, length  = 1)]

bl_components_sol = [
    env.new(f'bl_sol.{i}',  xt.Solenoid,
            length              = (L_SOL / N_SLICES),
            ks                  = 0,
            knl                 = [K0 * (L_SOL / N_SLICES), 0, 0],
            num_multipole_kicks = 1)
    for i in range(N_SLICES)]

bl_line    = env.new_line(
    components  = bl_components_in + bl_components_sol + bl_components_out)

########################################
# Line (horizontal rotated frame)
########################################
hrot_components_in = [
    env.new('hrot_drift0',       xt.Drift,       length  = 1),
    env.new('hshift_in',     xt.XYShift,     dx      = np.sin(XING_RAD) * L_SOL / 2),
    env.new('hrot_in',       xt.YRotation,   angle   = -np.rad2deg(XING_RAD))]

hrot_components_out = [
    env.new('hrot_out',      xt.YRotation,   angle   = np.rad2deg(XING_RAD)),
    env.new('hshift_out',    xt.XYShift,     dx      = np.sin(XING_RAD) * L_SOL / 2),
    env.new('hrot_drift1',       xt.Drift,       length  = 1)]

hrot_components_sol = [
    env.new(f'hrot_sol.{i}',  xt.Solenoid,
            length              = (L_SOL / N_SLICES) * np.cos(XING_RAD),
            ks                  = 0,
            knl                 = [K0 * (L_SOL / N_SLICES), 0, 0],
            num_multipole_kicks = 1,
            mult_rot_y_rad      = XING_RAD,
            mult_shift_x        = np.sin(XING_RAD) * L_SOL * (i/N_SLICES - 1/2))
    for i in range(N_SLICES)]

hrot_line    = env.new_line(
    components  = hrot_components_in + hrot_components_sol + hrot_components_out)

########################################
# Line (vertical rotated frame)
########################################
vrot_components_in = [
    env.new('vrot_drift0',   xt.Drift,       length  = 1),
    env.new('vshift_in',     xt.XYShift,     dy      = np.sin(XING_RAD) * L_SOL / 2),
    env.new('vrot_in',       xt.XRotation,   angle   = np.rad2deg(XING_RAD))]
# TODO: Minus sign difference here as still inconsistent definition with XRotation and YRotation
vrot_components_out = [
    env.new('vrot_out',      xt.XRotation,   angle   = -np.rad2deg(XING_RAD)),
    env.new('vshift_out',    xt.XYShift,     dy      = np.sin(XING_RAD) * L_SOL / 2),
    env.new('vrot_drift1',   xt.Drift,       length  = 1)]

vrot_components_sol = [
    env.new(f'vrot_sol.{i}',  xt.Solenoid,
            length              = (L_SOL / N_SLICES) * np.cos(XING_RAD),
            ks                  = 0,
            knl                 = [K0 * (L_SOL / N_SLICES), 0, 0],
            num_multipole_kicks = 1,
            mult_rot_x_rad      = XING_RAD,
            mult_shift_y        = np.sin(XING_RAD) * L_SOL * (i/N_SLICES - 1/2))
    for i in range(N_SLICES)]

vrot_line    = env.new_line(
    components  = vrot_components_in + vrot_components_sol + vrot_components_out)

################################################################################
# Comparisons
################################################################################
bl_twiss   = bl_line.twiss(
    method  = '4d',
    start   = xt.START,
    end     = xt.END,
    betx    = BETX,
    bety    = BETY)

hrot_twiss   = hrot_line.twiss(
    method  = '4d',
    start   = xt.START,
    end     = xt.END,
    betx    = BETX,
    bety    = BETY)

vrot_twiss   = vrot_line.twiss(
    method  = '4d',
    start   = xt.START,
    end     = xt.END,
    betx    = BETX,
    bety    = BETY)

bl_twiss.plot()
plt.title('Beamline Frame')
bl_twiss.plot('x y')
plt.title('Beamline Frame')

hrot_twiss.plot()
plt.title('Horizontal Rotated Frame')
hrot_twiss.plot('x y')
plt.title('Horizontal Rotated Frame')

vrot_twiss.plot()
plt.title('Vertical Rotated Frame')
vrot_twiss.plot('x y')
plt.title('Vertical Rotated Frame')

################################################################################
# Test Assertions
################################################################################
# Tolerances lower for derivative quantities (alfx, alfy, dpx, dpy)
assert np.isclose(bl_twiss['x'][-1],    hrot_twiss['x'][-1],    rtol = 1E-6)
assert np.isclose(bl_twiss['y'][-1],    hrot_twiss['y'][-1],    rtol = 1E-6)
assert np.isclose(bl_twiss['betx'][-1], hrot_twiss['betx'][-1], rtol = 1E-6)
assert np.isclose(bl_twiss['bety'][-1], hrot_twiss['bety'][-1], rtol = 1E-6)
assert np.isclose(bl_twiss['alfx'][-1], hrot_twiss['alfx'][-1], rtol = 1E-4)
assert np.isclose(bl_twiss['alfy'][-1], hrot_twiss['alfy'][-1], rtol = 1E-4)
assert np.isclose(bl_twiss['dx'][-1],   hrot_twiss['dx'][-1],   rtol = 1E-6)
assert np.isclose(bl_twiss['dy'][-1],   hrot_twiss['dy'][-1],   rtol = 1E-6)
assert np.isclose(bl_twiss['dpx'][-1],  hrot_twiss['dpx'][-1],  rtol = 1E-4)
assert np.isclose(bl_twiss['dpy'][-1],  hrot_twiss['dpy'][-1],  rtol = 1E-4)
assert np.isclose(bl_twiss['mux'][-1],  hrot_twiss['mux'][-1],  rtol = 1E-6)
assert np.isclose(bl_twiss['muy'][-1],  hrot_twiss['muy'][-1],  rtol = 1E-6)

assert np.isclose(bl_twiss['x'][-1],    vrot_twiss['x'][-1],    rtol = 1E-6)
assert np.isclose(bl_twiss['y'][-1],    vrot_twiss['y'][-1],    rtol = 1E-6)
assert np.isclose(bl_twiss['betx'][-1], vrot_twiss['betx'][-1], rtol = 1E-6)
assert np.isclose(bl_twiss['bety'][-1], vrot_twiss['bety'][-1], rtol = 1E-4)
assert np.isclose(bl_twiss['alfx'][-1], vrot_twiss['alfx'][-1], rtol = 1E-4)
assert np.isclose(bl_twiss['alfy'][-1], vrot_twiss['alfy'][-1], rtol = 1E-6)
assert np.isclose(bl_twiss['dx'][-1],   vrot_twiss['dx'][-1],   rtol = 1E-6)
assert np.isclose(bl_twiss['dy'][-1],   vrot_twiss['dy'][-1],   rtol = 1E-6)
assert np.isclose(bl_twiss['dpx'][-1],  vrot_twiss['dpx'][-1],  rtol = 1E-4)
assert np.isclose(bl_twiss['dpy'][-1],  vrot_twiss['dpy'][-1],  rtol = 1E-4)
assert np.isclose(bl_twiss['mux'][-1],  vrot_twiss['mux'][-1],  rtol = 1E-6)
assert np.isclose(bl_twiss['muy'][-1],  vrot_twiss['muy'][-1],  rtol = 1E-6)

########################################
# Show Plots
########################################
plt.show()
