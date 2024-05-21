# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

import xobjects as xo
import xtrack as xt

context = xo.ContextCpu()

q_x_set = 0.18
q_y_set = 0.22
Q_s_set = 0.025

beta_x = 1.0
beta_y = 10.0
alpha_x = 0.0
alpha_y = 0.0
beta_s = 12.244
energy = 45.6

gamma_x = (1.0 + alpha_x**2) / beta_x
gamma_y = (1.0 + alpha_y**2) / beta_y

# radiation properties
damping_rate_h = 5e-4 # horizontal emittance damping rate
damping_rate_v = 1e-3 # vertical emittance damping rate
damping_rate_s = 2e-3 # longitudinal emittance damping rate
equ_emit_x = 0.3e-9
equ_emit_y = 1e-12
equ_length = 3.5e-3
equ_delta = 3.8e-4
beta_s = equ_length / equ_delta
equ_emit_s = equ_length * equ_delta

############################################
# setting up diagonal terms of
# the damping and diffusion matrix
# based on radiation properties
damping_rate_x = damping_rate_h/2
damping_rate_px = damping_rate_h/2
damping_rate_y = damping_rate_v/2
damping_rate_py = damping_rate_v/2
damping_rate_pzeta = damping_rate_s

if True:
    gauss_noise_ampl_px = np.sqrt(equ_emit_x*damping_rate_h/beta_x)
    gauss_noise_ampl_x = beta_x*gauss_noise_ampl_px
    gauss_noise_ampl_py = np.sqrt(equ_emit_y*damping_rate_v/beta_y)
    gauss_noise_ampl_y = beta_y*gauss_noise_ampl_py
    gauss_noise_ampl_delta = np.sqrt(2*equ_emit_s*damping_rate_s/beta_s)
else:
    gauss_noise_ampl_px = 0.0
    gauss_noise_ampl_x = 0.0
    gauss_noise_ampl_py = 0.0
    gauss_noise_ampl_y = 0.0
    gauss_noise_ampl_delta = 0.0
#############################################

el = xt.LineSegmentMap(
    _context=context,
    qx=q_x_set,
    qy=q_y_set,
    qs=Q_s_set,
    bets=beta_s,
    betx=beta_x,
    bety=beta_y,
    alfx=alpha_x,
    alfy=alpha_y,
    damping_rate_x=damping_rate_x,damping_rate_px=damping_rate_px,
    damping_rate_y=damping_rate_y,damping_rate_py=damping_rate_py,
    damping_rate_zeta=0.0,damping_rate_pzeta=damping_rate_pzeta,
    gauss_noise_ampl_x = gauss_noise_ampl_x, gauss_noise_ampl_px = gauss_noise_ampl_px, 
    gauss_noise_ampl_y = gauss_noise_ampl_y, gauss_noise_ampl_py = gauss_noise_ampl_py,
    gauss_noise_ampl_zeta = 0.0, gauss_noise_ampl_pzeta = gauss_noise_ampl_delta,
)

part = xt.Particles(
    _context=context,
    x=[10 * np.sqrt(equ_emit_x * beta_x)],
    y=[10 * np.sqrt(equ_emit_y * beta_y)],
    pzeta=[10 * np.sqrt(equ_delta / beta_s)],
    p0c=energy * 1e9,
)

part._init_random_number_generator()

line = xt.Line(elements=[el])
line.build_tracker()
print('Tracking')
n_turns = int(1E5)
line.track(part,num_turns=n_turns,turn_by_turn_monitor=True)
print('Done tracking')

emit_x = 0.5 * (
    gamma_x * line.record_last_track.x[0,:] ** 2
    + 2 * alpha_x * line.record_last_track.x[0,:] * line.record_last_track.px[0,:]
    + beta_x * line.record_last_track.px[0,:] ** 2
)
emit_y = 0.5 * (
    gamma_y * line.record_last_track.y[0,:] ** 2
    + 2 * alpha_y * line.record_last_track.y[0,:] * line.record_last_track.py[0,:]
    + beta_y * line.record_last_track.py[0,:] ** 2
)
emit_s = 0.5 * (line.record_last_track.zeta[0,:] ** 2 / beta_s + beta_s * line.record_last_track.delta[0,:] ** 2)

if False:
    plt.figure(0)
    plt.plot(line.record_last_track.x[0,:], line.record_last_track.px[0,:], "x")
    plt.figure(1)
    plt.plot(line.record_last_track.y[0,:], line.record_last_track.py[0,:], "x")
    plt.figure(2)
    plt.plot(line.record_last_track.z[0,:], line.record_last_track.delta[0,:], "x")

turns = np.arange(n_turns)
fit_range = 1000
fit_x = linregress(turns[:fit_range], np.log(emit_x[:fit_range]))
fit_y = linregress(turns[:fit_range], np.log(emit_y[:fit_range]))
fit_s = linregress(turns[:fit_range], np.log(emit_s[:fit_range]))


print(
    -fit_x.slope / damping_rate_h,
    -fit_y.slope / damping_rate_v,
    -fit_s.slope / damping_rate_s,
)
averga_start = int(3e4)
emit_x_0 = np.average(emit_x[averga_start:])
emit_y_0 = np.average(emit_y[averga_start:])
emit_s_0 = np.average(emit_s[averga_start:])

# Definition of emittance damping rate
eps_x_expected = emit_x[0] * np.exp(-damping_rate_h * turns)
eps_y_expected = emit_y[0] * np.exp(-damping_rate_v * turns)
eps_s_expected = emit_s[0] * np.exp(-damping_rate_s * turns)

print(
    emit_x_0 / equ_emit_x,
    emit_y_0 / equ_emit_y,
    emit_s_0 / equ_emit_s
)

plt.figure(10)
sp1 = plt.subplot(311)
plt.plot(turns, emit_x)
plt.plot(turns, np.exp(fit_x.intercept + fit_x.slope * turns), "--k", label="fit")
plt.plot(turns, eps_x_expected, "--r", label="expected")
plt.plot([turns[0], turns[-1]], [emit_x_0, emit_x_0], "--g", label="equilibrium")
plt.ylabel(r"$\epsilon_x$ [m]")
plt.legend()

plt.subplot(312, sharex=sp1)
plt.plot(turns, emit_y)
plt.plot(turns, np.exp(fit_y.intercept + fit_y.slope * turns), "--k")
plt.plot(turns, eps_y_expected, "--r")
plt.plot([turns[0], turns[-1]], [emit_y_0, emit_y_0], "--g")
plt.ylabel(r"$\epsilon_y$ [m]")

plt.subplot(313, sharex=sp1)
plt.plot(turns, emit_s)
plt.plot(turns, np.exp(fit_s.intercept + fit_s.slope * turns), "--k")
plt.plot(turns, eps_s_expected, "--r")
plt.plot([turns[0], turns[-1]], [emit_s_0, emit_s_0], "--g")
plt.ylabel(r"$\epsilon_s$ [m]")
plt.xlabel("turn")

plt.show()
