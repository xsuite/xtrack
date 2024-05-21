# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from matplotlib import pyplot as plt

import xobjects as xo
import xtrack as xt

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
damping_rate_h = 5e-4  # horizontal emittance damping rate
damping_rate_v = 1e-3  # vertical emittance damping rate
damping_rate_s = 2e-3  # longitudinal emittance damping rate
equ_emit_x = 0.3e-9
equ_emit_y = 1e-12
equ_length = 3.5e-3
equ_delta = 3.8e-4
beta_s = equ_length / equ_delta
equ_emit_s = equ_length * equ_delta

# Setting up diagonal terms of the damping and diffusion matrix based
# on radiation properties
damping_rate_x = damping_rate_h/2
damping_rate_px = damping_rate_h/2
damping_rate_y = damping_rate_v/2
damping_rate_py = damping_rate_v/2
damping_rate_pzeta = damping_rate_s

gauss_noise_ampl_px = np.sqrt(equ_emit_x*damping_rate_h/beta_x)
gauss_noise_ampl_x = beta_x*gauss_noise_ampl_px
gauss_noise_ampl_py = np.sqrt(equ_emit_y*damping_rate_v/beta_y)
gauss_noise_ampl_y = beta_y*gauss_noise_ampl_py
gauss_noise_ampl_delta = np.sqrt(2*equ_emit_s*damping_rate_s/beta_s)

# Build the map
ring_map = xt.LineSegmentMap(
    qx=q_x_set,
    qy=q_y_set,
    qs=Q_s_set,
    bets=beta_s,
    betx=beta_x,
    bety=beta_y,
    alfx=alpha_x,
    alfy=alpha_y,
    damping_rate_x=damping_rate_x,
    damping_rate_px=damping_rate_px,
    damping_rate_y=damping_rate_y,
    damping_rate_py=damping_rate_py,
    damping_rate_zeta=0.0,
    damping_rate_pzeta=damping_rate_pzeta,
    gauss_noise_ampl_x=gauss_noise_ampl_x,
    gauss_noise_ampl_px=gauss_noise_ampl_px,
    gauss_noise_ampl_y=gauss_noise_ampl_y,
    gauss_noise_ampl_py=gauss_noise_ampl_py,
    gauss_noise_ampl_zeta=0.0,
    gauss_noise_ampl_pzeta=gauss_noise_ampl_delta,
)

# Build particles (all at the same position)
num_particles = 1000
part = xt.Particles(
    x=num_particles * [2 * np.sqrt(equ_emit_x * beta_x)],
    y=num_particles * [2 * np.sqrt(equ_emit_y * beta_y)],
    pzeta=num_particles * [0.15 * np.sqrt(equ_delta / beta_s)],
    p0c=energy * 1e9,
)
# Initialize random number generator
part._init_random_number_generator()

# Build line and tracker
line = xt.Line(elements=[ring_map])
line.build_tracker()

# Track!
num_turns = 40_000
line.track(part, num_turns=num_turns,
           turn_by_turn_monitor=True, with_progress=True)

# Compute emittances from tracking data
emit_x = 0.5 * (
    gamma_x * line.record_last_track.x ** 2
    + 2 * alpha_x *
    line.record_last_track.x * line.record_last_track.px[0, :]
    + beta_x * line.record_last_track.px ** 2
).mean(axis=0)
emit_y = 0.5 * (
    gamma_y * line.record_last_track.y ** 2
    + 2 * alpha_y *
    line.record_last_track.y * line.record_last_track.py[0, :]
    + beta_y * line.record_last_track.py ** 2
).mean(axis=0)
emit_s = 0.5 * (line.record_last_track.zeta ** 2 /
                beta_s + beta_s * line.record_last_track.delta ** 2
                ).mean(axis=0)

# Get equilibrium emittances by averaging last turns
averga_start = 30_000
emit_x_0 = np.average(emit_x[averga_start:])
emit_y_0 = np.average(emit_y[averga_start:])
emit_s_0 = np.average(emit_s[averga_start:])

# Expected emittance evolution
turns = np.arange(num_turns)
eps_x_expected = (emit_x[0] - emit_x_0) * \
    np.exp(-damping_rate_h * turns) + emit_x_0
eps_y_expected = (emit_y[0] - emit_y_0) * \
    np.exp(-damping_rate_v * turns) + emit_y_0
eps_s_expected = (emit_s[0] - emit_s_0) * \
    np.exp(-damping_rate_s * turns) + emit_s_0

# Plot
plt.close("all")

plt.figure(1)
sp1 = plt.subplot(311)
plt.plot(turns, emit_x)
plt.plot(turns, eps_x_expected, "--r", label="expected")
plt.axhline(y=emit_x_0, linestyle='--', color='g', label="equilibrium")
plt.ylabel(r"$\epsilon_x$ [m]")
plt.ylim(bottom=0)
plt.legend()

plt.subplot(312, sharex=sp1)
plt.plot(turns, emit_y)
plt.plot(turns, eps_y_expected, "--r")
plt.plot([turns[0], turns[-1]], [emit_y_0, emit_y_0], "--g")
plt.ylim(bottom=0)
plt.ylabel(r"$\epsilon_y$ [m]")

plt.subplot(313, sharex=sp1)
plt.plot(turns, emit_s)
plt.plot(turns, eps_s_expected, "--r")
plt.plot([turns[0], turns[-1]], [emit_s_0, emit_s_0], "--g")
plt.ylabel(r"$\epsilon_s$ [m]")
plt.ylim(bottom=0)
plt.xlabel("turn")

plt.show()
