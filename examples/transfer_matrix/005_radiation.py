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
energy = 45.6

gamma_x = (1.0 + alpha_x**2) / beta_x
gamma_y = (1.0 + alpha_y**2) / beta_y

damping_rate_x = 5e-4
damping_rate_y = 1e-3
damping_rate_s = 2e-3

equ_emit_x = 0.3e-9
equ_emit_y = 1e-12
equ_length = 3.5e-3
equ_delta = 3.8e-4
beta_s = equ_length / equ_delta
equ_emit_s = equ_length * equ_delta

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
    damping_rate_x=damping_rate_x,
    damping_rate_y=damping_rate_y,
    damping_rate_s=damping_rate_s,
    equ_emit_x=equ_emit_x,
    equ_emit_y=equ_emit_y,
    equ_emit_s=equ_emit_s,
)

part = xt.Particles(
    _context=context,
    x=[10 * np.sqrt(equ_emit_x * beta_x)],
    y=[10 * np.sqrt(equ_emit_y * beta_y)],
    zeta=[10 * np.sqrt(equ_emit_s * beta_s)],
    p0c=energy * 1e9,
)

part._init_random_number_generator()

n_turns = int(1e5)
x = np.zeros(n_turns, dtype=float)
px = np.zeros_like(x)
y = np.zeros_like(x)
py = np.zeros_like(x)
z = np.zeros_like(x)
delta = np.zeros_like(x)
emit_x = np.zeros_like(x)
emit_y = np.zeros_like(x)
emit_s = np.zeros_like(x)
for turn in range(n_turns):
    x[turn] = part.x[0]
    px[turn] = part.px[0]
    y[turn] = part.y[0]
    py[turn] = part.py[0]
    z[turn] = part.zeta[0]
    delta[turn] = part.delta[0]
    emit_x[turn] = 0.5 * (
        gamma_x * part.x[0] ** 2
        + 2 * alpha_x * part.x[0] * part.px[0]
        + beta_x * part.px[0] ** 2
    )
    emit_y[turn] = 0.5 * (
        gamma_y * part.y[0] ** 2
        + 2 * alpha_y * part.y[0] * part.py[0]
        + beta_y * part.py[0] ** 2
    )
    emit_s[turn] = 0.5 * (part.zeta[0] ** 2 / beta_s + beta_s * part.delta[0] ** 2)
    el.track(part)

plt.figure(0)
plt.plot(x, px, "x")
plt.figure(1)
plt.plot(y, py, "x")
plt.figure(2)
plt.plot(z, delta, "x")

turns = np.arange(n_turns)
fit_range = 1000
fit_x = linregress(turns[:fit_range], np.log(emit_x[:fit_range]))
fit_y = linregress(turns[:fit_range], np.log(emit_y[:fit_range]))
fit_s = linregress(turns[:fit_range], np.log(emit_s[:fit_range]))


print(
    -fit_x.slope / damping_rate_x,
    -fit_y.slope / damping_rate_y,
    -fit_s.slope / damping_rate_s,
)
averga_start = int(3e4)
emit_x_0 = np.average(emit_x[averga_start:])
emit_y_0 = np.average(emit_y[averga_start:])
emit_s_0 = np.average(emit_s[averga_start:])
length_0 = np.std(z[averga_start:])

# Definition of damping rate
eps_x_expected = emit_x[0] * np.exp(-damping_rate_x * turns)
eps_y_expected = emit_y[0] * np.exp(-damping_rate_y * turns)
eps_s_expected = emit_s[0] * np.exp(-damping_rate_s * turns)

print(
    emit_x_0 / equ_emit_x,
    emit_y_0 / equ_emit_y,
    emit_s_0 / equ_emit_s,
    length_0 / equ_length,
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
