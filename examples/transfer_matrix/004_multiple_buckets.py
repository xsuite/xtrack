# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import time

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants

import xobjects as xo
context = xo.ContextCpu(omp_num_threads=0)
import xtrack as xt
import xfields as xf
import xpart as xp

n_macroparticles = int(1e3)
nemitt_x = 2E-6
nemitt_y = 2E-6
p0c = 7e12
gamma = p0c/xp.PROTON_MASS_EV
physemit_x = nemitt_x/gamma
physemit_y = nemitt_y/gamma
beta_x = 1.0
beta_y = 1.0

sigma_z = 0.08
momentumCompaction = 3.483575072011584e-04
eta = momentumCompaction - 1.0 / gamma ** 2
circumference = 26658.883199999
frev = constants.c/circumference
f_RF = 400.8E6
h = f_RF / frev
betar = np.sqrt(1 - 1 / gamma ** 2)
p0 = constants.m_p * betar * gamma * constants.c
voltage = 16E6
Qs = np.sqrt(constants.e * voltage * eta * h / (2 * np.pi * betar * constants.c * p0))
beta_s = eta * circumference / (2 * np.pi * Qs);
sigma_delta = sigma_z / beta_s
Qx = 0.31
Qy = 0.32
bucket_length = 2.5E-9

linear_fixed_rf = True

particles = xp.Particles(_context=context,
    p0c=p0c,
    x=np.zeros(n_macroparticles),
    px=np.zeros(n_macroparticles),
    y=np.zeros(n_macroparticles),
    py=np.zeros(n_macroparticles),
    zeta=bucket_length*constants.c*betar*np.linspace(-5,5,n_macroparticles),
    delta=np.zeros(n_macroparticles),
)

if linear_fixed_rf:
    arc = xt.LineSegmentMap(
            betx=beta_x,qx=Qx,
            bety=beta_y,qy=Qy,
            longitudinal_mode='linear_fixed_rf',
            voltage_rf = voltage*10,
            frequency_rf = f_RF/10,
            lag_rf = 180.0,
            slippage_length = circumference,
            momentum_compaction_factor = momentumCompaction)
else:
    arc = xt.LineSegmentMap(
            betx=beta_x,qx=Qx,
            bety=beta_y,qy=Qy,
            bets=-beta_s,qs=Qs)

line = xt.Line(elements=[arc])
line.build_tracker()
line.track(particles,num_turns=int(1/Qs),turn_by_turn_monitor=True)
mon = line.record_last_track

plt.figure()
plt.plot(mon.zeta/(bucket_length*constants.c*betar),mon.delta/sigma_delta,'.r')

particles = xp.Particles(_context=context,
    p0c=p0c,
    x=np.zeros(n_macroparticles),
    px=np.zeros(n_macroparticles),
    y=np.zeros(n_macroparticles),
    py=np.zeros(n_macroparticles),
    zeta=bucket_length*constants.c*betar*np.linspace(-5,5,n_macroparticles),
    delta=np.zeros(n_macroparticles),
)

if linear_fixed_rf:
    arc = xt.LineSegmentMap(
            betx=beta_x,qx=Qx,
            bety=beta_y,qy=Qy,
            longitudinal_mode='linear_fixed_rf',
            voltage_rf = voltage,
            frequency_rf = f_RF,
            lag_rf = 180.0,
            slippage_length = circumference,
            momentum_compaction_factor = momentumCompaction)
else:
    arc = xt.LineSegmentMap(
            betx=beta_x,qx=Qx,
            bety=beta_y,qy=Qy,
            bets=-beta_s,qs=Qs,bucket_length=bucket_length)

line2 = xt.Line(elements=[arc])
line2.build_tracker()
line2.track(particles,num_turns=int(1/Qs),turn_by_turn_monitor=True)
mon = line2.record_last_track

plt.plot(mon.zeta/(bucket_length*constants.c*betar),mon.delta/sigma_delta,'.b')
plt.xlabel(r'$z$ [bucket_lenght$\cdot c\beta_r$]')
plt.ylabel('$\delta$ [$\sigma_{\delta}$]')
plt.show()


