import xtrack as xt
import xobjects as xo
import xtrack.synctime as st
import numpy as np
from scipy.constants import c as clight

turns = 2000
n_collective_elements = 30
p_delta_max = 0.005
p_delta = [p_delta_max, 0.0, -p_delta_max]
# p_delta = [0,0,0]

line = xt.load('../../test_data/psb_injection/line_and_particle.json')
line['br.c02'].voltage = 0.0

tw = line.twiss4d()
tw_plus = line.twiss4d(delta0=p_delta_max)
tw_minus = line.twiss4d(delta0=-p_delta_max)
circum = tw.line_length

eta = tw.slip_factor
# Expected number of turns to gain/lose one turn with respect to on-momentum particle:
n_slip = 1 / abs(eta * p_delta_max)

# Install evenly spaced markers to:
# 1) Mimic collective elements for synctime.
# 2) Be replaced by WrapZeta elements.
evenly_spaced_s = np.linspace(0, line.get_length(), n_collective_elements)
line.cut_at_s(evenly_spaced_s)
for i, s in enumerate(evenly_spaced_s):
    marker = xt.Marker()
    marker.iscollective = True
    line.insert(what=f"marker_{i}", obj=marker, at=s)

st.install_sync_time_at_collective_elements(line=line)
line.enable_time_dependent_vars = True

particles: xt.Particles = line.build_particles(delta=p_delta,
                                               x=tw.dx[0]*np.array(p_delta),
                                               px=tw.dpx[0]*np.array(p_delta))
particles_start = particles.copy()
st.prepare_particles_for_sync_time(particles=particles, line=line)


at_turn_log = []
zeta_log = []
state_log = []
for ii in range(turns):
    if ii % 10 == 0:
        print(f"Turn {ii}        ", end="\r", flush=True)

    i_sorted = np.argsort(particles.particle_id)
    at_turn_log.append(particles.at_turn[i_sorted].copy())
    zeta_log.append(particles.zeta[i_sorted].copy())
    state_log.append(particles.state[i_sorted].copy())

    line.track(particles, num_turns=1)

at_turn_log = np.array(at_turn_log)
zeta_log = np.array(zeta_log)
state_log = np.array(state_log)

beta0 = particles.beta0[0]
t_rev0 = line.get_length() / clight / beta0
t_sim = line['synctime_0'].frame_relative_length * t_rev0

# `at_turn` is to be intended as the number of times the particle has passed
# the start of the ring since the beginning of the simulation.

# If one wants to compute the number ot times a given particle has slipped a
# turn with respect to the on-momentum particle, one can take the difference of
# `at_turn` between the two particles, and add 1 if the particle is ahead of
# the reference particle in zeta (slippage in the present t_sim frame).

import matplotlib.pyplot as plt
plt.close("all")

i_ref = 1
is_ahead = np.zeros_like(zeta_log, dtype=int)
for i_part in range(len(p_delta)):
    is_ahead[:, i_part] = np.int64(zeta_log[:, i_part] > zeta_log[:, i_ref])

i_obs = 0
t = np.arange(turns) * t_sim
plt.figure(1)
ax1 = plt.subplot(3,1,1)
plt.plot(t / t_rev0, state_log[:, i_ref]>0, label=f'delta={p_delta[i_ref]}', color='k', alpha=0.4)
plt.plot(t / t_rev0, state_log[:, i_obs]>0, label=f'delta={p_delta[i_obs]}')
plt.ylabel("Particle alive")

plt.legend()
ax2 = plt.subplot(3,1,2, sharex=ax1)
plt.plot(t / t_rev0, zeta_log[:, i_ref], label=f"zeta delta={p_delta[i_ref]}", color='k', alpha=0.4)
plt.plot(t / t_rev0, zeta_log[:, i_obs], label=f"zeta delta={p_delta[i_obs]}")
plt.ylabel("zeta [m]")

ax4 = plt.subplot(3,1,3, sharex=ax1)
plt.plot(t / t_rev0, at_turn_log[:, i_obs]-at_turn_log[:, i_ref],
        color='C0', alpha=0.2, label="without z-diff correction")
plt.plot(t / t_rev0, at_turn_log[:, i_obs]-at_turn_log[:, i_ref] + is_ahead[:, i_obs],
         color='C0', label="with z-diff correction")
plt.xlabel(r"$t$ / $T_0$")
plt.ylabel(r"$\Delta N_{turns}$")
plt.legend()
for ii in range(7):
    plt.axvline(x=ii*n_slip, color='r', alpha=0.4, ls='--')

# Compute particle arrival time at each pass

arrival_times = []
for i_part in range(len(p_delta)):

    mask_active = state_log[:, i_part] > 0
    arrival_time = (-zeta_log[mask_active, i_part] / beta0 / clight
                    + np.arange(turns)[mask_active] * t_sim)
    arrival_times.append(arrival_time)

expected_arrival_time_on_momentum = np.arange(turns) * t_rev0
plt.suptitle(f'n_collective_elements = {n_collective_elements}')


plt.figure(2)

for ii in range(len(p_delta)):
    plt.plot((arrival_times[ii]-expected_arrival_time_on_momentum[:len(arrival_times[ii])])/t_rev0,
             label=f"delta={p_delta[ii]}")
plt.xlabel("Pass")
plt.ylabel(r"Delay with respect to on-momentum particle ($\Delta t$ / $T_0$)")
plt.legend()
plt.suptitle(f'n_collective_elements = {n_collective_elements}')

plt.show()