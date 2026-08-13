import xtrack as xt
import xobjects as xo
import xtrack.synctime as st
import numpy as np
from scipy.constants import c as clight

turns = 1000
n_collective_elements = 10
p_delta_max = 0.01
p_delta = [p_delta_max, 0.0, -p_delta_max]
# p_delta = [0,0,0]

line = xt.load('../../test_data/psb_injection/line_and_particle.json')

tw = line.twiss4d()
tw_plus = line.twiss4d(delta0=p_delta_max)
tw_minus = line.twiss4d(delta0=-p_delta_max)
circum = tw.line_length


# Install evenly spaced markers to:
# 1) Mimic collective elements for synctime.
# 2) Be replaced by WrapZeta elements.
# evenly_spaced_s = np.linspace(0, line.get_length(), n_collective_elements)
evenly_spaced_s = np.array([0])
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

arrival_times = []
for i_part in range(len(p_delta)):

    mask_active = state_log[:, i_part] > 0
    arrival_time = -zeta_log[mask_active, i_part] / beta0 / clight + np.arange(turns)[mask_active] * t_sim
    arrival_times.append(arrival_time)

expected_arrival_time_on_momentum = np.arange(turns) * t_rev0

import matplotlib.pyplot as plt
plt.close("all")
plt.figure(1)
for ii in range(len(p_delta)):
    plt.plot(np.abs(arrival_times[ii]-expected_arrival_time_on_momentum[:len(arrival_times[ii])])/t_rev0, label=f"delta={p_delta[ii]}")
plt.xlabel("Pass")
plt.ylabel("Delay with respect to on momentum particle [T_0]")
plt.legend()

plt.figure(2)
for ii in range(len(p_delta)):
    plt.plot(arrival_times[ii], label=f"delta={p_delta[ii]}")

plt.plot(expected_arrival_time_on_momentum, label="expected arrival time on momentum")
plt.xlabel("Pass")
plt.ylabel("Arrival time [s]")


i_ref = 1
is_ahead = np.zeros_like(zeta_log, dtype=int)
for i_part in range(len(p_delta)):
    is_ahead[:, i_part] = np.int64(zeta_log[:, i_part] > zeta_log[:, i_ref])



i_obs1 = 0
i_obs2 = 2
t = np.arange(turns) * t_sim
plt.figure(3)
ax1 = plt.subplot(4,1,1)
plt.plot(t / t_rev0, state_log[:, i_ref], label=f'delta={p_delta[i_ref]}', color='k')
plt.plot(t / t_rev0, state_log[:, i_obs1], label=f'delta={p_delta[i_obs1]}')
plt.plot(t / t_rev0, state_log[:, i_obs2], label=f'delta={p_delta[i_obs2]}')

plt.legend()
ax2 = plt.subplot(4,1,2, sharex=ax1)
plt.plot(t / t_rev0, zeta_log[:, i_ref], label=f"zeta delta={p_delta[i_ref]}", color='k')
plt.plot(t / t_rev0, zeta_log[:, i_obs1], label=f"zeta delta={p_delta[i_obs1]}")
plt.plot(t / t_rev0, zeta_log[:, i_obs2], label=f"zeta delta={p_delta[i_obs2]}")
plt.ylabel("zeta [m]")
ax3 = plt.subplot(4,1,3, sharex=ax1)
plt.plot(t / t_rev0, at_turn_log[:, i_ref], color='k')
plt.plot(t / t_rev0, at_turn_log[:, i_obs1])
plt.plot(t / t_rev0, at_turn_log[:, i_obs2])
ax4 = plt.subplot(4,1,4, sharex=ax1)
plt.plot(t / t_rev0, at_turn_log[:, i_obs1]-at_turn_log[:, i_ref],
        color='C0', alpha=0.4, label="at_turn difference")
plt.plot(t / t_rev0, at_turn_log[:, i_obs1]-at_turn_log[:, i_ref] + is_ahead[:, i_obs1], color='C0')
plt.plot(t / t_rev0, at_turn_log[:, i_obs2]-at_turn_log[:, i_ref],
        color='C1', alpha=0.4, label="at_turn difference")
plt.plot(t / t_rev0, at_turn_log[:, i_obs2]-at_turn_log[:, i_ref] + is_ahead[:, i_obs2], color='C1')
plt.xlabel("Time [T0]")
plt.ylabel("at_turn difference")
plt.show()