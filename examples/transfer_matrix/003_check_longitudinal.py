import numpy as np

import xtrack as xt
import xpart as xp

line = xt.Line.from_json(
    '../../test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json')
line.build_tracker()

# configuration = 'above transition'
# line['acta.31637'].lag = 180.
# line.particle_ref = xp.Particles(p0c=450e9, q0=1.0)

configuration = 'below transition'
line['acta.31637'].lag = 0.
line.particle_ref = xp.Particles(p0c=14e9, q0=1.0)

particle0 = line.build_particles(x_norm=0, y_norm=0, zeta=1e-3)


line.track(particle0.copy(), num_turns=500, turn_by_turn_monitor=True)
mon = line.record_last_track

# Build corresponding matrix
tw = line.twiss()
eta = tw.slip_factor # > 0 above transition
qs = tw.qs
circumference = line.get_length()

bet_s = eta * circumference / (2 * np.pi * qs)

# matrix = xt.LinearTransferMatrix(beta_s=bet_s, Q_s=qs)
matrix = xt.LinearTransferMatrix(
    voltage_rf=line['acta.31637'].voltage,
    frequency_rf=line['acta.31637'].frequency,
    lag_rf=line['acta.31637'].lag,
    momentum_compaction_factor=tw.momentum_compaction_factor,
    length=circumference)
line_matrix = xt.Line(elements=[matrix])
line_matrix.particle_ref = line.particle_ref.copy()

line_matrix.build_tracker()
line_matrix.track(particle0.copy(), num_turns=500, turn_by_turn_monitor=True)
mon_matrix = line_matrix.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
fig1 = plt.figure(1)
fig1.suptitle(configuration)
ax1 = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212, sharex=ax1)
ax1.set_ylabel('zeta')
ax2.set_ylabel('pzeta')
ax2.set_xlabel('turn')
ax1.plot(mon.zeta.T, label='lattice')
ax1.plot(mon_matrix.zeta.T, label='matrix')
ax2.plot(mon.pzeta.T)
ax2.plot(mon_matrix.pzeta.T)
ax1.legend()

fig1.subplots_adjust(left=0.2)

particles_dp0 = line.build_particles(x_norm=0, y_norm=0, 
        delta=np.linspace(-5e-3, 5e-3, 41))
line_matrix.track(particles_dp0.copy(), num_turns=500, turn_by_turn_monitor=True)
mon_matrix_dp = line_matrix.record_last_track

fig2 = plt.figure(2)
fig2.suptitle(configuration)
plt.plot(mon_matrix_dp.zeta.T, mon_matrix_dp.pzeta.T)

plt.show()