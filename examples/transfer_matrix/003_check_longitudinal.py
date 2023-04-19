import numpy as np

import xtrack as xt
import xpart as xp

line = xt.Line.from_json(
    '../../test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json')
line.build_tracker()

configuration = 'above transition'
model = 'linear'

if configuration == 'above transition':
    line['acta.31637'].lag = 180.
    line.particle_ref = xp.Particles(p0c=450e9, q0=1.0)
else:
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

if model == 'nonlinear':
    matrix = xt.SimplifiedAcceleratorSegment(
        qx=tw.qx, qy=tw.qy,
        dqx=tw.dqx, dqy=tw.dqy,
        betx=tw.betx[0], alfx=tw.alfx[0],
        bety=tw.bety[0], alfy=tw.alfy[0],
        dx=tw.dx[0], dpx=tw.dpx[0],
        dy=tw.dy[0], dpy=tw.dpy[0],
        voltage_rf=line['acta.31637'].voltage,
        frequency_rf=line['acta.31637'].frequency,
        lag_rf=line['acta.31637'].lag,
        momentum_compaction_factor=tw.momentum_compaction_factor,
        length=circumference)
elif model == 'linear':
    matrix = xt.SimplifiedAcceleratorSegment(
        qx=tw.qx, qy=tw.qy,
        dqx=tw.dqx, dqy=tw.dqy,
        betx=tw.betx[0], alfx=tw.alfx[0],
        bety=tw.bety[0], alfy=tw.alfy[0],
        dx=tw.dx[0], dpx=tw.dpx[0],
        dy=tw.dy[0], dpy=tw.dpy[0],
        bets=bet_s, qs=qs,
        length=circumference)

line_matrix = xt.Line(elements=[matrix])
line_matrix.particle_ref = line.particle_ref.copy()

line_matrix.build_tracker()
line_matrix.track(particle0.copy(), num_turns=500, turn_by_turn_monitor=True)
mon_matrix = line_matrix.record_last_track

# Match Gaussian distributions
p_line = xp.generate_matched_gaussian_bunch(num_particles=1000000,
    nemitt_x=1e-6, nemitt_y=1e-6, sigma_z=5e-2, line=line, engine='linear')
p_matrix = xp.generate_matched_gaussian_bunch(num_particles=1000000,
    nemitt_x=1e-6, nemitt_y=1e-6, sigma_z=5e-2, line=line_matrix, engine='linear')

assert np.isclose(np.std(p_line.zeta), np.std(p_matrix.zeta), rtol=1e-2)
assert np.isclose(np.std(p_line.pzeta), np.std(p_matrix.pzeta), rtol=2e-2)
assert np.isclose(np.std(p_line.x), np.std(p_matrix.x), rtol=1e-2)
assert np.isclose(np.std(p_line.px), np.std(p_matrix.px), rtol=1e-2)
assert np.isclose(np.std(p_line.y), np.std(p_matrix.y), rtol=1e-2)
assert np.isclose(np.std(p_line.py), np.std(p_matrix.py), rtol=1e-2)

tw_line = line.twiss()
tw_matrix = line_matrix.twiss()

if configuration == 'above transition':
    assert tw_line.betz0 < 0
    assert tw_matrix.betz0 < 0
elif configuration == 'below transition':
    assert tw_line.betz0 > 0
    assert tw_matrix.betz0 > 0
else:
    raise ValueError('Unknown configuration')

assert np.isclose(np.mod(tw_line.qx, 1), np.mod(tw_matrix.qx, 1), atol=1e-5, rtol=0)
assert np.isclose(np.mod(tw_line.qy, 1), np.mod(tw_matrix.qy, 1), atol=1e-5, rtol=0)

# TO BE CONTINUED....





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