import numpy as np
import xtrack as xt

line = xt.Line.from_json('fccee_p_ring_thin.json')
line.build_tracker()

tw_no_rad = line.twiss()

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True)

line.configure_radiation(model='quantum')
p = line.build_particles(num_particles=30)
line.track(p, num_turns=1000, turn_by_turn_monitor=True, time=True)
print(f'Tracking time: {line.time_last_track}')

ex = tw_rad.nemitt_x_rad / (tw_rad.gamma0 * tw_rad.beta0)
ey = tw_rad.nemitt_y_rad / (tw_rad.gamma0 * tw_rad.beta0)
ez = tw_rad.nemitt_zeta_rad / (tw_rad.gamma0 * tw_rad.beta0)

mon = line.record_last_track

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1)
spx = fig. add_subplot(3, 1, 1)
spx.plot(np.std(mon.x, axis=0))
spx.axhline(np.sqrt(ex * tw_rad.betx[0] + ey * tw_rad.betx2[0] + (np.std(p.delta) * tw_rad.dx[0])**2), color='red')
# spx.axhline(np.sqrt(ex_hof * tw.betx[0] + (np.std(p.delta) * tw.dx[0])**2), color='green')

spy = fig. add_subplot(3, 1, 2, sharex=spx)
spy.plot(np.std(mon.y, axis=0))
spy.axhline(np.sqrt(ex * tw_rad.bety1[0] + ey * tw_rad.bety[0] + (np.std(p.delta) * tw_rad.dy[0])**2), color='red')
# spy.axhline(np.sqrt(ey_hof * tw.bety[0] + (np.std(p.delta) * tw.dy[0])**2), color='green')

spz = fig. add_subplot(3, 1, 3, sharex=spx)
spz.plot(np.std(mon.zeta, axis=0))
spz.axhline(np.sqrt(ez * tw_rad.betz0), color='red')

plt.show()

# wig_values = np.linspace(0, 0.5, 11)

# nemitt_x = np.zeros_like(wig_values)
# nemitt_y = np.zeros_like(wig_values)
# b_plus = np.zeros_like(wig_values)

# for ii, wig_val in enumerate(wig_values):
#     print(f'{ii} / {len(wig_values)}')

#     line.vars['on_wiggler_v'] = wig_val
#     tw_rad = line.twiss(eneloss_and_damping=True)

#     nemitt_x[ii] = tw_rad.nemitt_x_rad
#     nemitt_y[ii] = tw_rad.nemitt_y_rad
#     b_plus[ii] = line.vars['b_plus']._value * wig_val

# line.configure_radiation(model='quantum')
# p = line.build_particles(num_particles=30)
# line.track(p, num_turns=1000, turn_by_turn_monitor=True, time=True)
# print(f'Tracking time: {line.time_last_track}')

# ex = tw_rad.nemitt_x_rad / (tw_rad.gamma0 * tw_rad.beta0)
# ey = tw_rad.nemitt_y_rad / (tw_rad.gamma0 * tw_rad.beta0)
# ez = tw_rad.nemitt_zeta_rad / (tw_rad.gamma0 * tw_rad.beta0)


# mon = line.record_last_track

# tw = tw_rad

# import matplotlib.pyplot as plt
# plt.close('all')
# fig = plt.figure(1)
# spx = fig. add_subplot(3, 1, 1)
# spx.plot(np.std(mon.x, axis=0))
# spx.axhline(np.sqrt(ex * tw.betx[0] + ey * tw.betx2[0] + (np.std(p.delta) * tw.dx[0])**2), color='red')
# # spx.axhline(np.sqrt(ex_hof * tw.betx[0] + (np.std(p.delta) * tw.dx[0])**2), color='green')

# spy = fig. add_subplot(3, 1, 2, sharex=spx)
# spy.plot(np.std(mon.y, axis=0))
# spy.axhline(np.sqrt(ex * tw.bety1[0] + ey * tw.bety[0] + (np.std(p.delta) * tw.dy[0])**2), color='red')
# # spy.axhline(np.sqrt(ey_hof * tw.bety[0] + (np.std(p.delta) * tw.dy[0])**2), color='green')

# spz = fig. add_subplot(3, 1, 3, sharex=spx)
# spz.plot(np.std(mon.zeta, axis=0))
# spz.axhline(np.sqrt(ez * tw.betz0), color='red')

# plt.show()