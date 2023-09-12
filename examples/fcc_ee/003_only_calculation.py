import numpy as np
import xtrack as xt

fname = 'fccee_t'

line = xt.Line.from_json(fname + '_thin.json')
line.cycle('qrdr2.3_entry', inplace=True)

line.build_tracker()
line.vars['on_wiggler_v'] = 0.5

# tt = line.get_table()
# wigs = tt.rows['mwi.*', tt.element_type=='Multipole'].name


# for nn in tt.rows['mwi.*.tilt.*'].name:
#     line.element_refs[nn].angle = 0

# for nn in wigs:
#     line.element_refs[nn].hyl = line.element_refs[nn].hxl._expr
#     line.element_refs[nn].hxl = 0
#     line.element_refs[nn].ksl[0] = line.element_refs[nn].knl[0]._expr
#     line.element_refs[nn].knl[0] = 0

tw_no_rad = line.twiss(method='4d')

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True)

ex = tw_rad.eq_gemitt_x
ey = tw_rad.eq_gemitt_y
ez = tw_rad.eq_gemitt_zeta

num_particles_test = 300
n_turns_track_test = 400

line.configure_radiation(model='quantum')
p = line.build_particles(num_particles=num_particles_test)
line.track(p, num_turns=n_turns_track_test, turn_by_turn_monitor=True, time=True)
mon = line.record_last_track
print(f'Tracking time: {line.time_last_track}')

import matplotlib.pyplot as plt
plt.close('all')
fig = plt.figure(1, figsize=(6.4, 4.8*1.3))
spx = fig. add_subplot(3, 1, 1)
spx.plot(np.std(mon.x, axis=0), label='track')
spx.axhline(
    np.sqrt(ex * tw_rad.betx[0] + ey * tw_rad.betx2[0] + (np.std(p.delta) * tw_rad.dx[0])**2),
    color='red', label='twiss')
spx.legend(loc='lower right')
spx.set_ylabel(r'$\sigma_{x}$ [m]')

spy = fig. add_subplot(3, 1, 2, sharex=spx)
spy.plot(np.std(mon.y, axis=0), label='track')
spy.axhline(
    np.sqrt(ex * tw_rad.bety1[0] + ey * tw_rad.bety[0] + (np.std(p.delta) * tw_rad.dy[0])**2),
    color='red', label='twiss')
spx.set_ylabel(r'$\sigma_{y}$ [m]')

spz = fig. add_subplot(3, 1, 3, sharex=spx)
spz.plot(np.std(mon.zeta, axis=0))
spz.axhline(np.sqrt(ez * tw_rad.betz0), color='red')
spz.set_ylabel(r'$\sigma_{z}$ [m]')

plt.suptitle(f'{fname} - ' r'$\varepsilon_y$ = ' f'{ey*1e12:.6f} pm')

plt.show()

