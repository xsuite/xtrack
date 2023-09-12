import numpy as np
import xtrack as xt

# fname = 'fccee_z'; gemitt_y_target = 1.4e-12; n_turns_track_test = 3000
# fname = 'fccee_w'; gemitt_y_target = 2.2e-12; n_turns_track_test = 1000
fname = 'fccee_h'; gemitt_y_target = 1.4e-12; n_turns_track_test = 400
fname = 'fccee_t'; gemitt_y_target = 2e-12; n_turns_track_test = 400


line = xt.Line.from_json(fname + '_thin.json')
line.cycle('qrdr2.3_entry', inplace=True)

line.build_tracker()

tw_no_rad = line.twiss(method='4d')

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad_wig_off = line.twiss(eneloss_and_damping=True)

line.vars['on_wiggler_v'] = 0.1
line.compensate_radiation_energy_loss()
opt = line.match(
    solve=False,
    eneloss_and_damping=True,
    compensate_radiation_energy_loss=True,
    targets=[
        xt.Target(eq_gemitt_y=gemitt_y_target, tol=1e-15, optimize_log=True)],
    vary=xt.Vary('on_wiggler_v', step=0.01, limits=(0.1, 2))
)

opt.solve()

tw_rad = line.twiss(eneloss_and_damping=True)

ex = tw_rad.eq_gemitt_x
ey = tw_rad.eq_gemitt_y
ez = tw_rad.eq_gemitt_zeta

line.configure_radiation(model='quantum')
p = line.build_particles(num_particles=300)
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
spy.set_ylabel(r'$\sigma_{y}$ [m]')

spz = fig. add_subplot(3, 1, 3, sharex=spx)
spz.plot(np.std(mon.zeta, axis=0))
spz.axhline(np.sqrt(ez * tw_rad.betz0), color='red')
spz.set_ylabel(r'$\sigma_{z}$ [m]')

plt.suptitle(f'{fname} - ' r'$\varepsilon_y$ = ' f'{ey*1e12:.6f} pm')

plt.show()