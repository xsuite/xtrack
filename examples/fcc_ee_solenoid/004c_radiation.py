import xtrack as xt
import xobjects as xo
import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

line = xt.Line.from_json('fccee_t_with_sol_corrected.json')
line.cycle('ip.1')
tw_no_rad = line.twiss(method='4d')
line.configure_radiation(model='mean')
tt = line.get_table(attr=True)

line.vars['on_corr_ip.1'] = 1
line.vars['on_sol_ip.1'] = 1


# # Radiation only in solenoid
# ttmult = tt.rows[tt.element_type == 'Multipole']
# for nn in ttmult.name:
#     line[nn].radiation_flag=0

# RF on
line.vars['voltca1'] = line.vv['voltca1_ref']
line.vars['voltca2'] = line.vv['voltca2_ref']
line.compensate_radiation_energy_loss()
tw = line.twiss()

eloss = np.diff(tw.ptau) * line.particle_ref.energy0[0]
ds = tt.length[:-1]

mask_ds = ds > 0

dE_ds = eloss * 0
dE_ds[mask_ds] = -eloss[mask_ds] / ds[mask_ds]

tw_rad = line.twiss(eneloss_and_damping=True)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
ax1 = plt.subplot(3, 1, 1)
plt.plot(tw.s[:-1], dE_ds * 1e-2 * 1e-3, '.-', label='dE/ds')
plt.xlabel('s [m]')
plt.ylabel('dE/ds [keV/m]')
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(tw.s, tt.ks, '.-', label='ks')
plt.xlabel('s [m]')
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(tw.s, tw_rad.delta, '.-', label='ks')
plt.xlabel('s [m]')

print('partition numbers: ', tw_rad.partition_numbers)
print('gemit_x: ', tw_rad.eq_gemitt_x)
print('gemit_y: ', tw_rad.eq_gemitt_y)

tw_rad = line.twiss(eneloss_and_damping=True)

ex = tw_rad.eq_gemitt_x
ey = tw_rad.eq_gemitt_y
ez = tw_rad.eq_gemitt_zeta

num_particles_test = 200
n_turns_track_test = 200

line.configure_radiation(model='quantum')
p = line.build_particles(num_particles=num_particles_test)

# Switch to multithreaded
line.discard_tracker()
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'),
                   use_prebuilt_kernels=False)

line.track(p, num_turns=n_turns_track_test, turn_by_turn_monitor=True, time=True,
           with_progress=10)
mon_at_start = line.record_last_track
print(f'Tracking time: {line.time_last_track}')

line.configure_radiation(model='mean')

import matplotlib.pyplot as plt
plt.close('all')

mon = mon_at_start

betx = tw_rad['betx', 0]
bety = tw_rad['bety', 0]
betx2 = tw_rad['betx2', 0]
bety1 = tw_rad['bety1', 0]
dx = tw_rad['dx', 0]
dy = tw_rad['dy', 0]

fig = plt.figure(100 + 1, figsize=(6.4, 4.8*1.3))
spx = fig. add_subplot(3, 1, 1)
spx.plot(np.std(mon.x, axis=0), label='track')
spx.axhline(
    np.sqrt(ex * betx + ey * betx2 + (np.std(p.delta) * dy)**2),
    color='red', label='twiss')
spx.legend(loc='lower right')
spx.set_ylabel(r'$\sigma_{x}$ [m]')
spx.set_ylim(bottom=0)

spy = fig. add_subplot(3, 1, 2, sharex=spx)
spy.plot(np.std(mon.y, axis=0), label='track')
spy.axhline(
    np.sqrt(ex * bety1 + ey * bety + (np.std(p.delta) * dy)**2),
    color='red', label='twiss')
spy.set_ylabel(r'$\sigma_{y}$ [m]')
spy.set_ylim(bottom=0)

spz = fig. add_subplot(3, 1, 3, sharex=spx)
spz.plot(np.std(mon.zeta, axis=0))
spz.axhline(np.sqrt(ez * tw_rad.bets0), color='red')
spz.set_ylabel(r'$\sigma_{z}$ [m]')
spz.set_ylim(bottom=0)

plt.suptitle(r'$\varepsilon_y$ = ' f'{ey*1e12:.6f} pm')

plt.show()
