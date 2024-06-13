import xtrack as xt
import xobjects as xo
import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

line = xt.Line.from_json('fccee_z_with_sol_corrected.json')
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

tw_rad = line.twiss(eneloss_and_damping=True, radiation_method='full')

ex = tw_rad.eq_gemitt_x
ey = tw_rad.eq_gemitt_y
ez = tw_rad.eq_gemitt_zeta

# Equilibrium beam sizes
beam_sizes = tw_rad.get_beam_covariance(
    gemitt_x=tw_rad.eq_gemitt_x, gemitt_y=tw_rad.eq_gemitt_y,
    gemitt_zeta=tw_rad.eq_gemitt_zeta)

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

# Approximated calculation based on Hofmann Eq. 14.19
twe = tw_no_rad
tt = line.get_table(attr=True)
hh = np.sqrt(np.diff(twe.px, append=0)**2 + np.diff(twe.py, append=0)**2)
dl = np.diff(twe.s, append=line.get_length())
gamma0 = line.particle_ref.gamma0[0]

dyprime = twe.dpy*(1 - twe.delta) - twe.py

cur_H_y = twe.gamy * twe.dy**2 + 2 * twe.alfy * twe.dy * dyprime + twe.bety * dyprime**2
I5_y  = np.sum(cur_H_y * hh**3 * dl)
I2_y = np.sum(hh**2 * dl)
I4_y = np.sum(twe.dy * hh**3 * dl) # to be generalized for combined function magnets

lam_comp = 2.436e-12 # [m]
ey_hof = 55 * np.sqrt(3) / 96 * lam_comp / 2 / np.pi * gamma0**2 * I5_y / (I2_y - I4_y)

# Plots
import matplotlib.pyplot as plt
plt.close('all')

line.configure_radiation(model='mean') # Leave the line in a twissable state
mon = line.record_last_track

fig = plt.figure(figsize=(6.4, 4.8*1.3))

spx = fig. add_subplot(3, 1, 1)
spx.plot(1e6 * np.std(mon.x, axis=0), label='track')
spx.axhline(1e6 * beam_sizes['sigma_x', 'ip.1'], color='red', label='twiss')
spx.legend(loc='lower right', fontsize='small')
spx.set_ylabel(r'$\sigma_{x}$ [$\mu m$]')
spx.set_ylim(bottom=0)

spy = fig. add_subplot(3, 1, 2, sharex=spx)
spy.plot(1e9 * np.std(mon.y, axis=0), label='track')
spy.axhline(1e9 * beam_sizes['sigma_y', 'ip.1'], color='red', label='twiss')
spy.set_ylabel(r'$\sigma_{y}$ [nm]')
spy.set_ylim(bottom=0)

spz = fig. add_subplot(3, 1, 3, sharex=spx)
spz.plot(np.std(1e3*mon.zeta, axis=0))
spz.axhline(1e3*beam_sizes['sigma_zeta', 'ip.1'], color='red', label='twiss')
spz.set_ylabel(r'$\sigma_{z}$ [mm]')
spz.set_ylim(bottom=0)
spz.set_xlabel('s [m]')
plt.subplots_adjust(left=.2, top=.95, hspace=.2)


plt.show()