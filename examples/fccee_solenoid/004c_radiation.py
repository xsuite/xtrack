import xtrack as xt
import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

line = xt.Line.from_json('fccee_z_thick_with_sol_corrected.json')
tw_no_rad = line.twiss(method='4d')
line.configure_radiation(model='mean')
tt = line.get_table(attr=True)
ttmult = tt.rows[tt.element_type == 'Multipole']

# RF on
line.vars['voltca1'] = 13.2


tw = line.twiss()

eloss = np.diff(tw.ptau) * line.particle_ref.energy0[0]
ds = tt.length[:-1]

mask_ds = ds > 0

dE_ds = eloss * 0
dE_ds[mask_ds] = -eloss[mask_ds] / ds[mask_ds]

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
plt.plot(tw.s[:-1], np.diff(tt.ks), '.-', label='ks')
plt.xlabel('s [m]')

plt.show()