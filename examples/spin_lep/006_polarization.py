import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np

vrfc231 = 12.65 # qs=0.6
method = '6d'

line = xt.Line.from_json('lep_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
line.particle_ref.gamma0 = 89207.78287659843 # to have a spin tune of 103.45
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]
line['vrfc231'] = vrfc231

# line.cycle('b2m.qf45.l6', inplace=True)

tt = line.get_table(attr=True)
tt_bend = tt.rows[(tt.element_type == 'RBend') | (tt.element_type == 'Bend')]
tt_quad = tt.rows[tt.element_type == 'Quadrupole']
tt_sext = tt.rows[tt.element_type == 'Sextupole']

# simplify the line to facilitate bmad comparison
for nn in tt_bend.name:
    line[nn].k1 = 0
    line[nn].knl[2] = 0
    line[nn].edge_entry_angle = 0
    line[nn].edge_exit_angle = 0

# for nn in tt_sext.name:
#     line[nn].k2 = 0

line.set(tt_bend, model='mat-kick-mat', integrator='uniform', num_multipole_kicks=5)
line.set(tt_quad, model='mat-kick-mat', integrator='uniform', num_multipole_kicks=5)

line['on_sol.2'] = 1
line['on_sol.4'] = 1
line['on_sol.6'] = 1
line['on_sol.8'] = 1
line['on_spin_bump.2'] = 1
line['on_spin_bump.4'] = 1
line['on_spin_bump.6'] = 1
line['on_spin_bump.8'] = 1
line['on_coupl_sol.2'] = 0
line['on_coupl_sol.4'] = 0
line['on_coupl_sol.6'] = 0
line['on_coupl_sol.8'] = 0
line['on_coupl_sol_bump.2'] = 0
line['on_coupl_sol_bump.4'] = 0
line['on_coupl_sol_bump.6'] = 0
line['on_coupl_sol_bump.8'] = 0

tw = line.twiss4d(polarization=True)

print('Xsuite polarization: ', tw.spin_polarization_eq)

import matplotlib.pyplot as plt
plt.close('all')


plt.figure(1)
ax1 = plt.subplot(3, 1, 1)
plt.plot(tw.s, tw.spin_x, label='Xsuite')
plt.ylabel('spin_x')
plt.legend()
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(tw.s, tw.spin_y, label='y')
plt.ylabel('spin_y')
plt.ylim(top=1)
plt.legend()
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.xlabel('s [m]')
plt.plot(tw.s, tw.spin_z, label='z')
plt.ylabel('spin_z')
plt.legend()

plt.figure(3, figsize=(8, 6))
ax1 = plt.subplot(3, 1, 1)
tw.plot(lattice_only=True, ax=ax1)
plt.plot(tw.s, tw.spin_dn_ddelta_x)
plt.ylabel('dn_ddelta_x')
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
tw.plot(lattice_only=True, ax=ax2)
plt.plot(tw.s, tw.spin_dn_ddelta_y)
plt.ylabel('dn_ddelta_y')
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
tw.plot(lattice_only=True, ax=ax3)
plt.plot(tw.s, tw.spin_dn_ddelta_z)
plt.ylabel('dn_ddelta_z')
plt.xlabel('s [m]')


plt.show()
