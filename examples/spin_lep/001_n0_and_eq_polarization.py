import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np

# Some references:
# CERN-SL-94-71-BI https://cds.cern.ch/record/267514
# CERN-LEP-Note-629 https://cds.cern.ch/record/442887

vrfc231 = 12.65 # qs=0.6
method = '6d'

line = xt.Line.from_json('lep_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
line.particle_ref.gamma0 = 89207.78287659843 # to have a spin tune of 103.45
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]
line['vrfc231'] = vrfc231

line['on_solenoids'] = 1
line['on_spin_bumps'] = 1
line['on_coupling_corrections'] = 1

tt = line.get_table(attr=True)
tt_bend = tt.rows[(tt.element_type == 'RBend') | (tt.element_type == 'Bend')]
tt_quad = tt.rows[tt.element_type == 'Quadrupole']
tt_sext = tt.rows[tt.element_type == 'Sextupole']

tt = line.get_table(attr=True)
tt_bend = tt.rows[(tt.element_type == 'RBend') | (tt.element_type == 'Bend')]
tt_quad = tt.rows[tt.element_type == 'Quadrupole']
tt_sext = tt.rows[tt.element_type == 'Sextupole']

# Set interators and multipole kicks
line.set(tt_bend, model='mat-kick-mat', integrator='uniform', num_multipole_kicks=5)
line.set(tt_quad, model='mat-kick-mat', integrator='uniform', num_multipole_kicks=5)

# Set solenoids, spin bumps and coupling corrections
line['on_sol.2'] = 'on_solenoids'
line['on_sol.4'] = 'on_solenoids'
line['on_sol.6'] = 'on_solenoids'
line['on_sol.8'] = 'on_solenoids'
line['on_spin_bump.2'] = 'on_spin_bumps'
line['on_spin_bump.4'] = 'on_spin_bumps'
line['on_spin_bump.6'] = 'on_spin_bumps'
line['on_spin_bump.8'] = 'on_spin_bumps'
line['on_coupl_sol.2'] = 'on_coupling_corrections * on_solenoids'
line['on_coupl_sol.4'] = 'on_coupling_corrections * on_solenoids'
line['on_coupl_sol.6'] = 'on_coupling_corrections * on_solenoids'
line['on_coupl_sol.8'] = 'on_coupling_corrections * on_solenoids'
line['on_coupl_sol_bump.2'] = 'on_coupling_corrections * on_spin_bumps'
line['on_coupl_sol_bump.4'] = 'on_coupling_corrections * on_spin_bumps'
line['on_coupl_sol_bump.6'] = 'on_coupling_corrections * on_spin_bumps'
line['on_coupl_sol_bump.8'] = 'on_coupling_corrections * on_spin_bumps'


tw = line.twiss4d(spin=True, polarization=True)

print('Xsuite polarization: ', tw.spin_polarization_eq)

import matplotlib.pyplot as plt
plt.close('all')


plt.figure(1, figsize=(8, 6))
ax1 = plt.subplot(3, 1, 1)
plt.plot(tw.s, tw.x, label='x')
plt.plot(tw.s, tw.y, label='y')
plt.ylabel('Closed orbit [m]')
plt.legend()
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(tw.s, tw.spin_y, label='y')
plt.ylabel('spin_y')
plt.ylim(top=1)
plt.legend()
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.xlabel('s [m]')
plt.plot(tw.s, tw.spin_z, label='z')
plt.plot(tw.s, tw.spin_x, label='x')
plt.ylabel('spin_z')
plt.legend()
plt.suptitle(f'Equilibrium polarization: {tw.spin_polarization_eq*100:.2f} %')

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

tw_ir4_right = tw.rows['ip4':'ip5']
plt.figure(4)
plt.axis('equal')
plt.plot(tw_ir4_right.spin_z*1e3, tw_ir4_right.spin_x*1e3)
plt.ylabel('spin_x [mrad]')
plt.xlabel('spin_z [mrad]')
plt.title('Spin precession in between IP4 and IP5')
plt.grid(alpha=0.5)

plt.show()
