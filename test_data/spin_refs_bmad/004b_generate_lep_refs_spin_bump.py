from typing import Literal
import xtrack as xt
import xobjects as xo
import numpy as np
from scipy.interpolate import interp1d

num_turns = 500

mode: Literal['generate', 'verify'] = 'verify'
file_name = 'lep_bmad_spin_bump.json'

vrfc231 = 12.65 # qs=0.6
method = '6d'

line = xt.Line.from_json('../lep/lep_sol.json')
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

line.set(tt_bend, model='mat-kick-mat', integrator='uniform', num_multipole_kicks=5)
line.set(tt_quad, model='mat-kick-mat', integrator='uniform', num_multipole_kicks=5)

line.to_json('lep_lattice_to_bmad.json')


on_sol = 1
on_spin_bump = 1
on_coupl_sol = 0
on_coupl_sol_bump = 0


line['on_sol.2'] = on_sol
line['on_sol.4'] = on_sol
line['on_sol.6'] = on_sol
line['on_sol.8'] = on_sol
line['on_spin_bump.2'] = on_spin_bump
line['on_spin_bump.4'] = on_spin_bump
line['on_spin_bump.6'] = on_spin_bump
line['on_spin_bump.8'] = on_spin_bump
line['on_coupl_sol.2'] = on_coupl_sol
line['on_coupl_sol.4'] = on_coupl_sol
line['on_coupl_sol.6'] = on_coupl_sol
line['on_coupl_sol.8'] = on_coupl_sol
line['on_coupl_sol_bump.2'] = on_coupl_sol_bump
line['on_coupl_sol_bump.4'] = on_coupl_sol_bump
line['on_coupl_sol_bump.6'] = on_coupl_sol_bump
line['on_coupl_sol_bump.8'] = on_coupl_sol_bump


def make_table(data):
    return xt.Table(data={k: np.array(v) for k, v in data.items()}, index='name')


if mode == 'generate':
    from bmad_track_twiss_spin import bmad_run
    bmad_data = bmad_run(line)
    bmad_data['spin'] = bmad_data['spin'].to_dict('list')
    bmad_data['optics'] = bmad_data['optics'].to_dict('list')

    spin_bmad = make_table(data=bmad_data['spin'])
    optics_bmad = make_table(data={**bmad_data['optics'], 'name': spin_bmad.name})
    spin_summary_bmad = bmad_data['spin_summary']

    xt.json.dump(bmad_data, file_name)
elif mode == 'verify':
    bmad_data = xt.json.load(file_name)

    spin_bmad = make_table(data=bmad_data['spin'])
    optics_bmad = make_table(data={**bmad_data['optics'], 'name': spin_bmad.name})
    spin_summary_bmad = bmad_data['spin_summary']


# Make the tables the same length
start, end = 'ip1', 'bemi.ql1a.l1'
spin_bmad = spin_bmad.rows[start.upper():end.upper()]
optics_bmad = optics_bmad.rows[start.upper():end.upper()]

tw = line.twiss4d(polarization=True).rows[start:end]

print('Xsuite polarization: ', tw.spin_polarization_eq)
print('Bmad polarization:   ', spin_summary_bmad['Polarization Limit DK'])
xo.assert_allclose(tw.spin_polarization_eq, spin_summary_bmad['Polarization Limit DK'], atol=3e-2, rtol=0)

import matplotlib.pyplot as plt
plt.close('all')

for kk in ['spin_x', 'spin_y', 'spin_z',
            'spin_dn_dpz_x', 'spin_dn_dpz_y', 'spin_dn_dpz_z']:
      spin_bmad[kk] *= -1

dn_dy_ref_bmad = 1 / np.sqrt(1 - spin_bmad.spin_x ** 2 - spin_bmad.spin_z ** 2) * (
        -spin_bmad.spin_x * spin_bmad.spin_dn_dpz_x - spin_bmad.spin_z * spin_bmad.spin_dn_dpz_z)

plt.figure(1)
ax1 = plt.subplot(3, 1, 1)
plt.plot(tw.s, tw.spin_x, label='Xsuite')
plt.plot(spin_bmad.s, spin_bmad.spin_x, label='Bmad')
plt.ylabel('spin_x')
plt.legend()
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(tw.s, tw.spin_y, label='y')
plt.plot(spin_bmad.s, spin_bmad.spin_y, label='y Bmad')
plt.ylabel('spin_y')
plt.legend()
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.xlabel('s [m]')
plt.plot(tw.s, tw.spin_z, label='z')
plt.plot(spin_bmad.s, spin_bmad.spin_z, label='z Bmad')
plt.ylabel('spin_z')
plt.legend()

spin_x_interp = interp1d(spin_bmad.s, spin_bmad.spin_x)(tw.s)
xo.assert_allclose(tw.spin_x, spin_x_interp, atol=1e-5, rtol=0)

spin_y_interp = interp1d(spin_bmad.s, spin_bmad.spin_y)(tw.s)
xo.assert_allclose(tw.spin_y, spin_y_interp, atol=3e-7, rtol=0)

spin_z_interp = interp1d(spin_bmad.s, spin_bmad.spin_z)(tw.s)
xo.assert_allclose(tw.spin_z, spin_z_interp, atol=8e-6, rtol=0)


plt.figure(3, figsize=(8, 6))
ax1 = plt.subplot(3, 1, 1)
tw.plot(lattice_only=True, ax=ax1)
plt.plot(tw.s, tw.spin_dn_ddelta_x, label='dn_ddelta_x')
plt.plot(spin_bmad.s, spin_bmad.spin_dn_dpz_x, label='dn_dpz_x')
plt.ylabel('dn_ddelta_x')
plt.legend()
ax2 = plt.subplot(3, 1, 2, sharex=ax1)
tw.plot(lattice_only=True, ax=ax2)
plt.plot(tw.s, tw.spin_dn_ddelta_y, label='dn_ddelta_y')
plt.plot(spin_bmad.s, spin_bmad.spin_dn_dpz_y, label='dn_dpz_y (Bmad)')
plt.plot(spin_bmad.s, dn_dy_ref_bmad, label='dn_dy_ref (Bmad)')
plt.ylabel('dn_ddelta_y')
plt.legend()
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
tw.plot(lattice_only=True, ax=ax3)
plt.plot(tw.s, tw.spin_dn_ddelta_z, label='dn_ddelta_z')
plt.plot(spin_bmad.s, spin_bmad.spin_dn_dpz_z, label='dn_dpz_z (Bmad)')
plt.ylabel('dn_ddelta_z')
plt.xlabel('s [m]')
plt.legend()

spin_dn_dpz_x_interp = interp1d(spin_bmad.s, spin_bmad.spin_dn_dpz_x)(tw.s)
xo.assert_allclose(tw.spin_dn_ddelta_x, spin_dn_dpz_x_interp, atol=0.1, rtol=0)

spin_dn_dpz_y_interp = interp1d(spin_bmad.s, spin_bmad.spin_dn_dpz_y)(tw.s)
xo.assert_allclose(tw.spin_dn_ddelta_y, spin_dn_dpz_y_interp, atol=0.003, rtol=0)

spin_dn_dpz_z_interp = interp1d(spin_bmad.s, spin_bmad.spin_dn_dpz_z)(tw.s)
xo.assert_allclose(tw.spin_dn_ddelta_z, spin_dn_dpz_z_interp, atol=0.1, rtol=0)

plt.show()
