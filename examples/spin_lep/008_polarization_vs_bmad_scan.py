"""
To intall bmad:
  conda install -c conda-forge bmad
  pip install pytao
"""

import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np

from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import m_e
from scipy.constants import hbar

num_turns = 500
bmad = True

vrfc231 = 0.
method = '4d'

# vrfc231 = 12.65 # qs=0.6
# method = '6d'

bump_strength_list = np.linspace(0, 1.3, 10)

pol_xsuite = []
pol_bmad = []

for bump_strength in bump_strength_list:

    line = xt.load('lep_sol.json')
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

    line['on_solenoids'] = 1
    line['on_spin_bumps'] = bump_strength
    line['on_coupling_corrections'] = 1

    if bmad:
        from bmad_track_twiss_spin import bmad_run
        bmad_data = bmad_run(line)
        df = bmad_data['spin']
        df_orb = bmad_data['optics']
        spin_summary_bmad = bmad_data['spin_summary']

    tw = line.twiss4d(polarization=True)

    print('Xsuite polarization: ', tw.spin_polarization_eq)
    pol_xsuite.append(tw.spin_polarization_eq)

    if bmad:
        print('Bmad polarization:   ', spin_summary_bmad['Polarization Limit DK'])
        pol_bmad.append(spin_summary_bmad['Polarization Limit DK'])

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1, figsize=(6.4, 4.8))
plt.plot(bump_strength_list, pol_xsuite, '.-', label='Xsuite')
plt.plot(bump_strength_list, pol_bmad, 'x-', label='Bmad')
plt.ylabel('Polarization')
plt.xlabel('Bump strength')
plt.legend()

plt.show()