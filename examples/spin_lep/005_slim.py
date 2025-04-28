import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np


line = xt.Line.from_json('lep_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
line.particle_ref.gamma0 = 89207.78287659843 # to have a spin tune of 103.45
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]

line['vrfc231'] = 12.65 # qs=0.6

# All off
line['on_sol.2'] = 1
line['on_sol.4'] = 0
line['on_sol.6'] = 0
line['on_sol.8'] = 0
line['on_spin_bump.2'] = 0
line['on_spin_bump.4'] = 0
line['on_spin_bump.6'] = 0
line['on_spin_bump.8'] = 0
line['on_coupl_sol.2'] = 0
line['on_coupl_sol.4'] = 0
line['on_coupl_sol.6'] = 0
line['on_coupl_sol.8'] = 0
line['on_coupl_sol_bump.2'] = 0
line['on_coupl_sol_bump.4'] = 0
line['on_coupl_sol_bump.6'] = 0
line['on_coupl_sol_bump.8'] = 0

tw = line.twiss(spin=True, radiation_integrals=True)

import polarization as pol

pol._add_polarization_to_tw(tw, line)