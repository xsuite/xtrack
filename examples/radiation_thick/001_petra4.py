import xtrack as xt
import numpy as np

from scipy.constants import hbar
from scipy.constants import electron_volt
from scipy.constants import c as clight

line = xt.Line.from_json('line_thick_P4_H6BA_v4.2.4.json')
line.particle_ref = xt.Particles(energy0=6e9, mass0=xt.ELECTRON_MASS_EV)

tw4d = line.twiss4d()
tw6d = line.twiss()

tt = line.get_table()
tt_bend = tt.rows[tt.element_type == 'Bend']
tt_quad = tt.rows[tt.element_type == 'Quadrupole']
tt_sext = tt.rows[tt.element_type == 'Sextupole']

for nn in tt_sext.name:
    line.get(nn).integrator = 'yoshida4'
    line.get(nn).num_multipole_kicks = 20

for nn in tt_bend.name:
    line.get(nn).integrator = 'yoshida4'
    line.get(nn).model = 'rot-kick-rot'
    line.get(nn).num_multipole_kicks = 10

tw4d_after = line.twiss4d()

line.build_tracker()
line.configure_radiation(model='mean')

for nn in tt_bend.name:
    line.get(nn).integrator = 'yoshida4'
    line.get(nn).model = 'drift-kick-drift-expanded'
    line.get(nn).num_multipole_kicks = 1

for nn in tt_bend.name:
    line.get(nn).integrator = 'teapot'
    line.get(nn).model = 'mat-kick-mat'
    line.get(nn).num_multipole_kicks = 1

tw_rad = line.twiss(eneloss_and_damping=True, strengths=True)

p = line.build_particles(x=np.linspace(-1e-3, 1e-3, 1000))
line.track(p, num_turns=5, with_progress=1, time=True)
print('Time for tracking: ', line.time_last_track)