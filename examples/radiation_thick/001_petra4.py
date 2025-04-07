import xtrack as xt
import numpy as np

from scipy.constants import hbar
from scipy.constants import electron_volt
from scipy.constants import c as clight

line = xt.Line.from_json('line_thick_P4_H6BA_v4.2.4.json')
line.particle_ref = xt.Particles(energy0=6e9, mass0=xt.ELECTRON_MASS_EV)

tw4d = line.twiss4d()
tw6d = line.twiss()

tt = line.get_table(attr=True)
tt_bend = tt.rows[tt.element_type == 'Bend']
tt_quad = tt.rows[tt.element_type == 'Quadrupole']
tt_sext = tt.rows[tt.element_type == 'Sextupole']

line.set(tt_sext, integrator='yoshida4', num_multipole_kicks=7)
line.set(tt_bend, model='mat-kick-mat', integrator='teapot', num_multipole_kicks=1)
line.set(tt_bend.rows['wgl.*'], model='drift-kick-drift-expanded',
         integrator='teapot', num_multipole_kicks=1)

tw_after = line.twiss4d()

line.configure_radiation(model='mean')
tw_rad = line.twiss(eneloss_and_damping=True, strengths=True)

p = line.build_particles(x=np.linspace(-1e-3, 1e-3, 1000))
line.track(p, num_turns=5, with_progress=1, time=True)
print('Time for tracking: ', line.time_last_track)