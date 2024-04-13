import xtrack as xt
import numpy as np

slice_mode = 'thick'

line = xt.Line(
    elements=[
        xt.Drift(length=1),
        xt.Bend(h=np.pi/2, length=1),
        xt.Drift(length=1),
        xt.Bend(h=np.pi/2, length=1),
        xt.Drift(length=1),
        xt.Bend(h=np.pi/2, length=1),
        xt.Drift(length=1),
        xt.Bend(h=np.pi/2, length=1),
    ]
)
line.build_tracker()

tt = line.get_table()
tt_bend = tt.rows[tt['element_type'] == 'Bend']

line.vars['tilt_bend_deg'] = 0
for nn in tt_bend.name:
    line.element_refs[nn].rot_s_rad = line.vars['tilt_bend_deg'] * np.pi / 180

line.slice_thick_elements(
    slicing_strategies=[xt.Strategy(xt.Teapot(20, mode=slice_mode))])
line.build_tracker()

line.vars['tilt_bend_deg'] = 90


sv = line.survey()

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(sv.Z, sv.X)
plt.plot(sv.Z, sv.Y)

plt.figure(2)
plt.plot(sv.s, sv.X)
plt.plot(sv.s, sv.Y)
plt.plot(sv.s, sv.Z)

plt.show()