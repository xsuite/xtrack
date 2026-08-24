import xtrack as xt
import misalignment_survey as ms
import numpy as np
import matplotlib.pyplot as plt

env = xt.Environment()

env.new('q', 'Quadrupole', length=2, k1=0.1,
        rot_shift_anchor=1, rot_y_rad=np.deg2rad(30))
env.new('b', 'Bend', length=2, angle=np.deg2rad(20),
        rot_shift_anchor=1, rot_y_rad=np.deg2rad(30))
line_thick = env.new_line(length=8, components=[
    env.place('q', at=2),
    env.place('b', at=6),
])

line_sliced = line_thick.copy(shallow=True)
line_sliced.cut_at_s(np.linspace(0, 8, 33))

# line = line_sliced
# name = 'q..3'

# line = line_thick
# name = 'q'

line = line_thick
name = 'b'

sv = line.survey()

elem = line[name]

XYZ_ref_start = sv['XYZ', name]
E_ref_start = sv['E_matrix', name]

XYZ_ref_end = sv['XYZ', name+'>>1']
E_ref_end = sv['E_matrix', name+'>>1']

XYZ_elem_start, E_elem_start, XYZ_elem_end, E_elem_end = (
    ms.get_misaligned_element_survey(
        elem,
        XYZ_ref_start,
        E_ref_start,
        XYZ_ref_end,
        E_ref_end,
    )
)

plt.close('all')
plt.figure(1)
plt.plot(sv.Z, sv.X, '.-', label='survey')
plt.plot(XYZ_elem_start[2], XYZ_elem_start[0], 'o', label='start elem')
plt.plot(XYZ_elem_end[2], XYZ_elem_end[0], 'o', label='end elem')
plt.axis('equal')
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.legend()

