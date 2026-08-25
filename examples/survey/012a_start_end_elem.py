import xtrack as xt
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

line = line_sliced

sv = line.survey(include_element_frames=True)

XYZ_elem_start = sv.XYZ_elem_start[:-1]
XYZ_elem_end = sv.XYZ_elem_end[:-1]

# Separate the elements with NaNs so that matplotlib draws one independent
# segment from XYZ_elem_start to XYZ_elem_end for each element.
Z_elem_segments = np.column_stack([
    XYZ_elem_start[:, 2], XYZ_elem_end[:, 2],
    np.full(len(XYZ_elem_start), np.nan),
]).ravel()
X_elem_segments = np.column_stack([
    XYZ_elem_start[:, 0], XYZ_elem_end[:, 0],
    np.full(len(XYZ_elem_start), np.nan),
]).ravel()

plt.close('all')
plt.figure(1)
plt.plot(sv.Z, sv.X, '.-', label='survey')
plt.plot(Z_elem_segments, X_elem_segments, 'o-', label='elements')
plt.axis('equal')
plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.legend()
