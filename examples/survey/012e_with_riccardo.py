import xtrack as xt
import xobjects as xo
import numpy as np
from dataclasses import dataclass

from ldbpoint import MADPoint


@dataclass
class Misalignment:
    dtheta: float
    dphi: float
    dpsi: float
    dx: float
    dy: float
    ds: float

    def apply_to_element(self, element):
        if hasattr(element, 'rbend_model') and element.rbend_model == 'straight-body':
            raise ValueError('straight-body rbends not yet supported for misalignment application')
        element.rot_shift_anchor = 0. # Defined at the entrance
        element.rot_y_rad = self.dtheta
        element.rot_x_rad = self.dphi
        element.rot_s_rad_no_frame = self.dpsi - element.rot_s_rad
        element.shift_x = self.dx
        element.shift_y = self.dy
        element.shift_s = self.ds


def misalignment_from_absolute_position(XYZ_elem_start, E_elem_start,
                                        XYZ_ref_start, E_ref_start):

    p_mat_elem_start = np.eye(4)
    p_mat_elem_start[:3, :3] = E_elem_start
    p_mat_elem_start[:3, 3] = XYZ_elem_start

    p_mat_ref_start = np.eye(4)
    p_mat_ref_start[:3, :3] = E_ref_start
    p_mat_ref_start[:3, 3] = XYZ_ref_start

    A = np.linalg.inv(p_mat_ref_start) @ p_mat_elem_start

    theta = np.arctan2(A[0, 2], A[2, 2])
    phi = np.arctan2(A[1, 2], np.sqrt(A[1, 0]**2 + A[1, 1]**2))
    psi = np.arctan2(A[1, 0], A[1, 1])
    dx = A[0, 3]
    dy = A[1, 3]
    ds = A[2, 3]

    return Misalignment(dtheta=theta, dphi=phi, dpsi=psi, dx=dx, dy=dy, ds=ds)

env = xt.Environment()
env.set_particle_ref('proton', p0c=400e9)

env.new('bend_1', 'RBend', length_straight=2,
        angle=0., # This dipole does not bend the reference frame
        k0=0.1,
        rbend_compensate_sagitta=False,
        rot_shift_anchor=1., # shift defined in the middle
        shift_x=0.5)
env.new('bend_2', 'RBend', length_straight=2,
        angle=-0.2, # This dipole bends the reference frame
        rot_s_rad = np.deg2rad(15),
        rbend_model='straight-body',
        rbend_compensate_sagitta=True,
        rot_shift_anchor=1., # shift defined in the middle
        rot_y_rad=np.deg2rad(10),
        shift_x=0.5
)
env.new('translation', 'Translation')
env.new('rotation', 'Rotation')

line = env.new_line(length=15, components=[
    env.place('bend_1', at=2),
    env.new('ref_change', 'Marker', at=4),
    env.place(['translation', 'rotation'], at='ref_change@end'),
    env.place('bend_2', at=10),
])

tw0 = line.twiss(betx=1, bety=1)

env['translation'].shift_x = tw0['x', 'translation']
env['rotation'].rot_y_rad = np.asin(tw0['px', 'rotation'])

sv = line.survey(include_element_frames=True)

elem_name = 'bend_2'

line_no_jumps = line.copy(shallow=False)

line_no_jumps['translation'].shift_x = 0
line_no_jumps['rotation'].rot_y_rad = 0

ee_nj = line_no_jumps[elem_name]

ee_nj.rbend_model = 'curved-body'
ee_nj.rot_x_rad = 0
ee_nj.rot_y_rad = 0
ee_nj.rot_s_rad_no_frame = 0
ee_nj.shift_x = 0
ee_nj.shift_y = 0
ee_nj.shift_z = 0
ee_nj.rot_shift_anchor = 0

sv0_nj = line_no_jumps.survey(include_element_frames=True)


XYZ_elem_start = sv['XYZ_elem_start', elem_name]
E_elem_start = sv['E_elem_start', elem_name]
XYZ_elem_end = sv['XYZ_elem_end', elem_name]
E_elem_end = sv['E_elem_end', elem_name]

XYZ_nj_ref_start = sv0_nj['XYZ_ref_start', elem_name]
E_nj_ref_start = sv0_nj['E_ref_start', elem_name]

p_mat_elem_start = np.eye(4)
p_mat_elem_start[:3, :3] = E_elem_start
p_mat_elem_start[:3, 3] = XYZ_elem_start


mp_elem_start = MADPoint(p_mat_elem_start)
mp_elem_start.rtheta(ee_nj.angle/2)
E_elem_start_rot = mp_elem_start.matrix[:3, :3]
XYZ_elem_start_rot = mp_elem_start.matrix[:3, 3]

misalignment = misalignment_from_absolute_position(
    XYZ_elem_start=XYZ_elem_start_rot,
    E_elem_start=E_elem_start_rot,
    XYZ_ref_start=XYZ_nj_ref_start,
    E_ref_start=E_nj_ref_start
)

misalignment.apply_to_element(ee_nj)

sv_nj = line_no_jumps.survey(include_element_frames=True)




xo.assert_allclose(np.cross(E_elem_start[:, 2], XYZ_elem_end - XYZ_elem_start), 0, atol=1e-12)

import matplotlib.pyplot as plt


def plot_exz(rotation_matrix, point, length=0.5, color='k'):
    """Plot the local x and z directions in the global Z-X plane."""
    if length <= 0:
        raise ValueError('length must be positive')

    rotation_matrix = np.asarray(rotation_matrix)
    point = np.asarray(point)
    ax = plt.gca()
    arrows = []

    for axis_index in (0, 2):
        direction = rotation_matrix[:, axis_index]
        delta_z = length * direction[2]
        delta_x = length * direction[0]
        projected_length = np.hypot(delta_z, delta_x)

        arrows.append(ax.arrow(
            point[2], point[0], delta_z, delta_x,
            width=0.025 * projected_length,
            head_width=0.15 * projected_length,
            head_length=0.25 * projected_length,
            length_includes_head=True,
            color=color,
        ))

    return arrows


def plot_exy(rotation_matrix, point, length=0.5, color='k'):
    """Plot the local x and y directions in the global X-Y plane."""
    if length <= 0:
        raise ValueError('length must be positive')

    rotation_matrix = np.asarray(rotation_matrix)
    point = np.asarray(point)
    ax = plt.gca()
    arrows = []

    for axis_index in (0, 1):
        direction = rotation_matrix[:, axis_index]
        delta_x = length * direction[0]
        delta_y = length * direction[1]
        projected_length = np.hypot(delta_x, delta_y)

        arrows.append(ax.arrow(
            point[0], point[1], delta_x, delta_y,
            width=0.025 * projected_length,
            head_width=0.15 * projected_length,
            head_length=0.25 * projected_length,
            length_includes_head=True,
            color=color,
        ))

    return arrows


plt.close('all')
plt.figure(1)
plt.plot(sv.Z, sv.X, '.-', label='survey')
plt.plot(sv0_nj.Z, sv0_nj.X, '--', label='survey no jumps')

plt.plot([sv['Z_elem_start', elem_name], sv['Z_elem_end', elem_name]],
            [sv['X_elem_start', elem_name], sv['X_elem_end', elem_name]],
            'x-',
            color='g'
)
plt.plot([sv0_nj['Z_ref_start', elem_name], sv0_nj['Z_ref_end', elem_name]],
            [sv0_nj['X_ref_start', elem_name], sv0_nj['X_ref_end', elem_name]],
            '+--',
            color='orange'
)
plt.plot([sv_nj['Z_elem_start', elem_name], sv_nj['Z_elem_end', elem_name]],
         [sv_nj['X_elem_start', elem_name], sv_nj['X_elem_end', elem_name]],
         '.--',
         color='r'
)

plot_exz(sv['E_elem_start', elem_name], sv['XYZ_elem_start', elem_name], length=0.5, color='g')
plot_exz(sv['E_elem_end', elem_name], sv['XYZ_elem_end', elem_name], length=0.5, color='g')
plot_exz(sv0_nj['E_elem_start', elem_name], sv0_nj['XYZ_elem_start', elem_name], length=0.5, color='orange')
plot_exz(sv0_nj['E_elem_end', elem_name], sv0_nj['XYZ_elem_end', elem_name], length=0.5, color='orange')
plot_exz(E_elem_start_rot, XYZ_elem_start_rot, length=0.5, color='r')



plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.axis('equal')
plt.legend()

plt.figure(2)
plot_exy(sv['E_elem_start', elem_name], sv['XYZ_elem_start', elem_name], length=0.5, color='g')
plot_exy(sv_nj['E_elem_start', elem_name], sv_nj['XYZ_elem_start', elem_name], length=0.3, color='orange')

plt.xlabel('X [m]')
plt.ylabel('Y [m]')

plt.show()