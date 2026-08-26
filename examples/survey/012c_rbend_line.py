from attr import dataclass

import xtrack as xt
import numpy as np
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

    p1_mat_elem_start = np.eye(4)
    p1_mat_elem_start[:3, :3] = E_elem_start
    p1_mat_elem_start[:3, 3] = XYZ_elem_start

    p2_mat_ref_start = np.eye(4)
    p2_mat_ref_start[:3, :3] = E_ref_start
    p2_mat_ref_start[:3, 3] = XYZ_ref_start

    A = np.linalg.inv(p2_mat_ref_start) @ p1_mat_elem_start
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
        # rbend_model='straight-body',
        rbend_compensate_sagitta=False,
        rot_shift_anchor=1., # shift defined in the middle
        shift_x=0.5)
env.new('bend_2', 'RBend', length_straight=2,
        angle=-0.2, # This dipole bends the reference frame
        # rot_s_rad = np.deg2rad(15),
        # rbend_model='straight-body',
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

line_sliced = line.copy(shallow=True)
line_sliced.slice_thick_elements(
        slicing_strategies=[
            # Slicing with thin elements
            xt.Strategy(slicing=None),
            xt.Strategy(slicing=xt.Uniform(5), element_type=xt.RBend), # (2) Selection by element type
    ])

tw0 = line.twiss(betx=1, bety=1)

env['translation'].shift_x = tw0['x', 'translation']
env['rotation'].rot_y_rad = np.asin(tw0['px', 'rotation'])

sv = line.survey(include_element_frames=True)
sv_sliced = line_sliced.survey()

# Make a fictitious line without jumps to have continuous reference frame for plotting
line_no_jumps = line.copy(shallow=False)
line_no_jumps.remove('translation')
line_no_jumps.remove('rotation')

sv_smooth_no_misalign = line_no_jumps.survey(include_element_frames=True)

name_elem = 'bend_2'

# Absolute position and orientation of the element start in the real line (with jumps)
XYZ_elem_start = sv['XYZ_elem_start', name_elem]
E_elem_start = sv['E_elem_start', name_elem]

# Absolute position of the reference point in the smooth line (without jumps)
XYZ_ref_start = sv_smooth_no_misalign['XYZ_ref_start', name_elem]
E_ref_start = sv_smooth_no_misalign['E_ref_start', name_elem]

# Compute the misalignment of the element with respect to the smooth reference frame
misalignment = misalignment_from_absolute_position(XYZ_elem_start, E_elem_start,
                                               XYZ_ref_start, E_ref_start)

# Apply the misalignment to the element in the smooth line
misalignment.apply_to_element(line_no_jumps[name_elem])


sv_smooth = line_no_jumps.survey(include_element_frames=True)


elems_to_plot = ['bend_2']

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(sv_sliced.Z, sv_sliced.X, '.-', label='survey')
plt.plot(sv_smooth.Z, sv_smooth.X, 'o--', label='survey no jumps')

for nn in elems_to_plot:
    plt.plot([sv['Z_elem_start', nn], sv['Z_elem_end', nn]],
             [sv['X_elem_start', nn], sv['X_elem_end', nn]],
             'x--', label=nn)
    plt.plot([sv_smooth['Z_elem_start', nn], sv_smooth['Z_elem_end', nn]],
             [sv_smooth['X_elem_start', nn], sv_smooth['X_elem_end', nn]],
             '.--', label=nn + ' no jumps')

plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.axis('equal')
plt.legend()

plt.show()