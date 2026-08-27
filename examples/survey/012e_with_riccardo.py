import xtrack as xt
import xobjects as xo
import numpy as np

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

XYZ_elem_start = sv['XYZ_elem_start', elem_name]
E_elem_start = sv['E_elem_start', elem_name]
XYZ_elem_end = sv['XYZ_elem_end', elem_name]
E_elem_end = sv['E_elem_end', elem_name]

xo.assert_allclose(np.cross(E_elem_start[:, 2], XYZ_elem_end - XYZ_elem_start), 0, atol=1e-12)

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

import matplotlib.pyplot as plt

plt.close('all')
plt.plot(sv.Z, sv.X, '.-', label='survey')
plt.plot(sv0_nj.Z, sv0_nj.X, '--', label='survey no jumps')

plt.plot([sv['Z_elem_start', elem_name], sv['Z_elem_end', elem_name]],
            [sv['X_elem_start', elem_name], sv['X_elem_end', elem_name]],
            'x-',
            color='g'
)
plt.plot([sv0_nj['Z_elem_start', elem_name], sv0_nj['Z_elem_end', elem_name]],
            [sv0_nj['X_elem_start', elem_name], sv0_nj['X_elem_end', elem_name]],
            '+--',
            color='orange'
)


plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.axis('equal')
plt.legend()
plt.show()