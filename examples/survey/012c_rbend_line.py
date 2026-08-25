import xtrack as xt
import numpy as np

env = xt.Environment()
env.set_particle_ref('proton', p0c=400e9)


env.new('bend_1', 'RBend', length_straight=2,
        angle=0, # This dipole does not bend the reference frame
        k0=0.1,
        rbend_model='straight-body',
        rbend_compensate_sagitta=False,
        rot_shift_anchor=1., # shift defined in the middle
        shift_x=0.5)
env.new('bend_2', 'RBend', length_straight=2,
        angle=-0.2, # This dipole bends the reference frame
        rbend_model='straight-body',
        rbend_compensate_sagitta=True,
        rot_shift_anchor=1., # shift defined in the middle
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

elems_to_plot = ['bend_1', 'bend_2']

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(sv_sliced.Z, sv_sliced.X, '.-', label='survey')

for nn in elems_to_plot:
    plt.plot([sv['Z_elem_start', nn], sv['Z_elem_end', nn]],
             [sv['X_elem_start', nn], sv['X_elem_end', nn]],
             'x--', label=nn)

plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.axis('equal')
plt.legend()

plt.show()