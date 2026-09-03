import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from xtrack._temp import survey_utils as su


elements_to_process = ['bend_1', 'bend_2', 'bend_3', 'q1', 'q2']

env = xt.Environment()
env.set_particle_ref('proton', p0c=400e9)

env.new('bend_1', 'RBend', length_straight=2,
        angle=0.,  # This dipole does not bend the reference frame
        k0=0.1,
        rbend_compensate_sagitta=False,
        rot_shift_anchor=1.,  # shift defined in the middle
        shift_x=0.5)

env.new('bend_2', 'Bend', length=2,
        angle=-0.2,  # This dipole bends the reference frame
        rot_s_rad=np.deg2rad(15),
        rot_shift_anchor=1.,  # shift defined in the middle
        rot_y_rad=np.deg2rad(10),
        shift_x=0.5)

env.new('bend_3', 'RBend', length_straight=2,
        angle=0.2,  # This dipole bends the reference frame
        rot_s_rad=np.deg2rad(15),
        # rbend_model='straight-body',
        rbend_compensate_sagitta=True,
        rot_shift_anchor=1.,  # shift defined in the middle
        rot_y_rad=np.deg2rad(-20),
        shift_x=-0.1)

env.new('q1', 'Quadrupole', length=0.7, k1=0.1,
        # Rotation about the chord is supplied separately through bgamma.
        rot_s_rad_no_frame=np.deg2rad(5))
env.new('q2', 'Quadrupole', length=0.7, k1=-0.1)

env.new('translation', 'Translation')
env.new('rotation', 'Rotation')

line = env.new_line(length=20, components=[
    env.place('bend_1', at=2),
    env.new('ref_change', 'Marker', at=4),
    env.place(['translation', 'rotation'], at='ref_change@end'),
    env.place('q1', at=6),
    env.place('bend_2', at=10),
    env.place('q2', at=12),
    env.place('bend_3', at=14),
])

tw0 = line.twiss(betx=1, bety=1)

env['translation'].shift_x = tw0['x', 'translation']
env['rotation'].rot_y_rad = np.asin(tw0['px', 'rotation'])

# Save the real line for the checks performed in 013b.
line.to_json('line.json')

# Survey of the real line
sv = line.survey(include_element_frames=True)

# Build a deep copy of the line and make it smooth (no jumps)
line_no_jumps = line.copy(shallow=False)

line_no_jumps['translation'].shift_x = 0
line_no_jumps['rotation'].rot_y_rad = 0

# Reset alignment data for elements in the smooth line
for elem_name in elements_to_process:
    ee_nj = line_no_jumps[elem_name]

    # MAD-X-style rbend description
    if hasattr(ee_nj, 'rbend_model'):
        ee_nj.rbend_model = 'curved-body'

    su.clear_element_misalignments(ee_nj)

# Survey and table of the smooth line
sv0_nj = line_no_jumps.survey(include_element_frames=True)
tt0_nj = line_no_jumps.get_table(attr=True)

# RST frame unit vectors for the smooth line
XYZ_rst_start = []
E_rst_start = []
for ii in range(len(sv0_nj)):
    XYZ_rst, E_rst = su.rst_from_reference_start(
        XYZ_ref_start=sv0_nj.XYZ_ref_start[ii],
        E_ref_start=sv0_nj.E_ref_start[ii],
        rot_s_rad=tt0_nj.rot_s_rad[ii],
        angle=tt0_nj.angle[ii],
    )
    XYZ_rst_start.append(XYZ_rst)
    E_rst_start.append(E_rst)
sv0_nj['XYZ_rst_start'] = np.array(XYZ_rst_start)
sv0_nj['E_rst_start'] = np.array(E_rst_start)

# Actual element displacements with respect to the smooth curve in RST coordinates
for nn in ['br_start', 'bs_start', 'bt_start', 'br_end', 'bs_end', 'bt_end']:
    sv0_nj[nn] = np.zeros(len(sv0_nj))
sv0_nj['bgamma'] = np.zeros(len(sv0_nj))

for ii, element in enumerate(line_no_jumps.elements):
    if element.allow_rot_and_shift:
        offset_start_rst, offset_end_rst, bgamma = (
            su.rst_start_end_offsets_tilt_from_positions(
                XYZ_rst_start=sv0_nj.XYZ_rst_start[ii],
                E_rst_start=sv0_nj.E_rst_start[ii],
                XYZ_elem_start=sv.XYZ_elem_start[ii],
                E_elem_start=sv.E_elem_start[ii],
                XYZ_elem_end=sv.XYZ_elem_end[ii],
                tilt=element.rot_s_rad,
                angle=getattr(element, 'angle', 0.0),
                rbend_angle=(
                    element.angle if isinstance(element, xt.RBend) else None),
            )
        )

        sv0_nj['br_start', ii] = offset_start_rst[0]
        sv0_nj['bs_start', ii] = offset_start_rst[1]
        sv0_nj['bt_start', ii] = offset_start_rst[2]
        sv0_nj['br_end', ii] = offset_end_rst[0]
        sv0_nj['bs_end', ii] = offset_end_rst[1]
        sv0_nj['bt_end', ii] = offset_end_rst[2]

        elem_name = sv0_nj.name[ii]
        if elem_name in elements_to_process:
            sv0_nj['bgamma', ii] = bgamma

# Output table
tt_align = xt.Table({
    'name': sv0_nj['name'],
    'element_type': sv0_nj['element_type'],

    # Reference trajectory (survey with no jumps)
    'X': sv0_nj['X'],
    'Y': sv0_nj['Y'],
    'Z': sv0_nj['Z'],
    'theta': sv0_nj['theta'],
    'phi': sv0_nj['phi'],
    'psi': sv0_nj['psi'],
    'angle': tt0_nj['angle'],
    'tilt': tt0_nj['rot_s_rad'],

    # Misalignment as RST offsets and tilt (SU convention)
    'br_start': sv0_nj['br_start'],
    'bs_start': sv0_nj['bs_start'],
    'bt_start': sv0_nj['bt_start'],
    'br_end': sv0_nj['br_end'],
    'bs_end': sv0_nj['bs_end'],
    'bt_end': sv0_nj['bt_end'],
    'bgamma': sv0_nj['bgamma'],

    # RST unit vectors at element start
    'E_rst_start': sv0_nj.E_rst_start,
})

tt_align.to_tfs('test_align.tfs', float_precision=17)

# Plot the real survey and the smooth reference used to define the RST offsets.
plt.close('all')
plt.figure(1)
plt.plot(sv.Z, sv.X, '.-', label='survey')
plt.plot(sv0_nj.Z, sv0_nj.X, '--', label='survey no jumps')

for elem_name in elements_to_process:
    plt.plot(
        [sv['Z_elem_start', elem_name], sv['Z_elem_end', elem_name]],
        [sv['X_elem_start', elem_name], sv['X_elem_end', elem_name]],
        'x-', color='g')
    plt.plot(
        [sv0_nj['Z_ref_start', elem_name],
         sv0_nj['Z_ref_end', elem_name]],
        [sv0_nj['X_ref_start', elem_name],
         sv0_nj['X_ref_end', elem_name]],
        '+--', color='orange')

    su.plot_exs(
        sv['E_elem_start', elem_name], sv['XYZ_elem_start', elem_name],
        length=0.5, color='g')
    su.plot_exs(
        sv['E_elem_end', elem_name], sv['XYZ_elem_end', elem_name],
        length=0.5, color='g')
    su.plot_exs(
        sv0_nj['E_ref_start', elem_name],
        sv0_nj['XYZ_ref_start', elem_name], length=0.5, color='orange')
    su.plot_exs(
        sv0_nj['E_ref_end', elem_name],
        sv0_nj['XYZ_ref_end', elem_name], length=0.5, color='orange')

plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.axis('equal')
plt.legend()

plt.show()
