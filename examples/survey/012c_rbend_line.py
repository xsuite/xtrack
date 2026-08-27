import xtrack as xt
import xobjects as xo
import numpy as np

from survey_utils import (
    misalignment_from_absolute_position,
    plot_exy,
    plot_exz,
    rst_endpoint_offsets_from_parameters,
    rst_from_reference_start,
)

elements_to_process = ['bend_1', 'bend_2', 'bend_3', 'q1', 'q2']

env = xt.Environment()
env.set_particle_ref('proton', p0c=400e9)

env.new('bend_1', 'RBend', length_straight=2,
        angle=0., # This dipole does not bend the reference frame
        k0=0.1,
        rbend_compensate_sagitta=False,
        rot_shift_anchor=1., # shift defined in the middle
        shift_x=0.5)

env.new('bend_2', 'Bend', length=2,
        angle=-0.2, # This dipole bends the reference frame
        rot_s_rad = np.deg2rad(15),
        rot_shift_anchor=1., # shift defined in the middle
        rot_y_rad=np.deg2rad(10),
        shift_x=0.5
)
env.new('bend_3', 'RBend', length_straight=2,
        angle=0.2, # This dipole bends the reference frame
        rot_s_rad = np.deg2rad(15),
        # rbend_model='straight-body',
        rbend_compensate_sagitta=True,
        rot_shift_anchor=1., # shift defined in the middle
        rot_y_rad=np.deg2rad(-20),
        shift_x=-0.1
)

env.new('q1', 'Quadrupole', length=0.7, k1=0.1)
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

sv = line.survey(include_element_frames=True)

line_no_jumps = line.copy(shallow=False)

line_no_jumps['translation'].shift_x = 0
line_no_jumps['rotation'].rot_y_rad = 0

for elem_name in elements_to_process:
    ee_nj = line_no_jumps[elem_name]

    if hasattr(ee_nj, 'rbend_model'):
        ee_nj.rbend_model = 'curved-body'
    ee_nj.rot_x_rad = 0
    ee_nj.rot_y_rad = 0
    ee_nj.rot_s_rad_no_frame = 0
    ee_nj.shift_x = 0
    ee_nj.shift_y = 0
    ee_nj.shift_s = 0
    ee_nj.rot_shift_anchor = 0

sv0_nj = line_no_jumps.survey(include_element_frames=True)
tt0_nj = line_no_jumps.get_table(attr=True)

XYZ_rst_start = []
E_rst_start = []
for ii in range(len(sv0_nj)):
    XYZ_rst, E_rst = rst_from_reference_start(
        XYZ_ref_start=sv0_nj.XYZ_ref_start[ii],
        E_ref_start=sv0_nj.E_ref_start[ii],
        rot_s_rad=tt0_nj.rot_s_rad[ii],
        angle=tt0_nj.angle[ii],
    )
    XYZ_rst_start.append(XYZ_rst)
    E_rst_start.append(E_rst)

tt_rst = xt.Table({
    'name': sv0_nj.name,
    'XYZ': np.array(XYZ_rst_start),
    'E': np.array(E_rst_start),
})

displacement_start = sv.XYZ_elem_start - tt_rst.XYZ
displacement_end = sv.XYZ_elem_end - tt_rst.XYZ
offset_start_rst = np.einsum(
    'nij,ni->nj', tt_rst.E, displacement_start)
offset_end_rst = np.einsum(
    'nij,ni->nj', tt_rst.E, displacement_end)

supports_misalignment = np.zeros(len(tt_rst), dtype=bool)
for ii, element in enumerate(line_no_jumps.elements):
    supports_misalignment[ii] = element.allow_rot_and_shift

offset_start_rst[~supports_misalignment] = 0.0
offset_end_rst[~supports_misalignment] = 0.0

for elem_name in elements_to_process:
    ee_nj = line_no_jumps[elem_name]

    misalignment = misalignment_from_absolute_position(
        XYZ_elem_start=sv['XYZ_elem_start', elem_name],
        E_elem_start=sv['E_elem_start', elem_name],
        XYZ_ref_start=sv0_nj['XYZ_ref_start', elem_name],
        E_ref_start=sv0_nj['E_ref_start', elem_name],
        rbend_angle=(ee_nj.angle if isinstance(ee_nj, xt.RBend) else None),
    )

    misalignment.apply_to_element(ee_nj)

sv_nj = line_no_jumps.survey(include_element_frames=True)
tt_nj = line_no_jumps.get_table(attr=True)

tt_align = xt.Table({
    'name': sv_nj['name'],
    'element_type': sv_nj['element_type'],
    # Reference trajectory (survey with no jumps)
    'X': sv_nj['X'],
    'Y': sv_nj['Y'],
    'Z': sv_nj['Z'],
    'theta': sv_nj['theta'],
    'phi': sv_nj['phi'],
    'psi': sv_nj['psi'],
    'angle': tt_nj['angle'],
    'tilt': tt_nj['rot_s_rad'],

    # Misalignment parameters (MAD-X convention)
    'dtheta': tt_nj['rot_y_rad'],
    'dphi': tt_nj['rot_x_rad'],
    'dpsi': tt_nj['rot_s_rad_no_frame'],
    'dx': tt_nj['shift_x'],
    'dy': tt_nj['shift_y'],
    'ds': tt_nj['shift_s'],

    # Misalignment as RST offsets and tilt (SU convention)
    'br_start': offset_start_rst[:, 0],
    'bs_start': offset_start_rst[:, 1],
    'bt_start': offset_start_rst[:, 2],
    'br_end': offset_end_rst[:, 0],
    'bs_end': offset_end_rst[:, 1],
    'bt_end': offset_end_rst[:, 2],
    'bgamma': -tt_nj['rot_s_rad_no_frame'],

    # RST unit vectors at element start
    'E_rst_start': tt_rst.E,
})

tt_align.to_tfs('test_align.tfs')

# Checks

for ii in range(len(sv0_nj)):
    chord = sv0_nj.XYZ_ref_end[ii] - sv0_nj.XYZ_ref_start[ii]
    chord_length = np.linalg.norm(chord)
    if chord_length > 0:
        chord /= chord_length
        xo.assert_allclose(
            tt_rst.E[ii, :, 1], chord, atol=1e-12, rtol=0)

b_E_rst = []
b_S_rst = []
element_lengths = []
for elem_name in elements_to_process:
    XYZ_rst = tt_rst['XYZ', elem_name]
    E_rst = tt_rst['E', elem_name]

    # E is the entree (element start), S is the sortie (element end).
    displacement_E = sv['XYZ_elem_start', elem_name] - XYZ_rst
    displacement_S = sv['XYZ_elem_end', elem_name] - XYZ_rst
    b_E = E_rst.T @ displacement_E
    b_S = E_rst.T @ displacement_S

    length = np.linalg.norm(
        sv['XYZ_elem_end', elem_name]
        - sv['XYZ_elem_start', elem_name]
    )
    b_E_formula, b_S_formula = rst_endpoint_offsets_from_parameters(
        line_no_jumps[elem_name], length)

    xo.assert_allclose(b_E, b_E_formula, atol=1e-12, rtol=0)
    xo.assert_allclose(b_S, b_S_formula, atol=1e-12, rtol=0)

    b_E_rst.append(b_E)
    b_S_rst.append(b_S)
    element_lengths.append(length)

b_E_rst = np.array(b_E_rst)
b_S_rst = np.array(b_S_rst)

tt_misalignment_rst = xt.Table({
    'name': np.array(elements_to_process),
    'b_E': b_E_rst,
    'b_S': b_S_rst,
    'b_EX': -b_E_rst[:, 0],
    'b_EY': b_E_rst[:, 1],
    'b_EZ': b_E_rst[:, 2],
    'b_SX': -b_S_rst[:, 0],
    'b_SY': b_S_rst[:, 1],
    'b_SZ': b_S_rst[:, 2],
    'l_E': np.array(element_lengths),
})

for elem_name in elements_to_process:
    xo.assert_allclose(
        sv_nj['XYZ_elem_start', elem_name],
        sv['XYZ_elem_start', elem_name],
        atol=1e-12,
        rtol=0,
    )
    xo.assert_allclose(
        sv_nj['E_elem_start', elem_name],
        sv['E_elem_start', elem_name],
        atol=1e-12,
        rtol=0,
    )
    xo.assert_allclose(
        sv_nj['XYZ_elem_end', elem_name],
        sv['XYZ_elem_end', elem_name],
        atol=1e-12,
        rtol=0,
    )
    xo.assert_allclose(
        sv_nj['E_elem_end', elem_name],
        sv['E_elem_end', elem_name],
        atol=1e-12,
        rtol=0,
    )

import matplotlib.pyplot as plt


plt.close('all')
plt.figure(1)
plt.plot(sv.Z, sv.X, '.-', label='survey')
plt.plot(sv0_nj.Z, sv0_nj.X, '--', label='survey no jumps')

for elem_name in elements_to_process:
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
    plot_exz(sv0_nj['E_ref_start', elem_name], sv0_nj['XYZ_ref_start', elem_name], length=0.5, color='orange')
    plot_exz(sv0_nj['E_ref_end', elem_name], sv0_nj['XYZ_ref_end', elem_name], length=0.5, color='orange')
    plot_exz(sv_nj['E_elem_start', elem_name], sv_nj['XYZ_elem_start', elem_name], length=0.3, color='red')
    plot_exz(sv_nj['E_elem_end', elem_name], sv_nj['XYZ_elem_end', elem_name], length=0.3, color='red')
    # plot_exz(E_elem_start_rot, XYZ_elem_start_rot, length=0.5, color='r')

plot_exz(sv['E_elem_start', elem_name], sv['XYZ_elem_start', elem_name], length=0.5, color='g')
plot_exz(sv['E_elem_end', elem_name], sv['XYZ_elem_end', elem_name], length=0.5, color='g')
plot_exz(sv0_nj['E_ref_start', elem_name], sv0_nj['XYZ_ref_start', elem_name], length=0.5, color='orange')
plot_exz(sv0_nj['E_ref_end', elem_name], sv0_nj['XYZ_ref_end', elem_name], length=0.5, color='orange')
plot_exz(sv_nj['E_elem_start', elem_name], sv_nj['XYZ_elem_start', elem_name], length=0.3, color='red')
plot_exz(sv_nj['E_elem_end', elem_name], sv_nj['XYZ_elem_end', elem_name], length=0.3, color='red')
# plot_exz(E_elem_start_rot, XYZ_elem_start_rot, length=0.5, color='r')

plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.axis('equal')
plt.legend()

plt.figure(2)
for elem_name in elements_to_process:
    plot_exy(sv['E_elem_start', elem_name], sv['XYZ_elem_start', elem_name], length=0.5, color='g')
    plot_exy(sv_nj['E_elem_start', elem_name], sv_nj['XYZ_elem_start', elem_name], length=0.3, color='orange')

plt.xlabel('X [m]')
plt.ylabel('Y [m]')

plt.show()
