import xtrack as xt
import xobjects as xo
import numpy as np

from ldbpoint import MADPoint
from survey_utils import (
    misalignment_from_absolute_position,
    plot_exy,
    plot_exz,
)

elements_to_process = ['bend_1', 'bend_2', 'bend_3', 'q1', 'q2']


def rst_from_reference_start(
        XYZ_ref_start, E_ref_start, rot_s_rad, angle):
    p_ref_start = np.eye(4)
    p_ref_start[:3, :3] = E_ref_start
    p_ref_start[:3, 3] = XYZ_ref_start

    p_rst_start = MADPoint(p_ref_start)
    p_rst_start.rpsi(rot_s_rad)
    p_rst_start.rtheta(-angle / 2)

    # S is along the chord, T is normal to the curvature plane, and R = S x T.
    es = p_rst_start.matrix[:3, 2]
    et = p_rst_start.matrix[:3, 1]
    er = np.cross(es, et)

    E_rst_start = np.column_stack((er, es, et))
    return p_rst_start.matrix[:3, 3].copy(), E_rst_start


def rst_endpoint_offsets_from_parameters(element, length):
    angle = getattr(element, 'angle', 0.0)

    rotation_tilt = MADPoint.psi_matrix(
        element.rot_s_rad)[:3, :3]
    rotation_half_angle = MADPoint.theta_matrix(
        -angle / 2)[:3, :3]
    rotation_misalignment = (
        MADPoint.theta_matrix(element.rot_y_rad)
        @ MADPoint.phi_matrix(element.rot_x_rad)
        @ MADPoint.psi_matrix(element.rot_s_rad_no_frame)
    )[:3, :3]

    # E_rst = E_ref @ rotation_tilt @ rotation_half_angle @ xys_from_rst.
    # The columns of xys_from_rst are R=-x, S=s, T=y, expressed in the
    # tilted chord frame.
    xys_from_rst = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    rst_from_xys = (
        xys_from_rst.T
        @ rotation_half_angle.T
        @ rotation_tilt.T
    )

    # The screenshot uses (DX, DS, DY). Xtrack uses (x, y, s), hence the
    # corresponding vectors below are (DX, DY, DS) and (0, 0, l_E).
    displacement_E_xys = np.array([
        element.shift_x,
        element.shift_y,
        element.shift_s,
    ])
    body_chord_xys = np.array([0.0, 0.0, length])

    b_E = rst_from_xys @ displacement_E_xys
    b_S = rst_from_xys @ (
        displacement_E_xys
        + rotation_misalignment
        @ rotation_tilt
        @ rotation_half_angle
        @ body_chord_xys
    )
    return b_E, b_S


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

for elem_name in elements_to_process:
    ee_nj = line_no_jumps[elem_name]

    XYZ_elem_start = sv['XYZ_elem_start', elem_name]
    E_elem_start = sv['E_elem_start', elem_name]
    XYZ_elem_end = sv['XYZ_elem_end', elem_name]
    E_elem_end = sv['E_elem_end', elem_name]

    XYZ_nj_ref_start = sv0_nj['XYZ_ref_start', elem_name]
    E_nj_ref_start = sv0_nj['E_ref_start', elem_name]

    p_mat_elem_start = np.eye(4)
    p_mat_elem_start[:3, :3] = E_elem_start
    p_mat_elem_start[:3, 3] = XYZ_elem_start

    if isinstance(ee_nj, xt.RBend):
        mp_elem_start = MADPoint(p_mat_elem_start)
        mp_elem_start.rtheta(ee_nj.angle/2)
        E_elem_start_rot = mp_elem_start.matrix[:3, :3]
        XYZ_elem_start_rot = mp_elem_start.matrix[:3, 3]
    else:
        E_elem_start_rot = E_elem_start
        XYZ_elem_start_rot = XYZ_elem_start

    misalignment = misalignment_from_absolute_position(
        XYZ_elem_start=XYZ_elem_start_rot,
        E_elem_start=E_elem_start_rot,
        XYZ_ref_start=XYZ_nj_ref_start,
        E_ref_start=E_nj_ref_start
    )

    misalignment.apply_to_element(ee_nj)

sv_nj = line_no_jumps.survey(include_element_frames=True)
tt_nj = line_no_jumps.get_table(attr=True)

XYZ_rst_start = []
E_rst_start = []
for elem_name in elements_to_process:
    ee_nj = line_no_jumps[elem_name]
    XYZ_rst, E_rst = rst_from_reference_start(
        XYZ_ref_start=sv_nj['XYZ_ref_start', elem_name],
        E_ref_start=sv_nj['E_ref_start', elem_name],
        rot_s_rad=ee_nj.rot_s_rad,
        angle=getattr(ee_nj, 'angle', 0.0),
    )
    XYZ_rst_start.append(XYZ_rst)
    E_rst_start.append(E_rst)

    chord = (
        sv_nj['XYZ_ref_end', elem_name]
        - sv_nj['XYZ_ref_start', elem_name]
    )
    chord /= np.linalg.norm(chord)
    xo.assert_allclose(E_rst[:, 1], chord, atol=1e-12, rtol=0)

tt_rst = xt.Table({
    'name': np.array(elements_to_process),
    'XYZ': np.array(XYZ_rst_start),
    'E': np.array(E_rst_start),
})

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

XYZ_rst_start_all = []
E_rst_start_all = []
for ii in range(len(sv_nj)):
    XYZ_rst, E_rst = rst_from_reference_start(
        XYZ_ref_start=sv_nj.XYZ_ref_start[ii],
        E_ref_start=sv_nj.E_ref_start[ii],
        rot_s_rad=tt_nj.rot_s_rad[ii],
        angle=tt_nj.angle[ii],
    )
    XYZ_rst_start_all.append(XYZ_rst)
    E_rst_start_all.append(E_rst)

XYZ_rst_start_all = np.array(XYZ_rst_start_all)
E_rst_start_all = np.array(E_rst_start_all)

displacement_start = sv.XYZ_elem_start - XYZ_rst_start_all
displacement_end = sv.XYZ_elem_end - XYZ_rst_start_all
offset_start_rst = np.einsum(
    'nij,ni->nj', E_rst_start_all, displacement_start)
offset_end_rst = np.einsum(
    'nij,ni->nj', E_rst_start_all, displacement_end)

supports_misalignment = np.zeros(len(sv_nj), dtype=bool)
for ii, element in enumerate(line_no_jumps.elements):
    supports_misalignment[ii] = element.allow_rot_and_shift

offset_start_rst[~supports_misalignment] = 0.0
offset_end_rst[~supports_misalignment] = 0.0

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
    'E_rst_start': E_rst_start_all,
})

tt_align.to_tfs('test_align.tfs')


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
