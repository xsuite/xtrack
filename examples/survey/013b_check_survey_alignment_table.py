import json

import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt

from xtrack._temp import survey_utils as su


elements_to_process = ['bend_1', 'bend_2', 'bend_3', 'q1', 'q2']

line = xt.Line.from_json('line.json')
tt_align = xt.Table.from_tfs('test_align.tfs')

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

# Restore the RST geometry saved in the table. Multidimensional TFS columns
# are represented by JSON strings when loaded.
sv0_nj['XYZ_rst_start'] = np.column_stack((
    tt_align.x,
    tt_align.y,
    tt_align.z,
))
sv0_nj['E_rst_start'] = np.array([
    json.loads(value) for value in tt_align.e_rst_start
])

# Check the saved smooth reference trajectory and RST frames.
for saved_column, survey_column in {
        'x': 'X',
        'y': 'Y',
        'z': 'Z',
        'theta': 'theta',
        'phi': 'phi',
        'psi': 'psi',
}.items():
    xo.assert_allclose(
        tt_align[saved_column], sv0_nj[survey_column], atol=1e-12, rtol=0)

xo.assert_allclose(tt_align.angle, tt0_nj.angle, atol=1e-12, rtol=0)
xo.assert_allclose(tt_align.tilt, tt0_nj.rot_s_rad, atol=1e-12, rtol=0)

for ii in range(len(sv0_nj)):
    XYZ_rst, E_rst = su.rst_from_reference_start(
        XYZ_ref_start=sv0_nj.XYZ_ref_start[ii],
        E_ref_start=sv0_nj.E_ref_start[ii],
        rot_s_rad=tt0_nj.rot_s_rad[ii],
        angle=tt0_nj.angle[ii],
    )
    xo.assert_allclose(
        sv0_nj.XYZ_rst_start[ii], XYZ_rst, atol=1e-12, rtol=0)
    xo.assert_allclose(
        sv0_nj.E_rst_start[ii], E_rst, atol=1e-12, rtol=0)

# Reconstruct MAD-X-style misalignments from the saved RST offsets.
misalignments = {}
for elem_name in elements_to_process:
    ee_nj = line_no_jumps[elem_name]
    offset_start_rst = np.array([
        tt_align['br_start', elem_name],
        tt_align['bs_start', elem_name],
        tt_align['bt_start', elem_name],
    ])
    offset_end_rst = np.array([
        tt_align['br_end', elem_name],
        tt_align['bs_end', elem_name],
        tt_align['bt_end', elem_name],
    ])
    misalignment = su.misalignment_from_rst_offsets(
        offset_start_rst=offset_start_rst,
        offset_end_rst=offset_end_rst,
        bgamma=tt_align['bgamma', elem_name],
        tilt=ee_nj.rot_s_rad,
        angle=getattr(ee_nj, 'angle', 0.0),
    )
    misalignments[elem_name] = misalignment

    misalignment_from_absolute = su.misalignment_from_absolute_position(
        XYZ_elem_start=sv['XYZ_elem_start', elem_name],
        E_elem_start=sv['E_elem_start', elem_name],
        XYZ_ref_start=sv0_nj['XYZ_ref_start', elem_name],
        E_ref_start=sv0_nj['E_ref_start', elem_name],
        rbend_angle=(ee_nj.angle if isinstance(ee_nj, xt.RBend) else None),
    )
    xo.assert_allclose(
        np.array([
            misalignment.dtheta,
            misalignment.dphi,
            misalignment.dpsi,
            misalignment.shift_x,
            misalignment.shift_y,
            misalignment.shift_s,
        ]),
        np.array([
            misalignment_from_absolute.dtheta,
            misalignment_from_absolute.dphi,
            misalignment_from_absolute.dpsi,
            misalignment_from_absolute.shift_x,
            misalignment_from_absolute.shift_y,
            misalignment_from_absolute.shift_s,
        ]),
        atol=1e-12,
        rtol=0,
    )
    misalignment.apply_to_element(line_no_jumps[elem_name])

sv_nj = line_no_jumps.survey(include_element_frames=True)
tt_nj = line_no_jumps.get_table(attr=True)

for survey_column, element_attribute in {
        'dtheta': 'rot_y_rad',
        'dphi': 'rot_x_rad',
        'dpsi': 'rot_s_rad_no_frame',
        'shift_x': 'shift_x',
        'shift_y': 'shift_y',
        'shift_s': 'shift_s',
}.items():
    for elem_name in elements_to_process:
        expected = getattr(misalignments[elem_name], survey_column)
        if survey_column == 'dpsi':
            expected -= line_no_jumps[elem_name].rot_s_rad
        xo.assert_allclose(
            expected, tt_nj[element_attribute, elem_name],
            atol=1e-12, rtol=0)

for ii in range(len(sv0_nj)):
    chord = sv0_nj.XYZ_ref_end[ii] - sv0_nj.XYZ_ref_start[ii]
    chord_length = np.linalg.norm(chord)
    if chord_length > 0:
        chord /= chord_length
        xo.assert_allclose(
            sv0_nj.E_rst_start[ii, :, 1], chord, atol=1e-12, rtol=0)

b_E_rst = []
b_S_rst = []
element_lengths = []
for elem_name in elements_to_process:
    XYZ_rst = sv0_nj['XYZ_rst_start', elem_name]
    E_rst = sv0_nj['E_rst_start', elem_name]

    # E is the entree (element start), S is the sortie (element end).
    element = line_no_jumps[elem_name]
    b_E, b_S, bgamma = su.rst_start_end_offsets_tilt_from_positions(
        XYZ_rst_start=XYZ_rst,
        E_rst_start=E_rst,
        XYZ_elem_start=sv['XYZ_elem_start', elem_name],
        E_elem_start=sv['E_elem_start', elem_name],
        XYZ_elem_end=sv['XYZ_elem_end', elem_name],
        tilt=element.rot_s_rad,
        angle=getattr(element, 'angle', 0.0),
        rbend_angle=(
            element.angle if isinstance(element, xt.RBend) else None),
    )

    length = np.linalg.norm(
        sv['XYZ_elem_end', elem_name]
        - sv['XYZ_elem_start', elem_name]
    )
    b_E_formula, b_S_formula = su.rst_start_end_offsets_from_parameters(
        line_no_jumps[elem_name], length)

    xo.assert_allclose(b_E, b_E_formula, atol=1e-12, rtol=0)
    xo.assert_allclose(b_S, b_S_formula, atol=1e-12, rtol=0)
    xo.assert_allclose(
        bgamma, tt_align['bgamma', elem_name], atol=1e-12, rtol=0)

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
    plt.plot(
        [sv_nj['Z_elem_start', elem_name],
         sv_nj['Z_elem_end', elem_name]],
        [sv_nj['X_elem_start', elem_name],
         sv_nj['X_elem_end', elem_name]],
        '.--', color='r')

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
    su.plot_exs(
        sv_nj['E_elem_start', elem_name],
        sv_nj['XYZ_elem_start', elem_name], length=0.3, color='red')
    su.plot_exs(
        sv_nj['E_elem_end', elem_name],
        sv_nj['XYZ_elem_end', elem_name], length=0.3, color='red')

plt.xlabel('Z [m]')
plt.ylabel('X [m]')
plt.axis('equal')
plt.legend()

plt.figure(2)
for elem_name in elements_to_process:
    su.plot_exy(
        sv['E_elem_start', elem_name], sv['XYZ_elem_start', elem_name],
        length=0.5, color='g')
    su.plot_exy(
        sv_nj['E_elem_start', elem_name],
        sv_nj['XYZ_elem_start', elem_name], length=0.3, color='orange')

plt.xlabel('X [m]')
plt.ylabel('Y [m]')

plt.show()
