from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..survey.frame import Frame


_LEGACY_SURVEY_TFS_HEADER = """
@ NAME             %06s "SURVEY"
@ TYPE             %06s "SURVEY"
@ TITLE            %08s "no-title"
@ ORIGIN           %17s "5.09.03 Darwin 64"
@ DATE             %08s "{date}"
@ TIME             %08s "{time}"
* NAME               KEYWORD                                    S                         L                     ANGLE                         X                         Y                         Z                     THETA                       PHI                       PSI                GLOBALTILT                      TILT    SLOT_ID ASSEMBLY_ID                  MECH_SEP                     V_POS
$ %s                 %s                                       %le                       %le                       %le                       %le                       %le                       %le                       %le                       %le                       %le                       %le                       %le         %d         %d                       %le                       %le
""".lstrip()


@dataclass
class Misalignment:
    dtheta: float
    dphi: float
    dpsi: float
    shift_x: float
    shift_y: float
    shift_s: float

    def apply_to_element(self, element):
        if (hasattr(element, 'rbend_model')
                and element.rbend_model == 'straight-body'):
            raise ValueError(
                'straight-body rbends not yet supported for misalignment '
                'application')
        element.rot_shift_anchor = 0.  # Defined at the entrance
        element.rot_y_rad = self.dtheta
        element.rot_x_rad = self.dphi
        element.rot_s_rad_no_frame = self.dpsi - element.rot_s_rad
        element.shift_x = self.shift_x
        element.shift_y = self.shift_y
        element.shift_s = self.shift_s


def clear_element_misalignments(element):
    element.rot_x_rad = 0
    element.rot_y_rad = 0
    element.rot_s_rad_no_frame = 0
    element.shift_x = 0
    element.shift_y = 0
    element.shift_s = 0
    element.rot_shift_anchor = 0


def misalignment_from_absolute_position(
        XYZ_elem_start, E_elem_start, XYZ_ref_start, E_ref_start,
        rbend_angle=None):
    """Infer MAD-X misalignments from absolute entrance position and frame.

    For an RBend, ``rbend_angle`` applies the half-angle transformation from
    its entrance frame to the frame used by the MAD-X misalignment convention.
    """
    frame_elem_start = Frame.from_survey(XYZ_elem_start, E_elem_start)

    if rbend_angle is not None:
        frame_elem_start.rotate_y(rbend_angle / 2)

    frame_ref_start = Frame.from_survey(XYZ_ref_start, E_ref_start)

    relative_frame = frame_ref_start.inverse() @ frame_elem_start
    A = relative_frame.matrix

    theta = np.arctan2(A[0, 2], A[2, 2])
    phi = np.arctan2(A[1, 2], np.sqrt(A[1, 0]**2 + A[1, 1]**2))
    psi = np.arctan2(A[1, 0], A[1, 1])
    shift_x = A[0, 3]
    shift_y = A[1, 3]
    shift_s = A[2, 3]

    return Misalignment(
        dtheta=theta,
        dphi=phi,
        dpsi=psi,
        shift_x=shift_x,
        shift_y=shift_y,
        shift_s=shift_s,
    )


def rst_from_reference_start(
        XYZ_ref_start, E_ref_start, rot_s_rad, angle):
    frame_rst_start = Frame.from_survey(XYZ_ref_start, E_ref_start)
    frame_rst_start.rotate_s(rot_s_rad)
    frame_rst_start.rotate_y(-angle / 2)

    # S is along the chord, T is normal to the curvature plane, and R = S x T.
    es = frame_rst_start.es
    et = frame_rst_start.ey
    er = np.cross(es, et)

    E_rst_start = np.column_stack((er, es, et))
    return frame_rst_start.XYZ.copy(), E_rst_start


def _rst_transform_frames(tilt, angle):
    tilted_chord_frame = Frame().rotate_s(tilt).rotate_y(-angle / 2)

    # This frame maps RST coordinates to the tilted chord coordinates.
    # The columns are R=-x, S=s, T=y, expressed in the tilted chord frame.
    rst_basis_frame = Frame.from_survey(
        XYZ=np.zeros(3),
        E_matrix=np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]),
    )
    xys_from_rst_frame = tilted_chord_frame @ rst_basis_frame
    return tilted_chord_frame, xys_from_rst_frame


def rst_start_end_offsets_tilt_from_positions(
        XYZ_rst_start, E_rst_start, XYZ_elem_start, E_elem_start,
        XYZ_elem_end, tilt=0.0, angle=0.0, rbend_angle=None):
    """Return RST endpoint offsets and bgamma from element survey data."""
    displacement_start = XYZ_elem_start - XYZ_rst_start
    displacement_end = XYZ_elem_end - XYZ_rst_start
    offset_start_rst = E_rst_start.T @ displacement_start
    offset_end_rst = E_rst_start.T @ displacement_end

    _, xys_from_rst_frame = _rst_transform_frames(tilt, angle)
    rst_start_frame = Frame.from_survey(XYZ_rst_start, E_rst_start)
    reference_start_frame = (
        rst_start_frame @ xys_from_rst_frame.inverse())
    element_start_frame = Frame.from_survey(XYZ_elem_start, E_elem_start)
    if rbend_angle is not None:
        element_start_frame.rotate_y(rbend_angle / 2)

    relative_frame = reference_start_frame.inverse() @ element_start_frame
    bgamma = -(relative_frame.psi - tilt)
    return offset_start_rst, offset_end_rst, bgamma


def misalignment_from_rst_offsets(
        offset_start_rst, offset_end_rst, bgamma, tilt=0.0, angle=0.0):
    """Infer MAD-X misalignments from RST entry and exit offsets.

    The endpoint offsets determine the translation and displaced chord
    direction. ``bgamma`` supplies the rotation about the chord, which cannot
    be inferred from endpoint positions alone. The principal-angle solution
    is returned. Following the SU convention, ``bgamma`` is the negative of
    the additional longitudinal rotation; ``tilt`` is the reference tilt.
    The returned ``dpsi`` includes that tilt, consistently with
    :func:`misalignment_from_absolute_position`.
    """
    offset_start_rst = np.asarray(offset_start_rst)
    offset_end_rst = np.asarray(offset_end_rst)

    tilted_chord_frame, xys_from_rst_frame = (
        _rst_transform_frames(tilt, angle))

    displacement_xys = xys_from_rst_frame.E_matrix @ offset_start_rst
    displaced_chord_rst = offset_end_rst - offset_start_rst
    length = np.linalg.norm(displaced_chord_rst)
    if length <= 0:
        raise ValueError('entry and exit offsets must define a chord')
    displaced_chord_xys = xys_from_rst_frame.E_matrix @ displaced_chord_rst

    rot_s_rad_no_frame = -bgamma
    longitudinal_rotation = Frame().rotate_s(rot_s_rad_no_frame)
    chord_before_theta_phi = (
        longitudinal_rotation @ tilted_chord_frame).es * length

    uy = chord_before_theta_phi[1]
    uz = chord_before_theta_phi[2]
    yz_norm = np.hypot(uy, uz)
    if yz_norm <= np.finfo(float).eps * length:
        raise ValueError('cannot infer dphi for a chord parallel to x')

    phi_reference = np.arctan2(uy, uz)
    dphi = (
        np.arcsin(np.clip(displaced_chord_xys[1] / yz_norm, -1.0, 1.0))
        - phi_reference
    )
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))

    chord_after_phi_z = -np.sin(dphi) * uy + np.cos(dphi) * uz
    dtheta = (
        np.arctan2(displaced_chord_xys[0], displaced_chord_xys[2])
        - np.arctan2(chord_before_theta_phi[0], chord_after_phi_z)
    )
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

    return Misalignment(
        dtheta=dtheta,
        dphi=dphi,
        dpsi=tilt + rot_s_rad_no_frame,
        shift_x=displacement_xys[0],
        shift_y=displacement_xys[1],
        shift_s=displacement_xys[2],
    )


def rst_rigid_chord(displ_start_rst, displ_end_rst, length):
    """Chord of a rigidly displaced element, in its own RST frame.

    ``displ_start_rst`` and ``displ_end_rst`` are the displacements of the
    element entrance and exit from their nominal positions, as given in an
    alignment request. Their transverse (R, T) components fix the direction of
    the chord, whose S component then follows from the nominal chord
    ``length`` being preserved. The requested exit S displacement plays no
    part: for a rigid element it is determined by the other two, not free.
    """
    displ_start_rst = np.asarray(displ_start_rst, dtype=float)
    displ_end_rst = np.asarray(displ_end_rst, dtype=float)

    chord_r = displ_end_rst[0] - displ_start_rst[0]
    chord_t = displ_end_rst[2] - displ_start_rst[2]
    transverse_sq = chord_r**2 + chord_t**2
    if transverse_sq >= length**2:
        raise ValueError(
            'transverse displacement larger than the element chord')

    return np.array([chord_r, np.sqrt(length**2 - transverse_sq), chord_t])


def misalignment_from_rst_displacements(
        displ_start_rst, displ_end_rst, length, bgamma, tilt=0.0, angle=0.0):
    """Infer MAD-X misalignments from requested RST end-point displacements.

    This is the entry point for an alignment (bump) request, which gives the
    displacement of each element end point from its nominal position, rather
    than the absolute offsets taken by
    :func:`misalignment_from_rst_offsets`.

    The two displacements and ``bgamma`` are seven numbers, while a rigid body
    has only six degrees of freedom, so the request is over-determined: in
    general no rigid motion puts both end points exactly where asked. The
    element is kept rigid here, at its nominal chord ``length``; see
    :func:`rst_rigid_chord` for which part of the request that discards.
    """
    displ_start_rst = np.asarray(displ_start_rst, dtype=float)
    chord_rst = rst_rigid_chord(displ_start_rst, displ_end_rst, length)

    # A nominal element runs from (0, 0, 0) to (0, length, 0) in its own RST
    # frame, independently of its tilt and bending angle.
    return misalignment_from_rst_offsets(
        offset_start_rst=displ_start_rst,
        offset_end_rst=displ_start_rst + chord_rst,
        bgamma=bgamma,
        tilt=tilt,
        angle=angle,
    )


def misalignment_from_geode_displacements(
        displ_start_rst, displ_end_rst, length, bgamma, tilt=0.0, angle=0.0):
    """Convert GEODE displacements and roll to MAD-X alignment parameters.

    The displacements are geometric RST vectors in the nominal tilted chord
    frame. A report's radial *deviation* must therefore already be negated.
    ``length`` is the chord length; ``angle`` is the RBend angle (zero for a
    straight element), and ``tilt`` is its nominal longitudinal rotation.

    Infer the crab in the chord frame, independently of ``bgamma``. As in
    :func:`rst_rigid_chord`, exit S is determined by rigidity. Compose this
    crab with the additional rotation ``-bgamma`` about the entrance reference
    tangent. With B the nominal chord frame and C_local the crab in that
    frame, ``C = B C_local B^-1`` expresses the crab in the entrance reference
    frame. With R the tangent roll, the final chord orientation is ``C R B``.

    Unlike :func:`misalignment_from_rst_displacements`, a rolled bend need
    not retain the transverse exit position specified before adding roll.
    Entrance translation and chord length are preserved. The returned dpsi
    includes design tilt and the crab/half-bend-angle coupling; in general it
    is not simply ``tilt - bgamma``.
    """
    # A rigid magnet needs a positive chord length.
    if length <= 0:
        raise ValueError('length must be positive')

    # The input is already geometric RST: the caller has corrected the
    # bump-report radial sign.
    displ_start_rst = np.asarray(displ_start_rst, dtype=float)

    # Infer the entrance-to-exit vector before the additional roll.
    # For entrance/exit displacements (r0, s0, t0) and (r1, s1, t1):
    #   chord_r = r1 - r0, chord_t = t1 - t0,
    #   chord_s = sqrt(length**2 - chord_r**2 - chord_t**2).
    # Rigidity determines chord_s, so the requested exit S is ignored.
    chord_r, chord_s, chord_t = rst_rigid_chord(
        displ_start_rst, displ_end_rst, length)

    # Rotations of the element chord from its nominal direction to the
    # misaligned direction implied by the endpoint displacements,
    # before applying the additional roll.

    # theta and phi are defined with respect to x = -R, y = T, s = S.
    theta = np.arctan2(-chord_r, chord_s)
    phi = np.arctan2(chord_t, np.hypot(chord_r, chord_s))

    # C_local transforms coordinates from the misaligned chord frame (before
    # additional roll) to the nominal chord frame.
    C_local = Frame().rotate_y(theta).rotate_x(-phi)

    # B transforms coordinates from the nominal chord frame to the
    # reference frame tangent to the trajectory at the element entrance.
    B, xys_from_rst = _rst_transform_frames(tilt, angle)

    # Additional roll R about the nominal entrance tangent, with the
    # GEODE-to-Xsuite sign convention.
    R = Frame().rotate_s(-bgamma)

    # The same crab rotation expressed in the entrance reference frame.
    C = B @ C_local @ B.inverse()

    # Apply the roll to the nominal chord frame, then apply the crab.
    misaligned_chord_frame = C @ R @ B

    # For a bend, rolling about the entrance tangent can move the exit.
    # The crab is not adjusted to cancel this motion.

    # Place the entrance at its requested displacement. xys_from_rst maps
    # RST to reference coordinates, including x = -R, y = T, s = S in the
    # chord frame and the nominal orientation B. The rotations above leave
    # the entrance at the origin; the exit and sockets follow the rigid body.
    misaligned_chord_frame.XYZ = xys_from_rst.E_matrix @ displ_start_rst

    # The frame holds the displaced entrance position and chord orientation.
    # Convert these to Xsuite/MAD-X shifts and rotation parameters, including
    # the RBend half-angle conversion from chord to entrance orientation.
    return misalignment_from_absolute_position(
        XYZ_elem_start=misaligned_chord_frame.XYZ,
        E_elem_start=misaligned_chord_frame.E_matrix,
        XYZ_ref_start=np.zeros(3),
        E_ref_start=np.eye(3),
        rbend_angle=angle,
    )


def rst_start_end_offsets_from_parameters(element, length):
    angle = getattr(element, 'angle', 0.0)

    tilted_chord_frame, xys_from_rst_frame = (
        _rst_transform_frames(element.rot_s_rad, angle))
    rst_from_xys_frame = xys_from_rst_frame.inverse()

    frame_misalignment = Frame()
    frame_misalignment.rotate_y(element.rot_y_rad)
    frame_misalignment.rotate_x(-element.rot_x_rad)
    frame_misalignment.rotate_s(element.rot_s_rad_no_frame)
    displaced_chord_frame = frame_misalignment @ tilted_chord_frame

    # The screenshot uses (DX, DS, DY). Xtrack uses (x, y, s), hence the
    # corresponding vectors below are (DX, DY, DS) and (0, 0, l_E).
    displacement_E_xys = np.array([
        element.shift_x,
        element.shift_y,
        element.shift_s,
    ])
    b_E = rst_from_xys_frame.E_matrix @ displacement_E_xys
    b_S = rst_from_xys_frame.E_matrix @ (
        displacement_E_xys + displaced_chord_frame.es * length)
    return b_E, b_S

def comp_psi_vbend(frame_start, frame_end, psi_tol_deg=20):
    psi = frame_start.psi
    if np.isclose(np.abs(psi), np.pi/2, atol=np.deg2rad(psi_tol_deg)):
        assert np.isclose(np.abs(frame_end.psi), np.pi/2, atol=np.deg2rad(psi_tol_deg))
        # Done inplace
        frame_start.rotate_s(-np.sign(psi) * np.pi/2)
        frame_end.rotate_s(-np.sign(psi) * np.pi/2)


def write_legacy_survey_tfs(
        file_name, *, survey, element_names, element_container,
        compensate_psi_vbend=False, psi_tol_deg=20):
    """Write element entrance and exit frames in the legacy survey format."""
    lines = []
    for ii, nn in enumerate(element_names):
        frames = survey.get_all_frames(nn)

        if compensate_psi_vbend:
            ff_elem_start = frames['elem_start']
            ff_elem_end = frames['elem_end']
            comp_psi_vbend(ff_elem_start, ff_elem_end, psi_tol_deg=psi_tol_deg)

        for place in ('start', 'end'):

            if place == 'start':
                frame = frames['elem_start']
                name = f'drift_{ii}'
                s = survey['s', nn]
                length = 0
                slot_id = 0
            else:
                frame = frames['elem_end']
                name = nn
                s = survey['s', nn + '>>1']
                length = np.linalg.norm(
                    frames['elem_end'].XYZ - frames['elem_start'].XYZ, 2)
                slot_id = element_container[nn].extra.get('slot_id', 0)

            line = ' '

            # NAME
            value = f'"{name.upper()}"'.ljust(20)
            line += value

            # KEYWORD
            value = '"ELEMENT"'.ljust(20)
            line += value

            # S
            line += f'{s:26.9f}'

            # L
            line += f'{length:26.9f}'

            # ANGLE
            line += f'{0:26.9f}'

            # X, Y, Z, THETA, PHI, PSI
            for value in (
                    frame.X, frame.Y, frame.Z,
                    frame.theta, frame.phi, frame.psi):
                line += f'{value:26.9f}'

            # GLOBALTILT
            line += f'{frame.psi:26.9f}'

            # TILT
            line += f'{0:26.9f}'

            # SLOT_ID
            line += f'{int(slot_id):11d}'

            # ASSEMBLY_ID
            line += f'{0:11d}'

            # MECH_SEP
            line += f'{0:26.9f}'

            # V_POS
            line += f'{0:26.9f}'

            lines.append(line)

    # The %08s field descriptors above fix the widths: MAD-X writes the date
    # as dd/mm/yy and the time as HH.MM.SS, eight characters each.
    now = datetime.now()
    header = _LEGACY_SURVEY_TFS_HEADER.format(
        date=now.strftime('%d/%m/%y'), time=now.strftime('%H.%M.%S'))

    output = header + '\n'.join(lines)
    with open(file_name, 'w') as fid:
        fid.write(output)


def plot_exs(rotation_matrix, point, length=0.5, color='k'):
    """Plot the local x and s directions in the global Z-X plane."""
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt

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
