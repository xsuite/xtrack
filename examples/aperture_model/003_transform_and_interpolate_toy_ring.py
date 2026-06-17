import xtrack as xt
import xobjects as xo
import numpy as np
import matplotlib.pyplot as plt
from xtrack.aperture.aperture import Aperture
from xtrack.aperture.transform import arc_matrix, poly2d_to_homogeneous, transform_matrix


TOY_RING_SEQUENCE = """
    ! Toy Ring, 4 arcs

    l_arc = 3;  ! length of the arc
    l_quad = 0.3;  ! length of the quads
    l_drift = 1;  ! length of the straight section drifts

    qf = 0.1;  ! qf strength
    qd = -0.7;  ! qd strength
    angle_arc = pi / 2;  ! arcs 90°

    mb: sbend, angle = angle_arc, l = l_arc, apertype=circle, aperture={0.1}, aper_offset={0.003, 0};
    mqf: quadrupole, k1 = qf, l = l_quad, apertype=rectangle, aperture={0.08, 0.04};
    mqd: quadrupole, k1 = qd, l = l_quad, apertype=ellipse, aperture={0.04, 0.08};
    ds: drift, l = l_drift;
    ap_ds: marker, apertype=rectellipse, aperture={0.022, 0.01715, 0.022, 0.022}, aper_tol={9e-4, 8e-4, 5e-4};
    dsa: line = (ap_ds, ds, ap_ds);

    ss_f: line = (dsa, mqf, dsa);
    ss_d: line = (dsa, mqd, dsa);

    ring: line = (ss_f, mb, ss_d, mb, ss_f, mb, ss_d, mb);

    beam, particle=proton, pc=1.2e9;
    use, period=ring;
"""

env = xt.load(string=TOY_RING_SEQUENCE, format='madx', install_limits=False)
env.set_particle_ref('proton', p0c=1.2e9)
ring = env['ring']

aper = Aperture.from_line_with_associated_apertures(ring)

sv = ring.survey()

ax = plt.figure().add_subplot(projection='3d')
ax.plot(sv.Z, sv.X, sv.Y, c='b', label='survey')
ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]')
ax.set_zlabel('Y [m]')

ax.auto_scale_xyz([-3, 5], [-7, 1], [-4, 4])

sv_rows = list(sv.rows)
for ii, row in enumerate(sv_rows):
    if row.name == '_end_point':
        continue
    z, x, y = row.Z, row.X, row.Y
    if ii + 1 < len(sv_rows):
        next_row = sv_rows[ii + 1]
        z = 0.5 * (row.Z + next_row.Z)
        x = 0.5 * (row.X + next_row.X)
        y = 0.5 * (row.Y + next_row.Y)
    ax.text(z, x, y + 0.3, row.name, fontsize=9, color='k')


def matrix_from_survey_point(sv_row):
    matrix = np.identity(4)
    matrix[:3, 0] = sv_row.ex
    matrix[:3, 1] = sv_row.ey
    matrix[:3, 2] = sv_row.ez
    matrix[:3, 3] = np.hstack([sv_row.X, sv_row.Y, sv_row.Z])
    return matrix


seen_installed_profiles = False
for pipe_pos in aper._model.pipe_positions:
    aper_pipe = aper._model.pipe_for_position(pipe_pos)
    sv_ref = sv.rows[pipe_pos.survey_index]

    sv_ref_matrix = matrix_from_survey_point(sv_ref)
    pipe_matrix = pipe_pos.transformation.to_nparray()

    for profile_pos in aper_pipe.positions:
        profile = aper._model.profile_for_position(profile_pos)

        num_points = 100
        poly = aper.polygon_for_profile(profile, num_points)
        poly_hom = poly2d_to_homogeneous(poly)

        profile_matrix_trans = transform_matrix(
            shift_x=profile_pos.shift_x,
            shift_y=profile_pos.shift_y,
            shift_z=0,
            rot_y_rad=profile_pos.rot_y_rad,
            rot_x_rad=profile_pos.rot_x_rad,
            rot_z_rad=profile_pos.rot_s_rad,
        )
        profile_matrix_arc = arc_matrix(
            length=profile_pos.shift_s,
            angle=aper_pipe.curvature * profile_pos.shift_s,
            tilt=0,
        )
        profile_position_matrix = profile_matrix_arc @ profile_matrix_trans

        poly_in_sv_frame = sv_ref_matrix @ pipe_matrix @ profile_position_matrix @ poly_hom

        xs, ys, zs = poly_in_sv_frame[:3]
        label = 'installed profiles' if not seen_installed_profiles else None
        ax.plot(zs, xs, ys, c='r', label=label)
        seen_installed_profiles = True


def poses_at_s(line, s_positions):
    """Return a local coordinate system (each represented by a homogeneous matrix) at all ``s_positions``."""
    poses = np.zeros(shape=(len(s_positions), 4, 4), dtype=np.float32)
    line_sliced = line.copy()
    line_sliced.cut_at_s(s_positions)
    survey_sliced = line_sliced.survey()
    sv_indices = np.searchsorted(survey_sliced.s, s_positions)

    for idx, sv_idx in enumerate(sv_indices):
        row = survey_sliced.rows[sv_idx]
        poses[idx, :3, 0] = row.ex
        poses[idx, :3, 1] = row.ey
        poses[idx, :3, 2] = row.ez
        poses[idx, :, 3] = np.hstack([row.X, row.Y, row.Z, 1])

    return poses


s_for_cuts = np.linspace(0, ring.get_length(), 300, endpoint=False)
profiles_table = aper.cross_sections_at_s(s_for_cuts)
profiles = profiles_table.cross_section
poses = profiles_table.pose

expected_poses = poses_at_s(ring, s_for_cuts)
xo.assert_allclose(poses, expected_poses, atol=1e-6, rtol=1e-6)

seen_cross_sections = False
for idx, s in enumerate(s_for_cuts):
    profile = profiles[idx]
    profile_hom = poly2d_to_homogeneous(profile)
    profile_in_sv_frame = poses[idx] @ profile_hom

    xs, ys, zs = profile_in_sv_frame[:3]
    label = 'interpolated cross sections' if not seen_cross_sections else None
    ax.plot(zs, xs, ys, c='g', label=label)
    seen_cross_sections = True

ax.legend()
plt.show()
