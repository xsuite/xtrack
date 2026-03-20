import xtrack as xt
import xobjects as xo
import numpy as np
import matplotlib.pyplot as plt
from xtrack.aperture.aperture import Aperture, transform_matrix
from xtrack.aperture.structures import ApertureModel, ApertureType, Circle, Profile, ProfilePosition, Rectangle, TypePosition


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
ax.plot(sv.Z, sv.X, sv.Y, c='b')
ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]')
ax.set_zlabel('Y [m]')

ax.auto_scale_xyz([0, 12], [-6, 6], [-6, 6])


def matrix_from_survey_point(sv_row):
    matrix = np.identity(4)
    matrix[:3, 0] = sv_row.ex
    matrix[:3, 1] = sv_row.ey
    matrix[:3, 2] = sv_row.ez
    matrix[:3, 3] = np.hstack([sv_row.X, sv_row.Y, sv_row.Z])
    return matrix


def poly2d_to_hom(poly2d):
    num_points = poly2d.shape[0]
    poly_hom = np.column_stack((poly2d, np.zeros(num_points), np.ones(num_points))).T
    return poly_hom


for type_pos in aper.model.type_positions:
    aper_type = aper.model.type_for_position(type_pos)
    sv_ref = sv.rows[type_pos.survey_index]

    sv_ref_matrix = matrix_from_survey_point(sv_ref)
    type_matrix = type_pos.transformation.to_nparray()

    for profile_pos in aper_type.positions:
        profile = aper.model.profile_for_position(profile_pos)

        num_points = 100
        poly = aper.polygon_for_profile(profile, num_points)
        poly_hom = poly2d_to_hom(poly)

        profile_position_matrix = transform_matrix(
            dx=profile_pos.shift_x,
            dy=profile_pos.shift_y,
            ds=profile_pos.s_position,
            theta=profile_pos.rot_y,
            phi=profile_pos.rot_x,
            psi=profile_pos.rot_s,
        )

        poly_in_sv_frame = sv_ref_matrix @ type_matrix @ profile_position_matrix @ poly_hom

        xs, ys, zs = poly_in_sv_frame[:3]
        ax.plot(zs, xs, ys, c='r')


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


s_for_cuts = np.linspace(1, 11, 20)
profiles, poses = aper.cross_sections_at_s(s_for_cuts)

expected_poses = poses_at_s(ring, s_for_cuts)
xo.assert_allclose(poses, expected_poses, atol=1e-6, rtol=1e-6)

for idx, s in enumerate(s_for_cuts):
    profile = profiles[idx]
    profile_hom = poly2d_to_hom(profile)
    profile_in_sv_frame = poses[idx] @ profile_hom

    xs, ys, zs = profile_in_sv_frame[:3]
    ax.plot(zs, xs, ys, c='g')

plt.show()
