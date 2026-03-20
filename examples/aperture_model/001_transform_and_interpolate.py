import xtrack as xt
import xobjects as xo
import numpy as np
import matplotlib.pyplot as plt
from xtrack.aperture.aperture import Aperture, transform_matrix
from xtrack.aperture.structures import ApertureModel, ApertureType, Circle, Profile, ProfilePosition, Rectangle, TypePosition


env = xt.Environment()

l = 1
dx = 1
angle = np.deg2rad(30)
l_straight = dx / np.sin(angle / 2)
rho = 0.5 * l_straight / np.sin(angle / 2)
l_curv = rho * angle

drift = env.new('drift', xt.Drift, length=l)
rot_plus = env.new('rot_plus', xt.Bend, length=l_curv, angle=angle, k0=0)
rot_minus = env.new('rot_minus', xt.Bend, length=l_curv, angle=-angle, k0=0)

line = env.new_line(
    name='line',
    components=[drift, rot_plus, drift, drift, rot_minus, drift],
)

sv = line.survey()

circle = Circle(radius=2)
rectangle = Rectangle(half_width=2, half_height=1)

profiles = [
    Profile(shape=circle, tol_r=0, tol_x=0, tol_y=0),
    Profile(shape=rectangle, tol_r=0, tol_x=0, tol_y=0),
]

types = [
    ApertureType(curvature=0., positions=[
        ProfilePosition(profile_index=1, s_position=0, rot_s=np.deg2rad(15)),
        ProfilePosition(profile_index=1, s_position=5.5, rot_s=np.deg2rad(90)),
        ProfilePosition(profile_index=0, s_position=11, rot_x=np.deg2rad(10)),
    ]),
]

type_positions = [
    TypePosition(
        type_index=0,
        survey_reference_name='drift::0',
        survey_index=sv.name.tolist().index('drift::0'),
        transformation=transform_matrix(dx=-1.5),
    ),
]

model = ApertureModel(
    line_name='line',
    type_positions=type_positions,
    types=types,
    profiles=profiles,
    type_names=['type0'],
    profile_names=['circle', 'rectangle'],
)

line_sliced = line.copy()
line_sliced.cut_at_s(np.linspace(0, line_sliced.get_length(), 100))
sv_sliced = line_sliced.survey()
ax = plt.figure().add_subplot(projection='3d')
ax.plot(sv_sliced.Z, sv_sliced.X, sv_sliced.Y, c='b')
ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]')
ax.set_zlabel('Y [m]')

ax.auto_scale_xyz([0, 12], [-6, 6], [-6, 6])

aper = Aperture(line, model)


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

expected_poses = poses_at_s(line, s_for_cuts)
xo.assert_allclose(poses, expected_poses, atol=1e-6, rtol=1e-6)

for idx, s in enumerate(s_for_cuts):
    profile = profiles[idx]
    profile_hom = poly2d_to_hom(profile)
    profile_in_sv_frame = poses[idx] @ profile_hom

    xs, ys, zs = profile_in_sv_frame[:3]
    ax.plot(zs, xs, ys, c='g')

plt.show()
