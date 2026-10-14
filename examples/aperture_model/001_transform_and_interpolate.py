import xtrack as xt
import xobjects as xo
import numpy as np
import matplotlib.pyplot as plt
from xtrack.aperture.transform import transform_matrix


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

circle = env.aperture.new_profile('circle', 'Circle', radius=2)
rectangle = env.aperture.new_profile('rectangle', 'Rectangle', max_x=2, max_y=1)
pipe = env.aperture.new_pipe(
    'pipe0', positions=[
        rectangle.place(at=0, rot_s_rad=np.deg2rad(15)),
        rectangle.place(at=5.5, rot_s_rad=np.deg2rad(90)),
        circle.place(at=11, rot_x_rad=np.deg2rad(10)),
    ],
)
aper = line.aperture
aper.place_pipe('pipe0', pipe, at='drift::0', shift_x=-1.5)
aper.build_model()

line_sliced = line.copy()
line_sliced.cut_at_s(np.linspace(0, line_sliced.get_length(), 100))
sv_sliced = line_sliced.survey()
ax = plt.figure().add_subplot(projection='3d')
ax.plot(sv_sliced.Z, sv_sliced.X, sv_sliced.Y, c='b')
ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]')
ax.set_zlabel('Y [m]')

ax.auto_scale_xyz([0, 12], [-6, 6], [-6, 6])

def matrix_from_survey_point(sv_row):
    matrix = np.identity(4)
    matrix[:3, 0] = sv_row.ex
    matrix[:3, 1] = sv_row.ey
    matrix[:3, 2] = sv_row.es
    matrix[:3, 3] = np.hstack([sv_row.X, sv_row.Y, sv_row.Z])
    return matrix


def poly2d_to_hom(poly2d):
    num_points = poly2d.shape[0]
    poly_hom = np.column_stack((poly2d, np.zeros(num_points), np.ones(num_points))).T
    return poly_hom


for pipe_pos in aper.pipe_positions:
    aper_pipe = pipe_pos.pipe
    sv_ref = sv.rows[pipe_pos.survey_index]

    sv_ref_matrix = matrix_from_survey_point(sv_ref)
    pipe_matrix = pipe_pos.transformation

    for profile_pos in aper_pipe:
        profile = profile_pos.profile

        num_points = 100
        poly = profile.build_polygon(len_points=num_points)
        poly_hom = poly2d_to_hom(poly)

        profile_position_matrix = transform_matrix(
            shift_x=profile_pos.shift_x,
            shift_y=profile_pos.shift_y,
            shift_z=profile_pos.shift_s,
            rot_y_rad=profile_pos.rot_y_rad,
            rot_x_rad=profile_pos.rot_x_rad,
            rot_z_rad=profile_pos.rot_s_rad,
        )

        poly_in_sv_frame = sv_ref_matrix @ pipe_matrix @ profile_position_matrix @ poly_hom

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
        poses[idx, :3, 2] = row.es
        poses[idx, :, 3] = np.hstack([row.X, row.Y, row.Z, 1])

    return poses


s_for_cuts = np.linspace(1, 11, 20)
cs_table = aper.get_cross_sections_at_s(s_for_cuts)
profiles, poses = cs_table.cross_section, cs_table.pose

expected_poses = poses_at_s(line, s_for_cuts)
xo.assert_allclose(poses, expected_poses, atol=1e-6, rtol=1e-6)

for idx, s in enumerate(s_for_cuts):
    profile = profiles[idx]
    profile_hom = poly2d_to_hom(profile)
    profile_in_sv_frame = poses[idx] @ profile_hom

    xs, ys, zs = profile_in_sv_frame[:3]
    ax.plot(zs, xs, ys, c='g')

plt.show()
