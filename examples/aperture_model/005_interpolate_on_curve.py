import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt

from xtrack.aperture.aperture import Aperture, transform_matrix
from xtrack.aperture.structures import (
    ApertureModel,
    ApertureType,
    Circle,
    Profile,
    ProfilePosition,
    TypePosition,
)


def matrix_from_survey_row(sv_row):
    out = np.identity(4)
    out[:3, 0] = sv_row.ex
    out[:3, 1] = sv_row.ey
    out[:3, 2] = sv_row.ez
    out[:3, 3] = np.array([sv_row.X[0], sv_row.Y[0], sv_row.Z[0]])
    return out


def poly2d_to_hom(poly2d):
    n = poly2d.shape[0]
    return np.column_stack([poly2d, np.zeros(n), np.ones(n)]).T


def arc_matrix(length: float, angle: float, tilt: float) -> np.ndarray:
    if abs(angle) < 1e-8:
        mat = np.eye(4)
        mat[2, 3] = length
        return mat

    ct = np.cos(tilt)
    st = np.sin(tilt)

    ca = np.cos(angle)
    sa = np.sin(angle)

    dx = length * (ca - 1) / angle
    ds = length * sa / angle

    mat = np.array([
        [ct * ca, -st,      -ct * sa,  ct * dx],
        [st * ca,  ct,      -st * sa,  st * dx],
        [sa,       0.0,      ca,       ds],
        [0.0,      0.0,      0.0,      1.0],
    ], dtype=float)

    return mat


env = xt.Environment()
angle = np.deg2rad(35.0)
length = 3.2
radius = 0.6

bend = env.new('bend', xt.Bend, length=length, angle=angle, k0=0)
drift = env.new('drift', xt.Drift, length=length)
anti_bend = env.new('anti_bend', xt.Bend, length=length, angle=-angle, k0=0)
line = env.new_line(name='line', components=[bend, drift, anti_bend])
sv = line.survey()

shape = Circle(radius=radius)
profiles = [
    Profile(shape=shape, tol_r=0, tol_x=0, tol_y=0),
]
profile_positions = [
    ProfilePosition(profile_index=0, s_position=0.0),
    ProfilePosition(profile_index=0, s_position=length),
]

model = ApertureModel(
    line=line,
    type_positions=[
        TypePosition(
            type_index=0,
            survey_reference_name=sv.name[0],
            survey_index=0,
            transformation=transform_matrix(),
        ),
        TypePosition(
            type_index=1,
            survey_reference_name=sv.name[1],
            survey_index=1,
            transformation=transform_matrix(),
        ),
        TypePosition(
            type_index=2,
            survey_reference_name=sv.name[2],
            survey_index=2,
            transformation=transform_matrix(),
        ),
    ],
    types=[
        ApertureType(curvature=angle / length, positions=profile_positions),
        ApertureType(curvature=0, positions=profile_positions),
        ApertureType(curvature=-angle / length, positions=profile_positions),
    ],
    profiles=profiles,
    type_names=['type0', 'type1', 'type2'],
    profile_names=['circ0'],
)

ap = Aperture(line=line, model=model, num_profile_points=256)

bounds_table = ap.get_bounds_table()
bounds_s = [0, length, length, 2 * length, 2 * length, 3 * length]
xo.assert_allclose(bounds_table.s, bounds_s, atol=1e-6, rtol=1e-6)
xo.assert_allclose(bounds_table.s_start, bounds_s, atol=1e-6, rtol=1e-6)
xo.assert_allclose(bounds_table.s_end, bounds_s, atol=1e-6, rtol=1e-6)
assert all(bounds_table.type_name == ['type0', 'type0', 'type1', 'type1', 'type2', 'type2'])
assert all(bounds_table.profile_name == ['circ0'])

s_samples = np.linspace(0, 3 * length, 51, dtype=np.float32)
sections, poses = ap.cross_sections_at_s(s_samples)

ax = plt.figure().add_subplot(projection="3d")

line_fine = line.copy()
line_fine.cut_at_s(s_samples)
sv_fine = line_fine.survey()
ax.plot(sv_fine.Z, sv_fine.X, sv_fine.Y, c="b", label="survey")

# Plot installed profiles (red)
for type_pos in ap.model.type_positions:
    aper_type = ap.model.type_for_position(type_pos)
    sv_ref_row = sv.rows[type_pos.survey_index]
    sv_ref_matrix = matrix_from_survey_row(sv_ref_row)
    type_pos_matrix = type_pos.transformation.to_nparray()

    for profile_pos in aper_type.positions:
        profile = ap.model.profile_for_position(profile_pos)
        poly = ap.polygon_for_profile(profile, 256)
        poly_hom = poly2d_to_hom(poly)
        profile_matrix_trans = transform_matrix(
            dx=profile_pos.shift_x,
            dy=profile_pos.shift_y,
            ds=0,
            theta=profile_pos.rot_y,
            phi=profile_pos.rot_x,
            psi=profile_pos.rot_s,
        )
        profile_matrix_arc = arc_matrix(
            length=profile_pos.s_position,
            angle=aper_type.curvature * profile_pos.s_position,
            tilt=0,
        )
        profile_matrix = profile_matrix_arc @ profile_matrix_trans
        poly_world = sv_ref_matrix @ type_pos_matrix @ profile_matrix @ poly_hom
        xw, yw, zw = poly_world[:3]
        ax.plot(zw, xw, yw, c="r", lw=2)

# Plot interpolated cross-sections (green)
for ii in range(len(s_samples)):
    sec_xy = sections[ii]
    sec_hom = poly2d_to_hom(sec_xy)
    sec_world = poses[ii] @ sec_hom
    xw, yw, zw = sec_world[:3]
    ax.plot(zw, xw, yw, c="g", alpha=0.65)

    pose_centers = poses[ii] @ np.array([0, 0, 0, 1]).T
    ax.scatter3D(pose_centers[2], pose_centers[0], pose_centers[1], c='g', alpha=0.65)

ax.set_xlabel("Z [m]")
ax.set_ylabel("X [m]")
ax.set_zlabel("Y [m]")
ax.auto_scale_xyz([0, 4], [-2, 2], [-2, 2])
ax.set_box_aspect((1, 1, 1))
ax.legend(loc="upper left")
plt.show()


for ii in range(1, len(sections)):
    xo.assert_allclose(np.linalg.norm(sections[ii], axis=1), radius, atol=1e-6, rtol=0)
