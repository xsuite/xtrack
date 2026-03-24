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


env = xt.Environment()
l = 1.0
angle = np.deg2rad(30.0)
l_straight = 1.0 / np.sin(angle / 2)
rho = 0.5 * l_straight / np.sin(angle / 2)
l_curv = rho * angle

drift = env.new("drift", xt.Drift, length=l)
bend_plus = env.new("bend_plus", xt.Bend, length=l_curv, angle=angle, k0=0)
bend_minus = env.new("bend_minus", xt.Bend, length=l_curv, angle=-angle, k0=0)

line = env.new_line(
    name="line",
    components=[drift, bend_plus, drift, drift, bend_minus, drift],
)
sv = line.survey()

s0, s1 = 0.0, 11.0
r0, r1 = 0.8, 2.0

profiles = [
    Profile(shape=Circle(radius=r0), tol_r=0, tol_x=0, tol_y=0),
    Profile(shape=Circle(radius=r1), tol_r=0, tol_x=0, tol_y=0),
]
profile_positions = [
    ProfilePosition(profile_index=0, s_position=s0),
    ProfilePosition(profile_index=1, s_position=s1),
]

model = ApertureModel(
    line=line,
    type_positions=[
        TypePosition(
            type_index=0,
            survey_reference_name=sv.name[0],
            survey_index=0,
            transformation=transform_matrix(dx=-1.5),
        ),
    ],
    types=[ApertureType(curvature=0.0, positions=profile_positions)],
    profiles=profiles,
    type_names=["type0"],
    profile_names=["circle0", "circle1"],
)

ap = Aperture(line=line, model=model)

# Build transform world<->type
sv_ref = sv.rows[0]
sv_ref_mat = matrix_from_survey_row(sv_ref)
type_matrix = model.type_positions[0].transformation.to_nparray()
world_from_type = sv_ref_mat @ type_matrix
type_from_world = np.linalg.inv(world_from_type)

# Compute interpolated cross-sections.
s_samples = np.linspace(1.0, 11.0, 21, dtype=np.float32)
sections_table = ap.cross_sections_at_s(s_samples)
sections = sections_table.cross_section
poses = sections_table.pose

# Assertions: in fixed type frame, all points should lie on a cone.
for ii in range(len(s_samples)):
    sec_xy = sections[ii]
    assert not np.isnan(sec_xy).any()

    sec_hom = np.column_stack(
        [sec_xy, np.zeros(len(sec_xy), dtype=np.float32), np.ones(len(sec_xy), dtype=np.float32)]
    )
    sec_world = (poses[ii] @ sec_hom.T).T
    sec_type = (type_from_world @ sec_world.T).T

    rr = np.linalg.norm(sec_type[:, :2], axis=1)
    z = sec_type[:, 2]
    expected_r = r0 + (r1 - r0) * (z - s0) / (s1 - s0)
    xo.assert_allclose(rr, expected_r, atol=1e-3, rtol=0)


ax = plt.figure().add_subplot(projection="3d")
ax.plot(sv.Z, sv.X, sv.Y, c="b", label="survey")

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
        profile_matrix = transform_matrix(
            dx=profile_pos.shift_x,
            dy=profile_pos.shift_y,
            ds=profile_pos.s_position,
            theta=profile_pos.rot_y,
            phi=profile_pos.rot_x,
            psi=profile_pos.rot_s,
        )
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

ax.set_xlabel("Z [m]")
ax.set_ylabel("X [m]")
ax.set_zlabel("Y [m]")
ax.set_title("Interpolated Aperture Cross-Sections: Cone in Curved Survey")
ax.legend(loc="upper left")
plt.show()
