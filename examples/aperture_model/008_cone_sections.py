import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from xtrack.aperture.transform import arc_matrix, transform_matrix

# A right circular cone with apex at the origin and its height along the z
# axis, which coincides with the survey of a straight drift line. The pipe
# profiles are conic sections of that cone, cut by planes oblique to its
# height. Each cut is an ellipse; the ellipse parameters and the pose that
# mounts the cut back onto the cone follow from the plane-cone geometry, so
# all installed profiles are exact sections of one and the same cone.
#
# The cut planes are parallel, so the installed ellipses are similar and
# pairwise disjoint. The cone is a ruled surface spanned by consecutive
# profiles, hence an interpolated cross-section of the pipe is an exact
# section of the cone too. At the queried s the cone is cut orthogonally to
# its height, so the interpolated cross-section must be a circle of radius
# s * tan(alpha); this is asserted below. The profiles are spaced closely
# enough for the queried plane to cut five of them at once.

alpha = np.deg2rad(15.0)  # cone half-opening angle
tan_alpha = np.tan(alpha)

# All cut planes are parallel: z + x tan(gamma) = h, with varying h.
gamma = np.deg2rad(20.0)
num_cuts = 12
h = 3.2 + 0.2 * np.arange(num_cuts)

drift_length = 7.5
s_query = 4.4
num_profile_points = 128


def cone_cut_parameters(h0, gamma0):
    """Ellipse parameters and mounting pose of a cone cut at height h0.

    The cut plane z + x tan(gamma) = h0 intersects the cone x^2 + y^2 =
    z^2 tan^2(alpha) in the ellipse

        D u^2 + 2 B u + v^2 = h0^2 tan^2(alpha),
        D = cos^2(gamma) - tan^2(alpha) sin^2(gamma),
        B = h0 tan^2(alpha) sin(gamma),

    with in-plane coordinates u along the plane's steepest direction (world
    x = u cos(gamma), world z = h0 - u sin(gamma)) and v along y. Completing
    the square gives the semi-axes a (along u) and b (along v) and the
    centre offset u_c = -B / D; the mounting pose follows from the
    in-plane-to-world map.
    """
    t2 = tan_alpha**2
    d = np.cos(gamma0)**2 - t2 * np.sin(gamma0)**2
    b_coef = h0 * t2 * np.sin(gamma0)
    u_c = -b_coef / d
    radius = np.sqrt(h0**2 * t2 + b_coef**2 / d)
    return {
        'a': radius / np.sqrt(d),  # semi-axis along u
        'b': radius,               # semi-axis along y
        'shift_x': u_c * np.cos(gamma0),
        'shift_s': h0 - u_c * np.sin(gamma0),
        'rot_y': gamma0,
    }


cuts = [cone_cut_parameters(h0, gamma) for h0 in h]

# Build the line and mount one Ellipse profile per cone cut.

env = xt.Environment()
drift = env.new('drift', xt.Drift, length=drift_length)
line = env.new_line(name='line', components=[drift])

for ii, cut in enumerate(cuts):
    env.aperture.new_profile(f'cut{ii}', 'Ellipse', a=cut['a'], b=cut['b'])

pipe = env.aperture.new_pipe('cone', curvature=0.0, positions=[
    env.aperture.profiles[f'cut{ii}'].place(
        at=cut['shift_s'], shift_x=cut['shift_x'], rot_y_rad=cut['rot_y'])
    for ii, cut in enumerate(cuts)
])
ap = line.aperture
ap.num_profile_points = num_profile_points
ap.place_pipe('cone_at', pipe, at='drift@start')
ap.build_model()


# Plotting and mounting helpers.

def poly2d_to_hom(poly2d):
    n = poly2d.shape[0]
    return np.column_stack([poly2d, np.zeros(n), np.ones(n)]).T


def point_polyline_distance(points, polygon):
    """Distance from each point to the closed polyline through `polygon`."""
    closed = np.vstack([polygon, polygon[:1]])
    start, end = closed[:-1], closed[1:]
    segment = end - start
    norm_sq = np.sum(segment**2, axis=1)
    norm_sq[norm_sq == 0] = 1
    t = np.clip(np.sum(
        (points[:, None, :] - start[None, :, :]) * segment[None, :, :], axis=2)
        / norm_sq[None, :], 0, 1)
    projections = start[None, :, :] + t[..., None] * segment[None, :, :]
    return np.linalg.norm(points[:, None, :] - projections, axis=2).min(axis=1)


def matrix_from_survey_row(sv_row):
    out = np.identity(4)
    out[:3, 0] = sv_row.ex
    out[:3, 1] = sv_row.ey
    out[:3, 2] = sv_row.es
    out[:3, 3] = np.array([sv_row.X[0], sv_row.Y[0], sv_row.Z[0]])
    return out


def installed_polygons_world():
    """World-space polygons of all installed profiles."""
    sv_ref = survey.rows[0]
    sv_ref_matrix = matrix_from_survey_row(sv_ref)
    pipe_pos_matrix = ap.pipe_positions[0].transformation
    out = []
    for cut, pipe_pos in zip(cuts, ap.pipe_positions[0].pipe):
        poly = pipe_pos.profile.build_polygon(len_points=num_profile_points)
        profile_local = transform_matrix(
            shift_x=cut['shift_x'], rot_y_rad=cut['rot_y'])
        profile_arc = arc_matrix(length=cut['shift_s'], angle=0.0, tilt=0.0)
        poly_world = (sv_ref_matrix @ pipe_pos_matrix @ profile_arc
                      @ profile_local @ poly2d_to_hom(poly))
        out.append(poly_world)
    return out


# Sanity checks of the mounting: every installed polygon must lie exactly on
# the cone, and the queried plane must cut several installed profiles.

survey = line.survey()
polygons_world = installed_polygons_world()

for poly_world in polygons_world:
    xw, yw, zw = poly_world[:3]
    cone_equation = xw**2 + yw**2 - zw**2 * tan_alpha**2
    xo.assert_allclose(cone_equation, 0.0, atol=1e-9)

bounds = ap.get_bounds_table()
num_cut = int(np.sum((bounds.s_start <= s_query) & (s_query <= bounds.s_end)))
print(f'Queried s = {s_query}, cutting {num_cut} of the '
      f'{num_cuts} installed profiles.')
assert num_cut >= 4

# Request interpolated cross-sections. An exact cone section at 90 degrees
# to the height is a circle of radius s * tan(alpha); this must hold both
# for a plane between profiles and for a plane exactly at an installed
# profile's own s. At an installed profile the queried plane is tangent to
# that profile's ellipse only at two points, so the section there is made
# almost entirely of the wall rulings towards the neighbouring profiles:
# an implementation shortcutting the interpolation at s == s_center would
# instead return the profile's orthogonal projection - an ellipse with
# semi-axes a * cos(gamma) and b, up to ~0.5% smaller than the circle.

s_queries = [s_query, cuts[6]['shift_s']]

results = {}
for plane_s in s_queries:
    cross_sections = ap.get_cross_sections_at_s([plane_s])
    section = np.asarray(cross_sections.cross_section[0])[:-1]
    pose = np.asarray(cross_sections.pose[0])

    expected_radius = plane_s * tan_alpha
    radii = np.linalg.norm(section, axis=1)
    print(f'Interpolated cross-section at s = {plane_s:.4f}: radii '
          f'[{radii.min():.6f}, {radii.max():.6f}], expected circle '
          f'radius {expected_radius:.6f}')
    xo.assert_allclose(radii, expected_radius, atol=1e-3, rtol=0)

    # Every profile cut by the plane must contribute: the endpoints of its
    # exact intersection with the plane must lie on the polygon.

    plane_normal = pose[:3, 2]
    plane_origin = pose[:3, 3]

    exact_points = []
    for poly_world in polygons_world:
        pts = poly_world[:3].T
        off = (pts - plane_origin) @ plane_normal
        for jj in range(len(pts)):
            a_off, b_off = off[jj], off[(jj + 1) % len(pts)]
            if a_off <= 0 <= b_off or b_off <= 0 <= a_off:
                f = a_off / (a_off - b_off)
                hit = pts[jj] + f * (pts[(jj + 1) % len(pts)] - pts[jj])
                exact_points.append((np.linalg.inv(pose)
                                     @ np.append(hit, 1.0))[:2])
    exact_points = np.array(exact_points)
    print(f'  {len(exact_points)} exact profile-plane intersection points; '
          f'all must lie on the reconstructed polygon.')
    xo.assert_allclose(
        point_polyline_distance(exact_points, section), 0.0, atol=1e-3,
        rtol=0)

    results[plane_s] = {
        'section': section, 'pose': pose,
        'expected_radius': expected_radius, 'exact_points': exact_points,
    }


# Plots for the first queried plane.

s_query = s_queries[0]
result = results[s_query]
section = result['section']
pose = result['pose']
expected_radius = result['expected_radius']
exact_points = result['exact_points']


# 3D view: survey (blue), cone guide circles (grey), installed profiles
# (red), queried plane (translucent), interpolated section (green), and the
# exact profile-plane intersections (orange).

ax = plt.figure(figsize=(11, 8)).add_subplot(projection='3d')

line_fine = line.copy()
line_fine.cut_at_s(np.linspace(0, drift_length, 200))
sv_fine = line_fine.survey()
ax.plot(sv_fine.Z, sv_fine.X, sv_fine.Y, c='b', lw=1, label='survey')

t_fine = np.linspace(0, 2 * np.pi, 400, endpoint=False)
for zc in (3.2, 3.8, s_query, 4.8):
    rc = zc * tan_alpha
    outline = np.array(
        [rc * np.cos(t_fine), rc * np.sin(t_fine), np.full_like(t_fine, zc)])
    ax.plot(outline[2], outline[0], outline[1], c='0.85', lw=0.6, ls=':')

for poly_world in polygons_world:
    xs, ys, zs = poly_world[:3]
    ax.plot(zs, xs, ys, c='r', lw=1.2)

plane_size = 1.25 * expected_radius
grid = np.linspace(-plane_size, plane_size, 2)
plane_uv = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
plane_world = pose @ np.column_stack(
    [plane_uv, np.zeros(len(plane_uv)), np.ones(len(plane_uv))]).T
plane_quad = Poly3DCollection(
    [plane_world[:3, [0, 1, 3, 2]].T[:, [2, 0, 1]]],
    facecolor='0.75', alpha=0.35, edgecolor='0.4', linewidth=1)
ax.add_collection3d(plane_quad)
ax.plot([np.nan], [np.nan], [np.nan], c='0.75', alpha=0.35,
        label=f'queried plane (s = {s_query:.2f})')

section_world = pose @ poly2d_to_hom(section)
xs, ys, zs = section_world[:3]
ax.plot(zs, xs, ys, c='g', lw=2.5, label='interpolated section')

points_world = pose @ poly2d_to_hom(exact_points)
ax.scatter(points_world[2], points_world[0], points_world[1],
           c='orange', s=25, label='installed profile \u2229 plane',
           depthshade=False)

ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]')
ax.set_zlabel('Y [m]')
ax.set_title(f'Pipe of oblique cone cuts: interpolation at s = {s_query} '
             f'is the circle r = {expected_radius:.4f}')
ax.legend(loc='upper left')
ax.set_box_aspect((2.2, 1, 1))

# The interpolated section in the plane frame: a circle of the expected
# radius, passing through every cut profile's exact intersection points.

fig, ax_plane = plt.subplots(figsize=(7, 7))
section_closed = np.vstack([section, section[:1]])
ax_plane.plot(section_closed[:, 0], section_closed[:, 1], c='g', lw=2,
              label='interpolated section')
ax_plane.scatter(section[:, 0], section[:, 1], c='g', s=8)
ax_plane.scatter(exact_points[:, 0], exact_points[:, 1], c='orange', s=45,
                 label='installed profile \u2229 plane', zorder=3)
circle_ref = plt.Circle((0, 0), expected_radius, fill=False, edgecolor='k',
                        ls='--', lw=0.8, label=f'circle r = {expected_radius:.4f}')
ax_plane.add_patch(circle_ref)
ax_plane.set_xlabel('x [m]')
ax_plane.set_ylabel('y [m]')
ax_plane.set_aspect('equal')
ax_plane.set_title(f'Interpolated cross-section at s = {s_query}')
ax_plane.legend(loc='upper right')

plt.show()
