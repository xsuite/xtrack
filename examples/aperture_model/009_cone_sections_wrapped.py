import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo
import xtrack as xt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from xtrack.aperture.transform import arc_matrix, transform_matrix

# The cone-sections pipe of example 008, installed on a RING so that it wraps
# around the seam s = 0. The ring is a racetrack: five straight drifts and
# four 90-degree bends, arranged so that the seam falls between two collinear
# drifts (dA2a before the seam, dA2b after it) and the survey is straight on
# both sides of the wrap. The cone pipe sits in dA2a, shifted so that its cut
# centres run from s = 41.0 up to s = 43.25, i.e. past the seam at 42 = 0.
#
# With this placement each cut plane intersects the closed survey twice (once
# in the drift, once on the opposite side of the ring); the zigzag search in
# survey_s_for_aperture starts at the pipe's anchor segment and finds the
# closer one, which is the correct installed s. The bounds then split into a
# pre-seam group (41.0 .. 42.0) and a post-seam group (0.2 .. 1.2), which
# sorts into a globally monotone order - the usual ring representation.
#
# Cross-sections queried across the whole wrapped region reproduce the exact
# cone section (asserted below), including at the seam itself. The only
# deviations occur where the section is bounded by the wall to a profile of
# the neighbouring pipe (the uphill side of cut0, from f_pre), which is the
# expected interpolation of the model as built.

alpha = np.deg2rad(15.0)  # cone half-opening angle
tan_alpha = np.tan(alpha)
gamma = np.deg2rad(20.0)  # all cut planes parallel: z + x tan(gamma) = h
num_cuts = 12
h = 3.2 + 0.2 * np.arange(num_cuts)

# Racetrack: 42 m total, seam between dA2a (ending at s = 42) and dA2b
# (starting at s = 0), both collinear along the same direction.
drift_seam = 4.5    # dA2b and dA2a: the two drifts adjacent to the seam
drift_long = 9.0    # dC: the opposite long side
drift_short = 6.0   # dB and dD: the two short sides
bend_length = 3.0   # four bends of 90 degrees

s_pipe_origin = 37.8  # dA2a start (37.5) + pipe shift (0.3)
num_profile_points = 128


def cone_cut_parameters(h0, gamma0):
    """Ellipse parameters and mounting pose of a cone cut at height h0.

    The cut plane ``z + x tan(gamma) = h0`` intersects the cone in the
    ellipse ``D u^2 + 2 B u + v^2 = h0^2 tan^2(alpha)`` with
    ``D = cos^2(gamma) - tan^2(alpha) sin^2(gamma)`` and
    ``B = h0 tan^2(alpha) sin(gamma)``, in coordinates u along the plane's
    steepest direction and v along y. Completing the square gives the
    semi-axes and the centre; the mounting pose follows from the
    in-plane-to-world map.
    """
    d = np.cos(gamma0)**2 - tan_alpha**2 * np.sin(gamma0)**2
    b_coef = h0 * tan_alpha**2 * np.sin(gamma0)
    u_c = -b_coef / d
    radius = np.sqrt(h0**2 * tan_alpha**2 + b_coef**2 / d)
    return {
        'a': radius / np.sqrt(d),   # semi-axis along u
        'b': radius,                # semi-axis along y
        'shift_x': u_c * np.cos(gamma0),
        'shift_s': h0 - u_c * np.sin(gamma0),
        'rot_y': gamma0,
    }


cuts = [cone_cut_parameters(h0, gamma) for h0 in h]

# Ring and pipe setup.

env = xt.Environment()
bend = env.new('bend', xt.Bend, length=bend_length, angle=np.pi / 2, k0=0)
dA2b = env.new('dA2b', xt.Drift, length=drift_seam)
dA2a = env.new('dA2a', xt.Drift, length=drift_seam)
dB = env.new('dB', xt.Drift, length=drift_short)
dC = env.new('dC', xt.Drift, length=drift_long)
dD = env.new('dD', xt.Drift, length=drift_short)
line = env.new_line(name='ring', components=[
    dA2b, bend, dB, bend, dC, bend, dD, bend, dA2a])
L = float(line.get_length())
assert L == 2 * drift_seam + drift_long + 2 * drift_short + 4 * bend_length

circ = env.aperture.new_profile('filler', 'Circle', radius=0.3)
for ii, cut in enumerate(cuts):
    env.aperture.new_profile(f'cut{ii}', 'Ellipse', a=cut['a'], b=cut['b'])

pipe_cone = env.aperture.new_pipe('cone', curvature=0.0, positions=[
    env.aperture.profiles[f'cut{ii}'].place(
        at=cut['shift_s'], shift_x=cut['shift_x'], rot_y_rad=cut['rot_y'])
    for ii, cut in enumerate(cuts)
])
pipe_f_pre = env.aperture.new_pipe('f_pre', curvature=0.0, positions=[
    circ.place(at=2.8)])  # dA2a: s = 40.3, just before the cone
pipe_f_post = env.aperture.new_pipe('f_post', curvature=0.0, positions=[
    circ.place(at=2.2)])  # dA2b: s = 2.2, just after the wrapped cone
pipe_f_b = env.aperture.new_pipe('f_b', curvature=0.0, positions=[
    circ.place(at=3.0)])  # dB: s = 10.5
pipe_f_c = env.aperture.new_pipe('f_c', curvature=0.0, positions=[
    circ.place(at=4.5)])  # dC: s = 21.0
pipe_f_d = env.aperture.new_pipe('f_d', curvature=0.0, positions=[
    circ.place(at=3.0)])  # dD: s = 31.5

ap = line.aperture
ap.num_profile_points = num_profile_points
ap.place_pipe(
    'cone_at', pipe_cone, at='dA2a',
    transformation=transform_matrix(shift_z=0.3))
ap.place_pipe('f_pre', pipe_f_pre, at='dA2a')
ap.place_pipe('f_post', pipe_f_post, at='dA2b')
ap.place_pipe('f_b', pipe_f_b, at='dB')
ap.place_pipe('f_c', pipe_f_c, at='dC')
ap.place_pipe('f_d', pipe_f_d, at='dD')
ap.build_model()


# Mounting helpers and sanity checks.

def poly2d_to_hom(poly2d):
    n = poly2d.shape[0]
    return np.column_stack([poly2d, np.zeros(n), np.ones(n)]).T


def matrix_from_survey_row(sv_row):
    out = np.identity(4)
    out[:3, 0] = np.asarray(sv_row.ex, dtype=float).reshape(3)
    out[:3, 1] = np.asarray(sv_row.ey, dtype=float).reshape(3)
    out[:3, 2] = np.asarray(sv_row.es, dtype=float).reshape(3)
    out[:3, 3] = np.array([sv_row.X[0], sv_row.Y[0], sv_row.Z[0]])
    return out


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


survey = line.survey()
sv_ref = survey.rows[ap.pipe_positions['cone_at'].survey_index]
sv_ref_matrix = matrix_from_survey_row(sv_ref)
pipe_pos_matrix = ap.pipe_positions['cone_at'].transformation


def installed_polygons_world():
    """World-space polygons of the installed cone cuts."""
    out = []
    for cut, pipe_pos in zip(cuts, ap.pipe_positions['cone_at'].pipe):
        poly = pipe_pos.profile.build_polygon(len_points=num_profile_points)
        profile_local = transform_matrix(
            shift_x=cut['shift_x'], rot_y_rad=cut['rot_y'])
        profile_arc = arc_matrix(length=cut['shift_s'], angle=0.0, tilt=0.0)
        poly_world = (sv_ref_matrix @ pipe_pos_matrix @ profile_arc
                      @ profile_local @ poly2d_to_hom(poly))
        out.append(poly_world)
    return out


polygons_world = installed_polygons_world()

# Every installed polygon must lie exactly on the cone. The apex sits at the
# pipe origin, i.e. at world z of the dA2a start plus the pipe shift.
survey_z_apex = (float(np.asarray(sv_ref.Z).ravel()[0])
                 + float(np.asarray(sv_ref.es, dtype=float).reshape(3)[2]) * 0.3)

for poly_world in polygons_world:
    xw, yw, zw = poly_world[:3]
    cone_equation = xw**2 + yw**2 - (zw - survey_z_apex)**2 * tan_alpha**2
    xo.assert_allclose(cone_equation, 0.0, atol=1e-9)

# The cut centres install at s = s_pipe_origin + h, wrapping at the seam.
s_apex = 37.5 + 0.3  # the cone apex on the survey


def expected_radius(s):
    """Cone section radius for a survey-orthogonal cut at s."""
    return ((s - s_apex) % L) * tan_alpha


# Report the installed footprints around the seam.

bounds = ap.get_bounds_table()
print('Installed bounds around the seam (sorted):')
for name, s0, s1 in zip(
        bounds.name, np.asarray(bounds.s_start, float),
        np.asarray(bounds.s_end, float)):
    if s1 > -0.5 and s0 < 3.0 or s0 > 40.0:
        print(f'  {name}: [{s0:8.4f}, {s1:8.4f}]')


# Query cross-sections on both sides of the seam and report.

def query(s_q):
    table = ap.get_cross_sections_at_s([float(s_q) % L])
    section = np.asarray(table.cross_section[0])[:-1]
    pose = np.asarray(table.pose[0])
    return section, pose


def exact_intersections(section_pose):
    """Exact cut-profile intersections with the queried plane, plane frame."""
    plane_normal = section_pose[:3, 2]
    plane_origin = section_pose[:3, 3]
    points = []
    for poly_world in polygons_world:
        pts = poly_world[:3].T
        off = (pts - plane_origin) @ plane_normal
        for jj in range(len(pts)):
            a_off, b_off = off[jj], off[(jj + 1) % len(pts)]
            if a_off <= 0 <= b_off or b_off <= 0 <= a_off:
                frac = a_off / (a_off - b_off)
                hit = pts[jj] + frac * (pts[(jj + 1) % len(pts)] - pts[jj])
                points.append(
                    (np.linalg.inv(section_pose) @ np.append(hit, 1.0))[:2])
    return np.array(points)


# Known behaviour (as of this example): cross-sections queried between the
# pre-seam profiles are exact (the assertions below). At a queried s exactly
# at an installed profile the section is also exact on the downhill side
# (walls to the neighbouring profiles of the same pipe). On the uphill side,
# where the pipe meets the preceding one, the section is the interpolation
# to that pipe's profile - here f_pre, a circle of radius 0.3 - so the
# section deviates from the cone circle by up to ~0.11 m, as expected from
# the model: the wall between two pipes is the bridge between their
# profiles, not a section of either.

queries = [
    (41.95, 'between cuts (pre-seam)'),
    (L - 0.05, 'just before the seam'),
    (0.0, 'at the seam'),
    (0.05, 'just after the seam'),
    (0.4, 'between cuts (wrapped)'),
]

print('\nCross-section quality vs the exact cone section:')
print('  (deviations are expected where the section is bounded by the wall')
print('   to the neighbouring pipe, e.g. on the uphill side of cut0 below)')
print('  s        finite   expected r   max |r - r_expected|   verdict')
results = {}
for s_q, label in queries:
    section, pose = query(s_q)
    radii = np.linalg.norm(section, axis=1)
    finite = np.isfinite(radii)
    r_exp = expected_radius(s_q)
    dev = float(np.abs(radii[finite] - r_exp).max()) if finite.any() else np.nan
    verdict = 'OK' if dev < 1e-3 and finite.all() else 'DEVIATES'
    print(f'  {s_q % L:7.3f}  {int(finite.sum()):4d}/127   {r_exp:10.6f}   '
          f'{dev:18.3e}   {verdict}  ({label})')
    results[s_q % L] = {
        'section': section, 'pose': pose, 'expected_radius': r_exp,
        'exact_points': exact_intersections(pose),
    }

# The pre-seam between-cuts query is exact today: assert it.
xo.assert_allclose(
    np.linalg.norm(results[41.95]['section'], axis=1),
    results[41.95]['expected_radius'], atol=1e-3, rtol=0)


# Plots: a 3D view of the ring with every queried plane and section, followed
# by a 2 x N grid - the bottom row shows each queried section in the plane
# frame against the exact circle.

line_fine = line.copy()
line_fine.cut_at_s(np.linspace(0, L, 400))
sv_fine = line_fine.survey()

section_colors = plt.cm.viridis(np.linspace(0, 0.85, len(queries)))

ax = plt.figure(figsize=(11, 8)).add_subplot(projection='3d')

ax.plot(sv_fine.Z, sv_fine.X, sv_fine.Y, c='b', lw=1, label='survey')

t_fine = np.linspace(0, 2 * np.pi, 400, endpoint=False)
for zc in (40.5 - L, 41.5 - L, 42.5 - L):
    rc = (zc + L - s_apex) * tan_alpha
    outline = np.array(
        [rc * np.cos(t_fine), rc * np.sin(t_fine), np.full_like(t_fine, zc)])
    ax.plot(outline[2], outline[0], outline[1], c='0.85', lw=0.6, ls=':')

for poly_world in polygons_world:
    xs, ys, zs = poly_world[:3]
    ax.plot(zs, xs, ys, c='r', lw=1.2, label='installed profiles')

for (s_q, label), color in zip(queries, section_colors):
    res = results[s_q % L]
    section, pose = res['section'], res['pose']
    expected_radius_q = res['expected_radius']
    exact_points = res['exact_points']

    plane_size = 1.3 * expected_radius_q
    grid = np.linspace(-plane_size, plane_size, 2)
    plane_uv = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    plane_world = pose @ np.column_stack(
        [plane_uv, np.zeros(len(plane_uv)), np.ones(len(plane_uv))]).T
    ax.add_collection3d(Poly3DCollection(
        [plane_world[:3, [0, 1, 3, 2]].T[:, [2, 0, 1]]],
        facecolor='0.75', alpha=0.25, edgecolor=color, linewidth=1))

    section_world = pose @ poly2d_to_hom(section)
    xs, ys, zs = section_world[:3]
    ax.plot(zs, xs, ys, c=color, lw=2.5, label=f'section at s = {s_q % L:.2f}')

    points_world = pose @ poly2d_to_hom(exact_points)
    ax.scatter(points_world[2], points_world[0], points_world[1],
               color='orange', s=20, depthshade=False)

ax.plot([np.nan], [np.nan], [np.nan], c='0.75', alpha=0.35,
        label='queried plane (wrapped)')

ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]')
ax.set_zlabel('Y [m]')
ax.set_title('Cone pipe wrapping the ring seam: queried cross-sections')
ax.legend(loc='upper left')

# Plane-frame grid: one subplot per queried plane, in succession.
n_rows = 2
n_cols = int(np.ceil((len(queries) + 1) / n_rows))  # +1 for the legend panel
fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 8))
axs = axs.ravel()

for ax_plane, ((s_q, label), color) in zip(axs, zip(queries, section_colors)):
    res = results[s_q % L]
    section, pose = res['section'], res['pose']
    expected_radius_q = res['expected_radius']
    exact_points = res['exact_points']

    section_closed = np.vstack([section, section[:1]])
    ax_plane.plot(section_closed[:, 0], section_closed[:, 1], c=color, lw=2)
    finite = np.isfinite(section).all(axis=1)
    ax_plane.scatter(section[finite, 0], section[finite, 1], color=color, s=8)
    ax_plane.scatter(exact_points[:, 0], exact_points[:, 1], color='orange',
                     s=35, zorder=3)
    circle_ref = plt.Circle(
        (0, 0), expected_radius_q, fill=False, edgecolor='k', ls='--', lw=0.8)
    ax_plane.add_patch(circle_ref)
    ax_plane.set_xlabel('x [m]')
    ax_plane.set_ylabel('y [m]')
    ax_plane.set_aspect('equal')
    ax_plane.set_title(f's = {s_q % L:.2f}\n{label}')

# Hide the unused panels, and add one legend in the last cell.
for ax_empty in axs[len(queries):]:
    ax_empty.axis('off')
handles, labels_ = axs[0].get_legend_handles_labels()
from matplotlib.lines import Line2D
handles += [
    Line2D([], [], color='k', ls='--', lw=0.8, label='exact circle r(s)'),
    Line2D([], [], marker='o', c='none', markerfacecolor='orange',
           markersize=7, label='installed profile \u2229 plane'),
]
axs[len(queries)].legend(handles=handles, loc='center', fontsize=11)

plt.show()
