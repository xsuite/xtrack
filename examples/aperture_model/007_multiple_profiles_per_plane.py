import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from xtrack.aperture.transform import arc_matrix, transform_matrix

# A straight pipe with several large profiles, all rotated about the vertical
# (y) axis by the same angle. Each tilted profile spans a longitudinal range of
# ``2 * radius * sin(rot_y)`` along the survey, larger than the spacing between
# the profiles. The survey still passes through the centre of every profile,
# but a single plane orthogonal to the survey cuts through several installed
# profiles at once.
#
# The cross-section interpolation combines all the installed profiles cut by
# the plane: for each polygon vertex it follows the corresponding vertex
# across consecutive profiles and intersects that segment with the plane, so
# the returned cross-section is the envelope of all the intersected profiles.

radius = 1.0
rot_y = np.deg2rad(45.0)
spacing = 0.28
num_profiles = 8
s_first = 2.5
drift_length = 5.2

# Centre the profile stack longitudinally in the drift.
s_installed = s_first + spacing * np.arange(num_profiles)
s_installed += 0.5 * (drift_length - (s_installed[-1] - s_installed[0])) \
              - s_installed[0]

# Build a straight line and install the tilted profiles.

env = xt.Environment()
drift = env.new('drift', xt.Drift, length=drift_length)
line = env.new_line(name='line', components=[drift])

profile = env.aperture.new_profile('circ', 'Circle', radius=radius)
pipe = env.aperture.new_pipe('tilted_pipe', curvature=0.0, positions=[
    profile.place(at=s, rot_y_rad=rot_y) for s in s_installed
])
ap = line.aperture
ap.num_profile_points = 128
ap.place_pipe('tp', pipe, at='drift@start')
ap.build_model()

# Query a plane in the middle of the stack. With the longitudinal footprints
# of the tilted profiles overlapping, this plane cuts every installed profile
# except the first and the last one.

bounds = ap.get_bounds_table()
s_query = 0.5 * (s_installed[0] + s_installed[-1])
num_cut = int(np.sum((bounds.s_start <= s_query) & (s_query <= bounds.s_end)))
print(f'Querying s = {s_query:.3f}, which cuts {num_cut} of the '
      f'{num_profiles} installed profiles at once.')

# Request the cross-section at that plane.

cross_sections = ap.get_cross_sections_at_s([s_query])
section = cross_sections.cross_section[0]
pose = cross_sections.pose[0]
num_nan = int(np.sum(np.all(np.isnan(section), axis=1)))
print(f'Cross-section at s = {s_query:.3f}: {num_nan} of {len(section)} '
      f'points could not be constructed (NaN).')

# Plotting helpers.

def matrix_from_survey_row(sv_row):
    out = np.identity(4)
    out[:3, 0] = sv_row.ex
    out[:3, 1] = sv_row.ey
    out[:3, 2] = sv_row.es
    out[:3, 3] = np.array([sv_row.X[0], sv_row.Y[0], sv_row.Z[0]])
    return out


def poly2d_to_hom(poly2d):
    n = poly2d.shape[0]
    return np.column_stack([poly2d, np.zeros(n), np.ones(n)]).T


def plot_polygon_3d(ax, points_world, **kwargs):
    """Plot a closed 3D polygon given as homogeneous points (4, n)."""
    points_world = np.hstack([points_world, points_world[:, :1]])
    xs, ys, zs = points_world[:3]
    ax.plot(zs, xs, ys, **kwargs)


def profile_plane_intersections(section=radius, rot_y=rot_y, s_positions=None):
    """Exact intersections of installed profile circles with the queried plane.

    For a straight survey along z, the queried plane is z = s_query, and a
    circle installed at s_center and rotated by rot_y about y intersects it at
    x = -(s_query - s_center) / tan(rot_y), y = +/- sqrt(R^2 - ds^2/sin^2(rot_y)),
    where ds = s_query - s_center (positive rot_y tilts the profile so that its
    local +x leans towards negative z). Profiles with |ds| > R * sin(rot_y) are
    not intersected.

    Returns a dict mapping the installed s to an (n, 2) array of plane-frame
    (x, y) intersection points.
    """
    out = {}
    for s_center in s_positions:
        ds = s_query - s_center
        cos_theta = ds / (section * np.sin(rot_y))
        if abs(cos_theta) > 1:
            continue
        theta = np.arccos(cos_theta)
        x = -ds / np.tan(rot_y)
        ys = section * np.sin(np.array([theta, -theta]))
        out[s_center] = np.column_stack([np.full(2, x), ys])
    return out


# 3D view: the survey (blue), installed profiles (red), the queried plane
# (grey), and the requested cross-section (green).

survey = line.survey()
ax = plt.figure(figsize=(11, 8)).add_subplot(projection='3d')

line_fine = line.copy()
line_fine.cut_at_s(np.linspace(0, drift_length, 200))
sv_fine = line_fine.survey()
ax.plot(sv_fine.Z, sv_fine.X, sv_fine.Y, c='b', label='survey')

for pipe_pos in ap.pipe_positions:
    sv_ref = survey.rows[pipe_pos.survey_index]
    sv_ref_matrix = matrix_from_survey_row(sv_ref)
    installed_pipe = pipe_pos.pipe
    for profile_pos in installed_pipe:
        poly = profile_pos.profile.build_polygon(len_points=128)
        profile_local = transform_matrix(
            shift_x=profile_pos.shift_x,
            shift_y=profile_pos.shift_y,
            rot_y_rad=profile_pos.rot_y_rad,
            rot_x_rad=profile_pos.rot_x_rad,
            rot_z_rad=profile_pos.rot_s_rad,
        )
        # Longitudinal placement of the profile within the pipe.
        profile_arc = arc_matrix(
            length=profile_pos.shift_s,
            angle=installed_pipe.curvature * profile_pos.shift_s,
            tilt=0,
        )
        profile_world = (sv_ref_matrix @ pipe_pos.transformation
                         @ profile_arc @ profile_local @ poly2d_to_hom(poly))
        plot_polygon_3d(ax, profile_world, c='r', lw=1.5)

# The queried plane, drawn as a translucent filled patch orthogonal to the
# survey, with a thin outline.
plane_size = 1.6 * radius
grid = np.linspace(-plane_size, plane_size, 2)
plane_uv = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
plane_world = pose @ np.column_stack(
    [plane_uv, np.zeros(len(plane_uv)), np.ones(len(plane_uv))]).T
plane_quad = Poly3DCollection(
    # Poly3DCollection expects vertices in axis order (X, Y, Z), while this
    # script plots physical (x, y, z) as ax.plot(z, x, y) - hence the reorder.
    [plane_world[:3, [0, 1, 3, 2]].T[:, [2, 0, 1]]], facecolor='0.75', alpha=0.35,
    edgecolor='0.4', linewidth=1)
ax.add_collection3d(plane_quad)
# Proxy artist for the legend.
ax.plot([np.nan], [np.nan], [np.nan], c='0.75', alpha=0.35,
        label=f'queried plane (s = {s_query:.2f})')

# Exact intersections of the installed profiles with the queried plane
# (orange): the reconstructed cross-section should pass through all of them.
intersections = profile_plane_intersections(s_positions=s_installed)
for ii, (s_center, points) in enumerate(intersections.items()):
    points_hom = poly2d_to_hom(points)
    points_world = pose @ points_hom
    label = 'installed profile ∩ plane' if ii == 0 else None
    ax.scatter(points_world[2], points_world[0], points_world[1],
               c='orange', s=25, label=label, depthshade=False)

# The cross-section returned by the model, in world coordinates.
section_world = pose @ poly2d_to_hom(section)
# Break the line wherever points are missing.
valid = np.isfinite(section_world[:3]).all(axis=0)
xs, ys, zs = section_world[:3]
xs, ys, zs = (np.where(valid, coord, np.nan) for coord in (xs, ys, zs))
ax.plot(zs, xs, ys, c='g', lw=3, label='cross-section at plane')

ax.set_xlabel('Z [m]')
ax.set_ylabel('X [m]')
ax.set_zlabel('Y [m]')
ax.set_title(f'Queried plane (s = {s_query:.2f}) cutting {num_cut} of '
                 f'{num_profiles} installed profiles at once')
ax.legend(loc='upper left')
ax.set_box_aspect((1.6, 1, 1))

# Cross-section in the plane frame. The green polygon is the envelope of all
# installed profiles cut by the plane; the orange dots are the exact
# intersections of the profiles with the plane, through which the
# cross-section passes.

fig, ax_plane = plt.subplots(figsize=(7, 7))
valid = np.isfinite(section).all(axis=1)

for ii, (s_center, points) in enumerate(intersections.items()):
    label = 'installed profile ∩ plane' if ii == 0 else None
    ax_plane.scatter(points[:, 0], points[:, 1], c='orange', s=45, label=label,
                     zorder=3)

section_closed = np.vstack([section, section[:1]])
ax_plane.plot(section_closed[:, 0], section_closed[:, 1], c='g', lw=2,
              label='cross-section', zorder=2)
ax_plane.scatter(section[valid, 0], section[valid, 1], c='g', s=8, zorder=2)

# Mark both ends of every gap left by missing (NaN) points, if any.
invalid = ~valid
if invalid.any():
    gap_edges = np.where(invalid[:-1] != invalid[1:])[0]
    gap_ends = np.unique(np.concatenate([gap_edges, gap_edges + 1]))
    finite_gap_ends = gap_ends[np.isfinite(section[gap_ends, 0])]
    ax_plane.scatter(section[finite_gap_ends, 0], section[finite_gap_ends, 1],
                     facecolors='none', edgecolors='r', s=120, linewidths=1.5,
                     label='gaps: missing (NaN) points in between', zorder=4)

ax_plane.set_xlabel('x [m]')
ax_plane.set_ylabel('y [m]')
ax_plane.set_aspect('equal')
ax_plane.set_title(f'Cross-section at s = {s_query:.2f}'
                   + (f' ({int(invalid.sum())} of {len(section)} points missing)'
                      if invalid.any() else ''))
ax_plane.legend(loc='upper right')

plt.show()
