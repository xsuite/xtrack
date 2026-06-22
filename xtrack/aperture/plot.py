from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from xtrack.aperture.structures import (
    Circle,
    Ellipse,
    Octagon,
    Polygon,
    Racetrack,
    Rectangle,
    RectEllipse,
    SVGShape,
)
from xtrack.aperture.transform import arc_matrix, transform_matrix
from xtrack.survey import survey_relative_transform


_GEOMETRY_TOL = 1e-12


@dataclass
class _ProjectedProfile:
    shift_s: float
    endpoints_curvilinear: np.ndarray


def _shape_projection_extents(shape, direction: np.ndarray) -> tuple[float, float]:
    """Return the extrema of a shape projected onto a unit direction."""
    dx, dy = direction

    if isinstance(shape, Circle):
        support = float(shape.radius)

    elif isinstance(shape, Ellipse):
        support = float(np.hypot(shape.half_major * dx, shape.half_minor * dy))

    elif isinstance(shape, Rectangle):
        support = float(shape.half_width * abs(dx) + shape.half_height * abs(dy))

    elif isinstance(shape, Racetrack):
        width = float(shape.half_width)
        height = float(shape.half_height)
        major = float(shape.half_major)
        minor = float(shape.half_minor)
        if major <= 0 or minor <= 0:
            support = width * abs(dx) + height * abs(dy)
        else:
            if major > width + _GEOMETRY_TOL or minor > height + _GEOMETRY_TOL:
                raise ValueError('Racetrack corner radii exceed its half dimensions.')
            support = (
                (width - major) * abs(dx)
                + (height - minor) * abs(dy)
                + np.hypot(major * dx, minor * dy)
            )

    elif isinstance(shape, Octagon):
        width = float(shape.half_width)
        height = float(shape.half_height)
        diagonal = float(shape.half_diagonal)
        corner_x = np.sqrt(2.0) * diagonal - height
        corner_y = np.sqrt(2.0) * diagonal - width
        if corner_x < -_GEOMETRY_TOL or corner_y < -_GEOMETRY_TOL:
            raise ValueError('Invalid octagon dimensions.')
        vertices = np.array([
            [width, -corner_y],
            [width, corner_y],
            [corner_x, height],
            [-corner_x, height],
            [-width, corner_y],
            [-width, -corner_y],
            [-corner_x, -height],
            [corner_x, -height],
        ])
        values = vertices @ direction
        return float(np.min(values)), float(np.max(values))

    elif isinstance(shape, Polygon):
        vertices = shape.vertices.to_nparray()
        if len(vertices) > 1 and np.allclose(
            vertices[0], vertices[-1], atol=_GEOMETRY_TOL, rtol=0
        ):
            vertices = vertices[:-1]
        if len(vertices) == 0:
            raise ValueError('Cannot project an empty polygon.')
        values = vertices @ direction
        return float(np.min(values)), float(np.max(values))

    elif isinstance(shape, RectEllipse):
        candidates = _rectellipse_boundary_candidates(shape, direction)
        values = candidates @ direction
        return float(np.min(values)), float(np.max(values))

    elif isinstance(shape, SVGShape):
        raise NotImplementedError('Floor projection is not implemented for SVGShape profiles.')

    else:
        raise TypeError(f'Unsupported aperture profile shape {type(shape).__name__}.')

    return -support, support


def _rectellipse_boundary_candidates(shape: RectEllipse, direction: np.ndarray) -> np.ndarray:
    width = float(shape.half_width)
    height = float(shape.half_height)
    major = float(shape.half_major)
    minor = float(shape.half_minor)
    candidates = []

    def add(x, y):
        if (
            abs(x) <= width + _GEOMETRY_TOL
            and abs(y) <= height + _GEOMETRY_TOL
            and (x / major) ** 2 + (y / minor) ** 2 <= 1 + _GEOMETRY_TOL
        ):
            candidates.append((x, y))

    for x in (-width, width):
        for y in (-height, height):
            add(x, y)

    ellipse_support = np.hypot(major * direction[0], minor * direction[1])
    if ellipse_support > 0:
        support_point = np.array([
            major**2 * direction[0] / ellipse_support,
            minor**2 * direction[1] / ellipse_support,
        ])
        add(*support_point)
        add(*-support_point)

    if width <= major + _GEOMETRY_TOL:
        y = minor * np.sqrt(max(0.0, 1.0 - (width / major) ** 2))
        for x in (-width, width):
            add(x, -y)
            add(x, y)

    if height <= minor + _GEOMETRY_TOL:
        x = major * np.sqrt(max(0.0, 1.0 - (height / minor) ** 2))
        for y in (-height, height):
            add(-x, y)
            add(x, y)

    # Axis extrema cover the unconstrained ellipse support for axial directions.
    add(-major, 0.0)
    add(major, 0.0)
    add(0.0, -minor)
    add(0.0, minor)

    if not candidates:
        raise ValueError('RectEllipse has no valid boundary points.')
    return np.unique(np.asarray(candidates, dtype=float), axis=0)


def _profile_projection_segment(
    shape,
    profile_to_plot: np.ndarray,
) -> np.ndarray | None:
    """Project a profile onto its floor-intersection line in profile coordinates."""
    plane_coefficients = np.asarray(profile_to_plot[1, [0, 1, 3]], dtype=float)
    normal = plane_coefficients[:2]
    offset = plane_coefficients[2]
    normal_sq = float(normal @ normal)

    if normal_sq <= _GEOMETRY_TOL**2:
        if abs(offset) <= _GEOMETRY_TOL:
            raise ValueError('Installed profile plane is coplanar with the plotting floor plane.')
        return None

    direction = np.array([-normal[1], normal[0]]) / np.sqrt(normal_sq)
    line_origin = -offset * normal / normal_sq
    min_projection, max_projection = _shape_projection_extents(shape, direction)
    origin_projection = float(line_origin @ direction)

    return np.array([
        line_origin + (min_projection - origin_projection) * direction,
        line_origin + (max_projection - origin_projection) * direction,
    ])


def _profile_to_pipe_matrix(profile_position, curvature: float) -> np.ndarray:
    local = transform_matrix(
        shift_x=float(profile_position.shift_x),
        shift_y=float(profile_position.shift_y),
        rot_y_rad=float(profile_position.rot_y_rad),
        rot_x_rad=float(profile_position.rot_x_rad),
        rot_z_rad=float(profile_position.rot_s_rad),
    )
    shift_s = float(profile_position.shift_s)
    return arc_matrix(shift_s, curvature * shift_s, 0.0) @ local


def _cartesian_to_curvilinear(points: np.ndarray, curvature: float, reference_s: float) -> np.ndarray:
    if abs(curvature) <= _GEOMETRY_TOL:
        return points.copy()

    radius = 1.0 / curvature
    sign = 1.0 if curvature >= 0 else -1.0
    radial_x = points[:, 0] + radius
    radial_z = points[:, 2]
    rho = sign * np.hypot(radial_x, radial_z)
    theta = np.arctan2(sign * radial_z, sign * radial_x)
    s_coord = theta / curvature

    period = 2.0 * np.pi / abs(curvature)
    s_coord += np.round((reference_s - s_coord) / period) * period

    return np.column_stack([rho - radius, points[:, 1], s_coord])


def _curvilinear_to_cartesian(points: np.ndarray, curvature: float) -> np.ndarray:
    if abs(curvature) <= _GEOMETRY_TOL:
        return points.copy()

    radius = 1.0 / curvature
    theta = curvature * points[:, 2]
    radial = radius + points[:, 0]
    return np.column_stack([
        radial * np.cos(theta) - radius,
        points[:, 1],
        radial * np.sin(theta),
    ])


def _installed_profile_projection(
    aperture,
    pipe_position,
    profile_position,
    pipe_to_plot: np.ndarray,
) -> _ProjectedProfile | None:
    pipe = pipe_position.pipe
    profile = profile_position.profile.raw
    profile_to_pipe = _profile_to_pipe_matrix(profile_position.raw, pipe.curvature)
    profile_to_plot = pipe_to_plot @ profile_to_pipe
    segment_local = _profile_projection_segment(profile.shape, profile_to_plot)
    if segment_local is None:
        return None

    segment_homogeneous = np.column_stack([
        segment_local,
        np.zeros(2),
        np.ones(2),
    ])
    segment_pipe = (profile_to_pipe @ segment_homogeneous.T).T[:, :3]
    segment_curvilinear = _cartesian_to_curvilinear(
        segment_pipe,
        pipe.curvature,
        reference_s=float(profile_position.shift_s),
    )
    return _ProjectedProfile(
        shift_s=float(profile_position.shift_s),
        endpoints_curvilinear=segment_curvilinear,
    )


def _orient_profile_segments(profiles: list[_ProjectedProfile | None]) -> None:
    previous = None
    for profile in profiles:
        if profile is None:
            previous = None
            continue
        if previous is not None:
            same = np.sum((previous - profile.endpoints_curvilinear) ** 2)
            reversed_order = np.sum((previous - profile.endpoints_curvilinear[::-1]) ** 2)
            if reversed_order < same:
                profile.endpoints_curvilinear = profile.endpoints_curvilinear[::-1].copy()
        previous = profile.endpoints_curvilinear


def _interpolate_profile_side(
    start: np.ndarray,
    end: np.ndarray,
    curvature: float,
    delta_s: float,
    max_curve_angle_rad: float,
) -> np.ndarray:
    if abs(curvature) <= _GEOMETRY_TOL:
        num_intervals = 1
    else:
        angle = abs(curvature * delta_s)
        num_intervals = max(1, int(np.ceil(angle / max_curve_angle_rad)))
    fractions = np.linspace(0.0, 1.0, num_intervals + 1)
    return start[None, :] + fractions[:, None] * (end - start)[None, :]


def _pipe_projection_polygons(
    projected_profiles: list[_ProjectedProfile | None],
    curvature: float,
    max_curve_angle_rad: float,
) -> list[np.ndarray]:
    """Build filled pipe sections in pipe Cartesian coordinates."""
    polygons = []
    for start, end in zip(projected_profiles[:-1], projected_profiles[1:]):
        if start is None or end is None:
            continue
        delta_s = end.shift_s - start.shift_s
        side_a = _interpolate_profile_side(
            start.endpoints_curvilinear[0],
            end.endpoints_curvilinear[0],
            curvature,
            delta_s,
            max_curve_angle_rad,
        )
        side_b = _interpolate_profile_side(
            start.endpoints_curvilinear[1],
            end.endpoints_curvilinear[1],
            curvature,
            delta_s,
            max_curve_angle_rad,
        )
        boundary_curvilinear = np.vstack([side_a, side_b[::-1]])
        polygons.append(_curvilinear_to_cartesian(boundary_curvilinear, curvature))
    return polygons


def _pipe_axis_points(pipe, max_curve_angle_rad: float) -> np.ndarray:
    start_s = float(pipe[0].shift_s)
    end_s = float(pipe[len(pipe) - 1].shift_s)
    if abs(pipe.curvature) <= _GEOMETRY_TOL:
        num_intervals = 1
    else:
        angle = abs(pipe.curvature * (end_s - start_s))
        num_intervals = max(1, int(np.ceil(angle / max_curve_angle_rad)))
    s_values = np.linspace(start_s, end_s, num_intervals + 1)
    points = np.column_stack([np.zeros_like(s_values), np.zeros_like(s_values), s_values])
    return _curvilinear_to_cartesian(points, pipe.curvature)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (transform @ homogeneous.T).T[:, :3]


def _hashed_colour(name: str, palette: list[str]) -> str:
    if not palette:
        return 'C0'
    digest = hashlib.sha1(name.encode('utf-8')).digest()
    return palette[int.from_bytes(digest[:4], 'little') % len(palette)]


def _split_wrapped_interval(start, end, line_length, s_tol):
    start = float(np.mod(start, line_length))
    end = float(np.mod(end, line_length))
    if start > end + s_tol:
        return [(start, line_length), (0.0, end)]
    return [(start, end)]


def _pipe_in_s_range(aperture, row, s_range, origin_s):
    if s_range is None:
        return True

    window_start = origin_s + s_range[0]
    window_end = origin_s + s_range[1]
    if not aperture.is_ring:
        return (
            row.s_end >= window_start - aperture.s_tol
            and row.s_start <= window_end + aperture.s_tol
        )

    line_length = aperture.line.get_length()
    if s_range[1] - s_range[0] >= line_length - aperture.s_tol:
        return True
    row_segments = _split_wrapped_interval(row.s_start, row.s_end, line_length, aperture.s_tol)
    window_segments = _split_wrapped_interval(window_start, window_end, line_length, aperture.s_tol)
    return any(
        row_start <= range_end + aperture.s_tol and range_start <= row_end + aperture.s_tol
        for row_start, row_end in row_segments
        for range_start, range_end in window_segments
    )


def _enable_pipe_annotations(ax, patches) -> None:
    if not patches:
        return

    annotation = ax.annotate(
        '',
        xy=(0, 0),
        xytext=(8, 8),
        textcoords='offset points',
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.9},
    )
    annotation.set_visible(False)

    def on_pick(event):
        artist = event.artist
        metadata = getattr(artist, '_xtrack_pipe_metadata', None)
        if metadata is None:
            return
        mouse = event.mouseevent
        annotation.xy = (mouse.xdata, mouse.ydata)
        s_min, s_max = metadata['s_range']
        annotation.set_text(
            f"{metadata['pipe_position_name']}\n"
            f"pipe: {metadata['pipe_name']}\n"
            f"survey ref.: {metadata['survey_reference']}\n"
            fr"$s \in [{round(s_min, 3)}, {round(s_max, 3)}]$"
        )
        annotation.set_visible(True)
        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect('pick_event', on_pick)


def plot_floor_projection(
    aperture,
    *,
    ax=None,
    max_curve_angle_rad=np.deg2rad(1),
    origin=None,
    s_range=None,
    aspect='auto',
):
    """Plot analytically projected installed pipe sections onto the floor."""
    if max_curve_angle_rad <= 0:
        raise ValueError('`max_curve_angle_rad` must be positive.')
    if s_range is not None and s_range[0] > s_range[1]:
        raise ValueError('The `origin` pipe position is outside of the `s_range` specified.')

    from matplotlib import pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

    ax = ax or plt.gca()
    palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])
    pipe_table = aperture.get_pipe_table()
    plot_shift = np.identity(4)
    origin_s = 0.0

    if origin is not None:
        origin_position = aperture.pipe_positions[origin]
        origin_row = pipe_table.rows[origin]
        origin_s = float(np.asarray(origin_row.s_start).item())
        origin_survey = survey_relative_transform(
            aperture.survey, 0, origin_position.survey_reference_name
        )
        plot_shift = np.linalg.inv(origin_survey @ origin_position.transformation)

    patches = []
    for row in pipe_table.rows:
        if not _pipe_in_s_range(aperture, row, s_range, origin_s):
            continue

        pipe_position = aperture.pipe_positions[row.name]
        pipe = pipe_position.pipe
        survey_transform = survey_relative_transform(
            aperture.survey, 0, pipe_position.survey_reference_name
        )
        pipe_to_plot = plot_shift @ survey_transform @ pipe_position.transformation
        colour = _hashed_colour(pipe.name, palette)

        projected_profiles = [
            _installed_profile_projection(aperture, pipe_position, profile_position, pipe_to_plot)
            for profile_position in pipe
        ]
        _orient_profile_segments(projected_profiles)

        for polygon_pipe in _pipe_projection_polygons(
            projected_profiles,
            pipe.curvature,
            max_curve_angle_rad,
        ):
            polygon_plot = _transform_points(polygon_pipe, pipe_to_plot)
            patch = PolygonPatch(
                polygon_plot[:, [2, 0]],
                closed=True,
                facecolor=colour,
                edgecolor=colour,
                alpha=0.45,
                linewidth=0.8,
                picker=True,
            )
            patch._xtrack_pipe_metadata = {
                'pipe_position_name': row.name,
                'pipe_name': pipe.name,
                'survey_reference': row.survey_reference,
                's_range': (row.s_start, row.s_end),
            }
            ax.add_patch(patch)
            patches.append(patch)

        axis_plot = _transform_points(
            _pipe_axis_points(pipe, max_curve_angle_rad),
            pipe_to_plot,
        )
        ax.plot(axis_plot[:, 2], axis_plot[:, 0], color=colour, linestyle='--')

    _enable_pipe_annotations(ax, patches)
    ax.set_xlabel('Z [m]')
    ax.set_ylabel('X [m]')
    ax.set_aspect(aspect)
    return ax
