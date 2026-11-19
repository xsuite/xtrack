"""Convert aperture sections to native tracking limits where geometry permits."""

from warnings import warn

import numpy as np
from xdeps.refs import BaseRef

from .profile_converters import (
    LimitEllipse, LimitPolygon, LimitRacetrack, LimitRect, LimitRectEllipse,
)
from .structures import Circle, Ellipse, Octagon, Polygon, Racetrack, Rectangle, RectEllipse
from .transform import arc_matrix


_GEOMETRY_ATOL = 1e-10
_CONTOUR_ATOL = 1e-6  # metres, for numerical fits to sampled cuts


def _value(value):
    if isinstance(value, BaseRef):
        return value._value
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    return value


def _same_field(a, b):
    return (np.array_equal(a._value, b._value)
            and str(a._expr) == str(b._expr))


class _Expressions:
    """Track which aperture bindings a native limit can reproduce.

    Each read returns a live expression or a constant. Comparing the reads for
    one section with the bindings on its source profiles identifies what froze
    when constructing that limit.
    """

    def __init__(self, env):
        self.tasks = {lhs: rhs for lhs, rhs in env.ref_manager.dump()
                      if lhs.startswith('aperture.')}
        self.used = set()

    def read(self, ref):
        if ref._expr is not None:
            self.used.add(str(ref))
            return ref._expr
        return ref._value

    def under(self, *prefixes):
        return {lhs for lhs in self.tasks if lhs.startswith(prefixes)}

    def mirror(self, first, second):
        """Equivalent profiles may have separate targets for the same expressions."""
        first, second = str(first), str(second)
        self.used.update(second + lhs[len(first):] for lhs in tuple(self.used)
                         if lhs.startswith(first + '.'))


def _same_shape(first, second, expressions):
    a, b = first._value, second._value
    if a.__class__ is not b.__class__:
        return False
    if not all(np.array_equal(value, b._to_dict()[key])
               for key, value in a._to_dict().items()):
        return False
    def bindings(ref):
        prefix = str(ref) + '.'
        return {lhs[len(prefix):]: rhs for lhs, rhs in expressions.tasks.items()
                if lhs.startswith(prefix)}
    return bindings(first) == bindings(second)


def _shape_limit(shape, ref, expressions, reflection=1):
    """Return a native limit class and parameters, retaining parameter expressions."""
    read = lambda key: expressions.read(getattr(ref, key))
    if isinstance(shape, Circle):
        radius = read('radius')
        return LimitEllipse, dict(a=radius, b=radius)
    if isinstance(shape, Rectangle):
        x, y = read('max_x'), read('max_y')
        return LimitRect, dict(min_x=-x, max_x=x, min_y=-y, max_y=y)
    if isinstance(shape, Ellipse):
        return LimitEllipse, dict(a=read('a'), b=read('b'))
    if isinstance(shape, RectEllipse):
        return LimitRectEllipse, {key: read(key) for key in ('max_x', 'max_y', 'a', 'b')}
    if isinstance(shape, Racetrack):
        if not (0 <= shape.a <= shape.max_x and 0 <= shape.b <= shape.max_y):
            return None  # These profiles cannot be represented by LimitRacetrack.
        x, y = read('max_x'), read('max_y')
        return LimitRacetrack, dict(min_x=-x, max_x=x, min_y=-y, max_y=y,
                                   a=read('a'), b=read('b'))
    if isinstance(shape, Polygon):
        vertices = shape.vertices.to_nparray()
        count = len(vertices)
        if count > 1 and np.array_equal(vertices[0], vertices[-1]):
            count -= 1
        return LimitPolygon, dict(
            x_vertices=[expressions.read(ref.vertices[ii, 0]) for ii in range(count)],
            y_vertices=[reflection * expressions.read(ref.vertices[ii, 1])
                        for ii in range(count)])
    if isinstance(shape, Octagon):
        # Preserve the eight corners exactly in the ordinary octagon regime.
        # Other clipping regimes use the general polygon sampler.
        diagonal = np.sqrt(2) * shape.max_diagonal
        if not max(shape.max_x, shape.max_y) < diagonal < shape.max_x + shape.max_y:
            return None
        x, y = read('max_x'), read('max_y')
        diagonal = np.sqrt(2) * read('max_diagonal')
        u, v = diagonal - y, diagonal - x
        return LimitPolygon, dict(x_vertices=[x, u, -u, -x, -x, -u, u, x],
                                  y_vertices=[reflection * yy for yy in
                                              (v, y, y, v, -v, -y, -y, -v)])
    return None


def _pipe_distance_at_plane(origin, curvature, shifts):
    """Return the candidate arc length along the pipe at the survey plane."""
    if len(shifts) == 1:
        # The kernel projects a lone profile from its own plane.
        return shifts[0]
    if abs(curvature) < 1e-14:
        return origin[2]

    # In pipe coordinates the bend centre is at x = -1/curvature.
    # The radial angle gives the azimuth of a perpendicular cut, up to a turn.
    sign = np.sign(curvature)
    angle = np.arctan2(sign * origin[2], sign * (origin[0] + 1 / curvature))
    distance = angle / curvature
    turn_length = 2 * np.pi / abs(curvature)
    return distance + np.round((np.mean(shifts) - distance) / turn_length) * turn_length


def _perpendicular_cut(aperture, placement, pipe, plane):
    """Locate the cut in pipe coordinates and bracket it with two profiles."""
    shifts = np.array([position.shift_s for position in pipe.positions])
    if not len(shifts) or np.any(np.diff(shifts) <= 0):
        return None

    survey_pose = aperture._survey_data.pose.to_nparray()[placement.survey_index]
    pipe_pose = survey_pose @ placement.transformation.to_nparray()
    plane_in_pipe = np.linalg.inv(pipe_pose) @ plane
    curvature = float(pipe.curvature)
    distance = _pipe_distance_at_plane(plane_in_pipe[:3, 3], curvature, shifts)
    axis_in_plane = np.linalg.inv(plane_in_pipe) @ arc_matrix(distance, distance * curvature, 0.)

    # The pipe tangent must be normal to the cutting plane, in either direction.
    if (not np.allclose(axis_in_plane[:2, 2], 0, atol=_GEOMETRY_ATOL, rtol=0)
            or not np.allclose(axis_in_plane[2, :2], 0, atol=_GEOMETRY_ATOL, rtol=0)):
        return None
    # A single bound can be projected onto any parallel plane. With neighbours,
    # the longitudinal coordinate must lie in the plane and within the pipe.
    if aperture._aperture_bounds.count > 1 and abs(axis_in_plane[2, 3]) > _GEOMETRY_ATOL:
        return None
    if len(shifts) == 1:
        return 0, 0, distance, axis_in_plane
    if not shifts[0] - _GEOMETRY_ATOL <= distance <= shifts[-1] + _GEOMETRY_ATOL:
        return None
    left = np.searchsorted(shifts, distance, side='right') - 1
    left = int(np.clip(left, 0, len(shifts) - 2))
    return left, left + 1, distance, axis_in_plane


def _interpolated_limit(a, b, shape_a, shape_b, distance, axis_in_plane, expressions):
    """Build a limit from two profile-position refs in a perpendicular cut."""
    if any(abs(angle._value) > _GEOMETRY_ATOL
           for pos in (a, b) for angle in (pos.rot_x_rad, pos.rot_y_rad)):
        return None
    if not _same_field(a.rot_s_rad, b.rot_s_rad):
        return None
    same_shape = _same_shape(shape_a, shape_b, expressions)
    if not same_shape and not (isinstance(shape_a._value, Circle)
                               and isinstance(shape_b._value, Circle)):
        return None

    rotation = axis_in_plane[:2, :2]
    # Looking along a reversed pipe reflects its transverse coordinates.
    reflection = 1 if np.linalg.det(rotation) > 0 else -1
    spec = _shape_limit(shape_a._value, shape_a, expressions, reflection)
    if spec is None:
        return None
    limit_class, parameters = spec
    if same_shape:
        expressions.mirror(shape_a, shape_b)

    def interpolate(first, second):
        start_value = expressions.read(first)
        end_value = expressions.read(second)
        if _same_field(first, second):
            return start_value
        # The cut stays fixed. Keep the endpoint positions live so moving a
        # profile also changes the radius or transverse offset at this cut.
        start, end = expressions.read(a.shift_s), expressions.read(b.shift_s)
        weight = (distance - start) / (end - start)
        return (1 - weight) * start_value + weight * end_value

    if not same_shape:
        radius = interpolate(shape_a.radius, shape_b.radius)
        parameters.update(a=radius, b=radius)
    x = interpolate(a.shift_x, b.shift_x)
    y = interpolate(a.shift_y, b.shift_y)
    roll = expressions.read(a.rot_s_rad)
    expressions.read(b.rot_s_rad)  # The two roll bindings were checked equal.
    parameters.update(
        shift_x=axis_in_plane[0, 3] + rotation[0, 0] * x + rotation[0, 1] * y,
        shift_y=axis_in_plane[1, 3] + rotation[1, 0] * x + rotation[1, 1] * y,
        rot_s_rad=np.arctan2(rotation[1, 0], rotation[0, 0]) + reflection * roll)
    return limit_class, parameters


def _native_section(aperture, pipe_index, plane, expressions):
    """Locate a perpendicular cut, then translate its two profiles and bindings."""
    model = aperture._model
    placement = model.pipe_positions[pipe_index]
    pipe = model.pipe_for_position(placement)
    cut = _perpendicular_cut(aperture, placement, pipe, plane)
    if cut is None:
        return None, set()

    left, right, distance, axis_in_plane = cut
    refs = aperture.line.env.aperture.ref
    pipe_ref = refs.pipes[model.pipe_name_for_position(placement)]
    a, b = pipe_ref.positions[left], pipe_ref.positions[right]
    shape_a = refs.profiles[model.profile_names[a.profile_index._value]].shape
    shape_b = refs.profiles[model.profile_names[b.profile_index._value]].shape
    spec = _interpolated_limit(a, b, shape_a, shape_b, distance, axis_in_plane, expressions)
    relevant = expressions.under(str(shape_a) + '.', str(shape_b) + '.',
                                 str(a) + '.', str(b) + '.', str(pipe_ref) + '.curvature')
    return spec, relevant


def _attach_expressions(env, name, parameters):
    ref = env.ref[name]
    for key, value in parameters.items():
        if isinstance(value, BaseRef):
            setattr(ref, key, value)
        elif isinstance(value, (list, tuple)):
            for ii, item in enumerate(value):
                if isinstance(item, BaseRef):
                    getattr(ref, key)[ii] = item
    if 'x_vertices' not in parameters:
        return
    # LimitPolygon's impact-point kernel uses cached normals. Keep these in the
    # expression graph too when the original polygon vertices are variable.
    if not any(isinstance(value, BaseRef) for key in ('x_vertices', 'y_vertices')
               for value in parameters[key]):
        return
    count = len(parameters['x_vertices'])
    area = sum(ref.x_vertices[ii] * ref.y_vertices[(ii + 1) % count]
               - ref.x_vertices[(ii + 1) % count] * ref.y_vertices[ii]
               for ii in range(count))
    orientation = area / (area * area) ** .5
    for ii in range(count):
        jj = (ii + 1) % count
        dx = ref.x_vertices[jj] - ref.x_vertices[ii]
        dy = ref.y_vertices[jj] - ref.y_vertices[ii]
        length = (dx * dx + dy * dy) ** .5
        ref.x_normal[ii] = -orientation * dy / length
        ref.y_normal[ii] = orientation * dx / length


def _sampled_bindings(aperture, candidates, s, expressions):
    """Find bindings frozen by polygon sampling, including both sides of a gap."""
    if not expressions.tasks:
        return set()
    if not candidates:
        bounds = aperture._aperture_bounds
        centres = bounds.s_positions.to_nparray()
        order = np.argsort(centres)
        insertion = np.searchsorted(centres[order], s)
        for neighbour in (insertion - 1, insertion):
            if aperture.is_ring and len(order):
                neighbour %= len(order)
            if 0 <= neighbour < len(order):
                candidates.append(int(bounds.pipe_position_indices[order[neighbour]]))

    model = aperture._model
    refs = aperture.line.env.aperture.ref
    prefixes = []
    for index in candidates:
        placement = model.pipe_positions[index]
        prefixes.append(str(refs.pipes[model.pipe_name_for_position(placement)]) + '.')
        for position in model.pipe_for_position(placement).positions:
            name = model.profile_names[position.profile_index]
            prefixes.append(str(refs.profiles[name].shape) + '.')
    return expressions.under(*prefixes)


def _validated_names(env, names, positions):
    if isinstance(names, str):
        raise TypeError('names must be a collection of element names, not a string.')
    names = list(names)
    if len(names) != len(positions):
        raise ValueError('Provide one element name per s position.')
    if len(set(names)) != len(names) or len(set(positions)) != len(positions):
        raise ValueError('Named limits require unique names and s positions.')
    for name in names:
        if not isinstance(name, str):
            raise TypeError('Element names must be strings.')
        env._check_name_clashes(name)
    return names


def _profile_intervals(aperture):
    """Open intervals between adjacent profile footprints, indexed by placed pipe."""
    bounds = aperture._aperture_bounds
    pipes = bounds.pipe_position_indices.to_nparray()
    indices = bounds.profile_position_indices.to_nparray()
    starts, ends = bounds.s_start.to_nparray(), bounds.s_end.to_nparray()
    period = aperture.line.get_length() if aperture.is_ring else None
    # A tilted profile can cross the survey extent even when its centre plane
    # misses the orbit. Its measured footprint still determines the interval.
    span = ends - starts
    if period:
        span %= period
    centres = starts + span / 2

    # Group by pipe before pairing endpoints. At a joint the adjoining pipe
    # can have a coincident profile, and a pipe may be installed backwards.
    order = np.lexsort((indices, pipes))
    intervals = {}
    for group in np.split(order, np.flatnonzero(np.diff(pipes[order])) + 1):
        if len(group) < 2:
            continue
        a, b = group[:-1], group[1:]
        delta = centres[b] - centres[a]
        if period:
            delta -= np.rint(delta / period) * period
        left, right = np.where(delta >= 0, a, b), np.where(delta >= 0, b, a)
        width = starts[right] - ends[left]
        if period:
            width += np.rint((abs(delta) - width) / period) * period
        intervals[int(pipes[group[0]])] = ends[left], width
    return intervals, period


def _interior_segment(interval, period, s, tol):
    if interval is None:
        return None
    starts, widths = interval
    offset = s - starts
    if period:
        offset %= period
    matches = np.flatnonzero((offset > tol) & (offset < widths - tol))
    return int(matches[0]) if len(matches) == 1 else None


def _profile_to_fit(model, pipe_index, left):
    """Identify circular pipe segments and segments with identical rectellipse ends."""
    pipe = model.pipe_for_position(model.pipe_positions[pipe_index])
    a, b = pipe.positions[left], pipe.positions[left + 1]
    shape = model.profiles[a.profile_index].shape
    if isinstance(shape, Circle) and isinstance(model.profiles[b.profile_index].shape, Circle):
        return int(a.profile_index)
    if a.profile_index == b.profile_index and isinstance(shape, RectEllipse):
        return int(a.profile_index)
    return None


def _fit_circular_cut(vertices):
    """Recover the ellipse of an oblique cylinder or cone from its actual cut."""
    # Centre and scale before solving the conic equation q(x, y) + l(x, y) = 1.
    # On a tapered pipe the sample angles change around the cut.
    origin = vertices.mean(axis=0)
    scale = np.max(np.abs(vertices - origin))
    if scale == 0:
        return None
    x, y = ((vertices - origin) / scale).T
    design = np.column_stack((x**2, 2*x*y, y**2, x, y))
    coeff, _, rank, _ = np.linalg.lstsq(design, np.ones(len(x)), rcond=None)
    xx, xy, yy, lx, ly = coeff
    values, rotation = np.linalg.eigh([[xx, xy], [xy, yy]])
    if rank < 5 or values[0] <= 0:
        return None

    # Complete the square in principal-axis coordinates.
    offset = -(rotation.T @ [lx, ly]) / (2 * values)
    radii = scale * np.sqrt((1 + np.dot(values, offset**2)) / values)
    centre = origin + scale * (rotation @ offset)
    local = (vertices - centre) @ rotation
    unit = local / radii
    lengths = np.linalg.norm(unit, axis=1)
    if np.any(lengths == 0):
        return None
    # Compare each vertex to the ellipse along the ray from the fitted centre.
    error = np.linalg.norm(local, axis=1) * np.abs(1 - 1 / lengths)
    if np.max(error) > _CONTOUR_ATOL:
        return None
    # A folded or partial contour can also have every vertex on a conic.
    angles = np.arctan2(unit[:, 1], unit[:, 0])
    steps = np.diff(np.unwrap(np.r_[angles, angles[0]]))
    if not np.isclose(abs(steps.sum()), 2 * np.pi) or (steps.min() < 0 < steps.max()):
        return None
    return LimitEllipse, dict(a=radii[0], b=radii[1], shift_x=centre[0], shift_y=centre[1],
                              rot_s_rad=np.arctan2(rotation[1, 0], rotation[0, 0]))


def _fit_rectellipse_cut(shape, template, vertices):
    """Fit the transformed source profile and check the displacement at every vertex."""
    xy, inverse = template
    transform = inverse @ vertices
    x_axis, y_axis, centre = transform
    # LimitRectEllipse has perpendicular clipping axes. Keep the first axis,
    # remove shear from the second, then check the resulting contour in metres.
    width = np.linalg.norm(x_axis)
    if width == 0:
        return None
    u = x_axis / width
    v = np.array([-u[1], u[0]])
    height = v @ y_axis
    axes = np.array([x_axis, height * v])
    residual = xy @ axes + centre - vertices
    if height == 0 or np.max(np.linalg.norm(residual, axis=1)) > _CONTOUR_ATOL:
        return None
    return LimitRectEllipse, dict(max_x=shape.max_x * width, max_y=shape.max_y * abs(height),
                                  a=shape.a * width, b=shape.b * abs(height),
                                  shift_x=centre[0], shift_y=centre[1],
                                  rot_s_rad=np.arctan2(u[1], u[0]))


def _section_specs(aperture, positions, prefer_native, keep_bindings):
    """Select unambiguous native sections and remember the first cut to lose each binding."""
    aperture._ensure_geometry()
    placement_names = aperture._model.pipe_position_names
    sources = []
    order = np.argsort(positions, kind='stable')
    planes = aperture.get_poses_at_s(positions[order])[np.argsort(order)]
    expressions = _Expressions(aperture.line.env)
    intervals, period = _profile_intervals(aperture)
    templates = {}
    specs, lost = [], {}
    candidates_at_s = aperture._pipe_candidates_at_s(positions)
    for ii, (s, plane, candidates) in enumerate(zip(positions, planes, candidates_at_s)):
        sources.append(placement_names[candidates[0]] if len(candidates) == 1 else None)
        spec, relevant = None, set()
        if len(candidates) == 1 and prefer_native:
            expressions.used.clear()
            pipe = candidates[0]
            left = _interior_segment(intervals.get(pipe), period, s, aperture.s_tol)
            if left is not None or len(placement_names) == 1:
                spec, relevant = _native_section(aperture, pipe, plane, expressions)
            if spec is None and left is not None:
                profile = _profile_to_fit(aperture._model, pipe, left)
                if profile is not None:
                    templates[ii] = profile
        if spec is None:
            relevant = _sampled_bindings(aperture, candidates, s, expressions)
            untranslated = relevant
        else:
            untranslated = relevant - expressions.used if keep_bindings else relevant
        for target in untranslated:
            lost.setdefault(target, s)
        specs.append(spec)

    return specs, lost, templates, sources


def _sample_missing_sections(aperture, positions, specs, templates):
    fallback = [idx for idx, spec in enumerate(specs) if spec is None]
    if not fallback:
        return
    fallback = sorted(fallback, key=positions.__getitem__)
    sections = aperture.get_cross_sections_at_s(positions[fallback])
    fits = {}
    points = aperture._profile_polygons.points.to_nparray() if templates else None
    for profile in set(templates.values()):
        if not isinstance(aperture._model.profiles[profile].shape, RectEllipse):
            continue
        xy = points[profile, :-1]
        design = np.column_stack((xy, np.ones(len(xy))))
        fits[profile] = xy, np.linalg.pinv(design)
    for ii, row in zip(fallback, sections.rows):
        vertices = row.cross_section[:-1]
        if not np.all(np.isfinite(vertices)):
            raise ValueError(f'No finite aperture cross-section at s={positions[ii]:g}.')
        # Coarse display meshes provide too few points to check a fitted contour.
        if ii in templates and len(vertices) >= 16:
            profile = templates[ii]
            shape = aperture._model.profiles[profile].shape
            if isinstance(shape, Circle):
                specs[ii] = _fit_circular_cut(vertices)
            else:
                specs[ii] = _fit_rectellipse_cut(shape, fits[profile], vertices)
        if specs[ii] is None:
            specs[ii] = LimitPolygon, dict(x_vertices=vertices[:, 0], y_vertices=vertices[:, 1])


def limit_elements_at_s(aperture, s_positions, *, names=None, prefer_native=True):
    """Recognize native cuts first, sample the remaining cuts in one kernel call,
    then instantiate every limit before registering any names or expressions.
    """
    positions = np.asarray(s_positions, dtype=float)
    if positions.ndim != 1 or not np.all(np.isfinite(positions)):
        raise ValueError('s_positions must be a one-dimensional collection of finite positions.')
    if not len(positions):
        if names is not None and len(names):
            raise ValueError('Provide one element name per s position.')
        return {}
    env = aperture.line.env
    if names is not None:
        names = _validated_names(env, names, positions)

    specs, lost, templates, sources = _section_specs(aperture, positions, prefer_native, keep_bindings=names is not None)
    _sample_missing_sections(aperture, positions, specs, templates)
    elements = [cls(installed_pipe_name=source,
                    **{key: _value(value) for key, value in parameters.items()})
                for (cls, parameters), source in zip(specs, sources)]
    if lost:
        detail = '; '.join(f'{target} (s={s:g})' for target, s in sorted(lost.items()))
        reason = ('standalone elements have no expression manager; supply names to preserve supported bindings'
                  if names is None else 'the sampled geometry or parameter has no supported expression translation')
        warn(f'Aperture expressions were not translated to limit elements and were frozen at their '
             f'current values: {detail}. Reason: {reason}. Regenerate limits after changing these parameters.',
             UserWarning, stacklevel=3)
    if names is not None:
        for name, element, (_, parameters) in zip(names, elements, specs):
            env.elements[name] = element
            _attach_expressions(env, name, parameters)
    return dict(zip(positions, elements))
