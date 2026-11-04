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


def _value(value):
    if isinstance(value, BaseRef):
        return value._value
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    return value


def _inverse_pose(pose):
    out = np.eye(4)
    out[:3, :3] = pose[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ pose[:3, 3]
    return out


def _same_field(a, b):
    return (np.array_equal(a._value, b._value)
            and str(a._expr) == str(b._expr))


class _Expressions:
    def __init__(self, env):
        self.env = env
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


def _native_section(aperture, pipe_index, plane, expressions):
    """Recognize sections perpendicular to a pipe, without fitting sampled polygons.

    Equal profiles retain their shape under translation along the pipe. Circles
    also retain their shape when their radii are interpolated. General tapered,
    twisted or oblique sections are left to the polygon intersection algorithm.
    """
    env_aperture = aperture.line.env.aperture
    placement = aperture._model.pipe_positions[pipe_index]
    pipe = aperture._model.pipe_for_position(placement)
    pipe_name = aperture._model.pipe_name_for_position(placement)
    pipe_ref = env_aperture.ref.pipes[pipe_name]
    profile_indices = {int(position.profile_index) for position in pipe.positions}
    shape_refs = {ii: env_aperture.ref.profiles[aperture._model.profile_names[ii]].shape
                  for ii in profile_indices}
    relevant = expressions.under(str(pipe_ref) + '.',
                                 *(str(ref) + '.' for ref in shape_refs.values()))
    if not len(pipe.positions):
        return None, relevant

    survey_pose = aperture._survey_data.pose.to_nparray()[placement.survey_index]
    pipe_pose = survey_pose @ placement.transformation.to_nparray()
    plane_in_pipe = _inverse_pose(pipe_pose) @ plane
    shifts = np.array([position.shift_s for position in pipe.positions])
    if np.any(np.diff(shifts) <= 0):
        return None, relevant
    curvature = float(pipe.curvature)
    if len(shifts) == 1:
        # A lone profile is projected from its own plane; it does not define
        # an extrusion following the pipe curvature.
        q = shifts[0]
    elif abs(curvature) < 1e-14:
        q = plane_in_pipe[2, 3]
    else:
        sign = np.sign(curvature)
        q = np.arctan2(sign * plane_in_pipe[2, 3],
                       sign * (plane_in_pipe[0, 3] + 1 / curvature)) / curvature
        period = 2 * np.pi / abs(curvature)
        q += np.round((np.mean(shifts) - q) / period) * period
    axis_pose = pipe_pose @ arc_matrix(q, q * curvature, 0.)
    axis_in_plane = _inverse_pose(plane) @ axis_pose
    # The cutting plane must be perpendicular to the local pipe axis.
    if (not np.allclose(axis_in_plane[:2, 2], 0, atol=_GEOMETRY_ATOL, rtol=0)
            or not np.allclose(axis_in_plane[2, :2], 0, atol=_GEOMETRY_ATOL, rtol=0)
            or (aperture._aperture_bounds.count > 1
                and abs(axis_in_plane[2, 3]) > _GEOMETRY_ATOL)):
        return None, relevant
    if len(shifts) > 1 and not shifts[0] - _GEOMETRY_ATOL <= q <= shifts[-1] + _GEOMETRY_ATOL:
        return None, relevant
    if len(shifts) == 1:
        left = right = 0
    else:
        left = int(np.clip(np.searchsorted(shifts, q, side='right') - 1, 0, len(shifts) - 2))
        right = left + 1
    first, second = pipe.positions[left], pipe.positions[right]
    a, b = pipe_ref.positions[left], pipe_ref.positions[right]
    shape_a, shape_b = shape_refs[int(first.profile_index)], shape_refs[int(second.profile_index)]
    relevant = expressions.under(str(shape_a) + '.', str(shape_b) + '.',
                                 str(a) + '.', str(b) + '.')
    relevant |= expressions.under(str(pipe_ref) + '.curvature')
    if any(abs(getattr(position, field)) > _GEOMETRY_ATOL
           for position in (first, second) for field in ('rot_x_rad', 'rot_y_rad')):
        return None, relevant
    if not _same_field(a.rot_s_rad, b.rot_s_rad):
        return None, relevant
    same_shape = _same_shape(shape_a, shape_b, expressions)
    circles = isinstance(shape_a._value, Circle) and isinstance(shape_b._value, Circle)
    if not same_shape and not circles:
        return None, relevant

    rotation = axis_in_plane[:2, :2]
    reflection = 1 if np.linalg.det(rotation) > 0 else -1
    spec = _shape_limit(shape_a._value, shape_a, expressions, reflection)
    if spec is None:
        return None, relevant
    limit_class, parameters = spec
    if same_shape:
        expressions.mirror(shape_a, shape_b)

    needs_weight = (not same_shape or not _same_field(a.shift_x, b.shift_x)
                    or not _same_field(a.shift_y, b.shift_y))
    weight = 0.
    if needs_weight and left != right:
        start, end = expressions.read(a.shift_s), expressions.read(b.shift_s)
        weight = (q - start) / (end - start)

    def blend(first_ref, second_ref):
        first_value, second_value = expressions.read(first_ref), expressions.read(second_ref)
        if _same_field(first_ref, second_ref):
            return first_value
        return (1 - weight) * first_value + weight * second_value

    if not same_shape:
        radius = blend(shape_a.radius, shape_b.radius)
        parameters.update(a=radius, b=radius)
    x, y = blend(a.shift_x, b.shift_x), blend(a.shift_y, b.shift_y)
    roll = expressions.read(a.rot_s_rad)
    expressions.read(b.rot_s_rad)
    parameters.update(
        shift_x=axis_in_plane[0, 3] + rotation[0, 0] * x + rotation[0, 1] * y,
        shift_y=axis_in_plane[1, 3] + rotation[1, 0] * x + rotation[1, 1] * y,
        rot_s_rad=np.arctan2(rotation[1, 0], rotation[0, 0]) + reflection * roll)
    return (limit_class, parameters), relevant


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


def limit_elements_at_s(aperture, s_positions, *, names=None, prefer_native=True):
    positions = np.asarray(s_positions, dtype=float)
    if positions.ndim != 1 or not np.all(np.isfinite(positions)):
        raise ValueError('s_positions must be a one-dimensional collection of finite positions.')
    if not len(positions):
        if names is not None and len(names):
            raise ValueError('Provide one element name per s position.')
        return {}
    env = aperture.line.env
    if names is not None:
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

    aperture._ensure_geometry()
    pipe_table = aperture.get_pipe_table()
    # The table is sorted by s, independently of the model's insertion order.
    model_indices = {name: ii for ii, name in enumerate(aperture._model.pipe_position_names)}
    pipe_indices = np.array([model_indices[name] for name in pipe_table.name], dtype=int)
    starts, ends = pipe_table.s_span_start, pipe_table.s_span_end
    planes = aperture.get_poses_at_s(positions)
    expressions = _Expressions(env)
    specs, lost = [], {}
    fallback = []
    for ii, (s, plane) in enumerate(zip(positions, planes)):
        inside = (starts - _GEOMETRY_ATOL <= s) & (s <= ends + _GEOMETRY_ATOL)
        if aperture.is_ring:
            inside |= ((starts > ends)
                       & ((s >= starts - _GEOMETRY_ATOL) | (s <= ends + _GEOMETRY_ATOL)))
        candidates = list(pipe_indices[inside])
        if aperture._aperture_bounds.count == 1:
            candidates = [int(aperture._aperture_bounds.pipe_position_indices[0])]
        spec, relevant = None, set()
        if len(candidates) == 1 and prefer_native:
            expressions.used.clear()
            spec, relevant = _native_section(aperture, candidates[0], plane, expressions)
        if spec is None:
            fallback.append(ii)
            # Include neighbours for sections bridging pipes or lying in gaps.
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
            for jj in candidates:
                placement = aperture._model.pipe_positions[jj]
                pipe = aperture._model.pipe_for_position(placement)
                pipe_name = aperture._model.pipe_name_for_position(placement)
                relevant |= expressions.under(str(env.aperture.ref.pipes[pipe_name]) + '.')
                for position in pipe.positions:
                    name = aperture._model.profile_names[int(position.profile_index)]
                    relevant |= expressions.under(str(env.aperture.ref.profiles[name].shape) + '.')
            untranslated = relevant
        else:
            untranslated = relevant - expressions.used if names is not None else relevant
        for target in untranslated:
            lost.setdefault(target, s)
        specs.append(spec)

    if fallback:
        sections = aperture.get_cross_sections_at_s(positions[fallback])
        for ii, row in zip(fallback, sections.rows):
            vertices = row.cross_section[:-1]
            if not np.all(np.isfinite(vertices)):
                raise ValueError(f'No finite aperture cross-section at s={positions[ii]:g}.')
            specs[ii] = LimitPolygon, dict(x_vertices=vertices[:, 0], y_vertices=vertices[:, 1])

    elements = [cls(**{key: _value(value) for key, value in parameters.items()})
                for cls, parameters in specs]
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
