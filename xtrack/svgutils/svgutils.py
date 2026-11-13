import numpy as np

from .parser import _tokenize_path, parse_path
from .path import Line, Move, Close


def svg_to_points(svgpath, scale=0.001, curved_steps=10, line_steps=2):
    """
    Convert an svg path to a set of points

    Parameters
    ----------
    svgpath : str
        svg path string describing the aperture
    scale: float
        scale factor (default: 0.001), by default conversion from [mm] to [m]
    curved_steps : int
        number of steps for curved segments (default: 10)
    line_steps : int
         number of steps for line segments (default: 2)

    """

    if line_steps < 2:
        raise ValueError("line_steps must be greater than or equal to 2")
    if curved_steps < 2:
        raise ValueError("curved_steps must be greater than or equal to 2")

    curve = parse_path(svgpath)
    points = []
    for seg in curve:
        name=seg.__class__.__name__
        if name in ["Line","Close"]:
            for ii in range(0, line_steps - 1):
                points.append(seg.point(ii / line_steps))
        elif name=="Move":
            pass
        else:
            for ii in range(0, curved_steps - 1):
                points.append(seg.point(ii / curved_steps))
    if abs(seg.end-points[0])>1e-10:
      points.append(seg.end)


    points = np.array(points) * scale
    return points.real, -points.imag


def svg_to_points_by_resolution(
    svgpath: str,
    resolution: float,
    scale: float = 0.001,
) -> np.ndarray:
    """Approximate a single closed SVG path by a polygon in physical metres.

    Straight segments contribute endpoints only. Curved segments use
    ``max(2, ceil(segment.length() * scale / resolution))`` intervals uniform
    in their parameter. Thus ``resolution`` is nominal average spacing,
    not a maximum spacing or geometric error bound. SVG y is inverted.

    Return a float64 array of shape ``(n, 2)``, including exact closure.
    Orientation is preserved; use the aperture Polygon builder for CCW,
    uniform arc-length resampling of this approximation.
    """
    if not isinstance(svgpath, str) or not svgpath.strip():
        raise ValueError('svgpath must be a non-empty SVG path string.')

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be finite and positive.')

    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('resolution must be finite and positive.')

    # Validate numeric input before constructing arcs or integrating lengths.
    for token in _tokenize_path(svgpath):
        if any(value is None or not np.isfinite(value) for value in token[1:]):
            raise ValueError('SVG path data must be finite.')

    parsed_path = parse_path(svgpath)
    subpath_count = sum(isinstance(segment, Move) for segment in parsed_path)

    if subpath_count != 1:
        raise ValueError('SVG path must contain exactly one subpath.')

    boundary_segments = [
        segment for segment in parsed_path if not isinstance(segment, Move)
    ]

    if not boundary_segments:
        raise ValueError('SVG path must have a non-degenerate boundary.')

    if not all(
        np.isfinite(segment.start) and np.isfinite(segment.end)
        for segment in boundary_segments
    ):
        raise ValueError('SVG path data must be finite.')

    path_start = boundary_segments[0].start
    path_end = boundary_segments[-1].end

    if not np.isclose(path_end, path_start, rtol=0, atol=1e-10):
        raise ValueError('SVG path must be closed.')

    # Sample each segment in SVG coordinates, keeping shared endpoints once.
    sampled_points = []

    for segment in boundary_segments:
        segment_length_m = segment.length() * scale

        if not np.isfinite(segment_length_m):
            raise ValueError('SVG segment length must be finite.')

        if segment_length_m == 0:
            continue

        if isinstance(segment, (Line, Close)):
            segment_points = [segment.start, segment.end]
        else:
            num_intervals = max(2, int(np.ceil(segment_length_m / resolution)))
            parameter_values = np.linspace(0, 1, num_intervals + 1)
            segment_points = [segment.point(t) for t in parameter_values]

            # Arc.point() can round its endpoints. Use the exact path endpoints
            # so adjacent segments share the same vertex.
            segment_points[0] = segment.start
            segment_points[-1] = segment.end

        if sampled_points and sampled_points[-1] == segment_points[0]:
            sampled_points.extend(segment_points[1:])
        else:
            sampled_points.extend(segment_points)

    if not sampled_points:
        raise ValueError('SVG path must have a non-degenerate boundary.')

    # Convert to metres and change SVG's downward y axis to physical y-up.
    physical_points = np.asarray(sampled_points, dtype=complex) * scale
    vertices = np.column_stack((physical_points.real, -physical_points.imag))

    if not np.isfinite(vertices).all():
        raise ValueError('SVG polygon coordinates must be finite.')

    # Close exactly, then remove consecutive duplicate vertices.
    vertices[-1] = vertices[0]

    different_from_previous = np.any(np.diff(vertices, axis=0) != 0, axis=1)
    keep_vertex = np.concatenate(([True], different_from_previous))
    vertices = vertices[keep_vertex]

    if len(vertices) < 4:
        raise ValueError('SVG polygon must have at least three distinct vertices.')

    # Translate before computing area to avoid cancellation for small apertures
    # far from the origin. Preserve the input orientation.
    relative_vertices = vertices - vertices[0]
    edge_starts = relative_vertices[:-1]
    edge_ends = relative_vertices[1:]

    twice_signed_area = np.sum(
        edge_starts[:, 0] * edge_ends[:, 1]
        - edge_ends[:, 0] * edge_starts[:, 1]
    )

    if twice_signed_area == 0:
        raise ValueError('SVG polygon must have nonzero area.')

    return vertices
