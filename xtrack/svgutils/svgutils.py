import numpy as np
from .parser import parse_path
from .path import Line, Move, Close


def svg_to_points(svgpath, scale=0.001, curved_steps=10, line_steps=2):
    """
    Convert an svg path to a set of points

    Parameters:
    -----------
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
