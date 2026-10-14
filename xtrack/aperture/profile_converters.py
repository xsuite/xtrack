from functools import singledispatch
from typing import List, Optional, Tuple, Union

import numpy as np

from xtrack.aperture.structures import (
    Circle, Ellipse, Octagon, Polygon, Racetrack, Rectangle, RectEllipse,
    ShapeTypes
)
from xtrack.beam_elements.limit_ellipse import LimitEllipse
from xtrack.beam_elements.limit_polygon import LimitPolygon
from xtrack.beam_elements.limit_racetrack import LimitRacetrack
from xtrack.beam_elements.limit_rect import LimitRect
from xtrack.beam_elements.limit_rect_ellipse import LimitRectEllipse

LimitElement = Union[
    LimitRect,
    LimitEllipse,
    LimitRectEllipse,
    LimitRacetrack,
    LimitPolygon,
]


@singledispatch
def profile_from_limit_element(element: LimitElement) -> Tuple[ShapeTypes, float, float]:
    """
    Convert a limit beam element to a profile object.

    Parameters
    ----------
    element : LimitElement
        Element to convert to a profile.

    Returns
    -------
    Tuple[ShapeTypes, float, float]
        A tuple consisting of the profile type, x offset, and y offset.
    """
    raise NotImplementedError(f"Unsupported element type: {type(element)}")


@profile_from_limit_element.register
def _profile_from_limit_rect(element: LimitRect) -> Tuple[ShapeTypes, float, float]:
    max_x = (element.max_x - element.min_x) / 2
    max_y = (element.max_y - element.min_y) / 2
    x = (element.min_x + element.max_x) / 2
    y = (element.min_y + element.max_y) / 2
    rectangle = Rectangle(max_x=max_x, max_y=max_y)
    return rectangle, x, y


@profile_from_limit_element.register
def _profile_from_limit_ellipse(element: LimitEllipse) -> Tuple[ShapeTypes, float, float]:
    rx = element.a
    ry = element.b
    ellipse = Ellipse(a=rx, b=ry)
    return ellipse, 0, 0


@profile_from_limit_element.register
def _profile_from_limit_rect_ellipse(element: LimitRectEllipse) -> Tuple[ShapeTypes, float, float]:
    max_x = element.max_x
    max_y = element.max_y
    rx = element.a
    ry = element.b
    rect_ellipse = RectEllipse(max_x=max_x, max_y=max_y, a=rx, b=ry)
    return rect_ellipse, 0, 0


@profile_from_limit_element.register
def _profile_from_limit_racetrack(element: LimitRacetrack) -> Tuple[ShapeTypes, float, float]:
    max_x = (element.max_x - element.min_x) / 2
    max_y = (element.max_y - element.min_y) / 2
    x = (element.min_x + element.max_x) / 2
    y = (element.min_y + element.max_y) / 2
    rx = element.a
    ry = element.b
    racetrack = Racetrack(
        max_x=max_x,
        max_y=max_y,
        a=rx,
        b=ry,
    )
    return racetrack, x, y


@profile_from_limit_element.register
def _profile_from_limit_polygon(element: LimitPolygon) -> Tuple[ShapeTypes, float, float]:
    xs = np.asarray(element.x_closed)
    ys = np.asarray(element.y_closed)
    polygon = Polygon(vertices=np.column_stack([xs, ys]))
    return polygon, 0, 0


def profile_from_madx_aperture(shape: str, params: List[float]) -> Optional[ShapeTypes]:
    converter, allowed_len_params = {
        'circle': (_profile_from_madx_circle, 1),
        'rectangle': (_profile_from_madx_rectangle, 2),
        'ellipse': (_profile_from_madx_ellipse, 2),
        'rectellipse': (_profile_from_madx_rectellipse, 4),
        'racetrack': (_profile_from_madx_racetrack, 4),
        'octagon': (_profile_from_madx_octagon, 4),
    }[shape]

    if np.any(np.asarray(params[allowed_len_params:]) != 0):
        raise ValueError(
            f"Extra non-zero parameters provided for MAD-X aperture shape "
            f"{shape}. Accepted number of params is {allowed_len_params}; "
            f"provided {params}."
        )

    # MAD-X may pad the parameter list with zeros.
    params = np.asarray(params[:allowed_len_params])

    # If all params are zero, we ignore the aperture
    if np.all(params == 0):
        return None

    return converter(*params)


def _profile_from_madx_circle(radius) -> Circle:
    return Circle(radius=radius)


def _profile_from_madx_rectangle(max_x, max_y) -> Rectangle:
    return Rectangle(max_x=max_x, max_y=max_y)


def _profile_from_madx_ellipse(a, b) -> Ellipse:
    return Ellipse(a=a, b=b)


def _profile_from_madx_rectellipse(max_x, max_y, a, b) -> RectEllipse:
    return RectEllipse(
        max_x=max_x,
        max_y=max_y,
        a=a,
        b=b,
    )


def _profile_from_madx_racetrack(max_x, max_y, a, b) -> Racetrack:
    return Racetrack(
        max_x=max_x,
        max_y=max_y,
        a=a,
        b=b,
    )


def _profile_from_madx_octagon(max_x, max_y, angle_0, angle_1) -> Octagon:
    # TODO: Handle inconsistencies coming from angle_1
    x = 0.5 * max_x * (np.tan(angle_0) + 1)
    diag = np.sqrt(2) * x
    return Octagon(max_x=max_x, max_y=max_y, max_diagonal=diag)
