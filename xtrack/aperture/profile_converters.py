from functools import singledispatch
from typing import List, Optional, Tuple, Union

import numpy as np
from xtrack.aperture.structures import (
    Circle,
    Ellipse,
    Octagon,
    Polygon,
    Racetrack,
    Rectangle,
    RectEllipse,
    ShapeTypes
)
from xtrack.beam_elements import apertures

LimitTypes = Union[
    apertures.LimitRect,
    apertures.LimitEllipse,
    apertures.LimitRectEllipse,
    apertures.LimitRacetrack,
    apertures.LimitPolygon,
]


@singledispatch
def profile_from_limit_element(element: LimitTypes) -> Tuple[ShapeTypes, float, float]:
    """
    Convert a limit beam element to a profile object.

    Parameters
    ----------
    element: LimitTypes
        Element to convert to a profile.
    Returns:
        A tuple consting of the profile type, x offset, and y offset.
    """
    raise NotImplementedError(f"Unsupported element type: {type(element)}")


@profile_from_limit_element.register
def _profile_from_limit_rect(element: apertures.LimitRect) -> Tuple[ShapeTypes, float, float]:
    half_width = (element.max_x - element.min_x) / 2
    half_height = (element.max_y - element.min_y) / 2
    x = (element.min_x + element.max_x) / 2
    y = (element.min_y + element.max_y) / 2
    rectangle = Rectangle(half_width=half_width, half_height=half_height)
    return rectangle, x, y


@profile_from_limit_element.register
def _profile_from_limit_ellipse(element: apertures.LimitEllipse) -> Tuple[ShapeTypes, float, float]:
    rx = element.a
    ry = element.b
    ellipse = Ellipse(half_major=rx, half_minor=ry)
    return ellipse, 0, 0


@profile_from_limit_element.register
def _profile_from_limit_rect_ellipse(element: apertures.LimitRectEllipse) -> Tuple[ShapeTypes, float, float]:
    max_x = element.max_x
    max_y = element.max_y
    rx = element.a
    ry = element.b
    rect_ellipse = RectEllipse(half_width=max_x, half_height=max_y, half_major=rx, half_minor=ry)
    return rect_ellipse, 0, 0


@profile_from_limit_element.register
def _profile_from_limit_racetrack(element: apertures.LimitRacetrack) -> Tuple[ShapeTypes, float, float]:
    half_width = (element.max_x - element.min_x) / 2
    half_height = (element.max_y - element.min_y) / 2
    x = (element.min_x + element.max_x) / 2
    y = (element.min_y + element.max_y) / 2
    rx = element.a
    ry = element.b
    racetrack = Racetrack(
        half_width=half_width,
        half_height=half_height,
        half_major=rx,
        half_minor=ry,
    )
    return racetrack, x, y


@profile_from_limit_element.register
def _profile_from_limit_polygon(element: apertures.LimitPolygon) -> Tuple[ShapeTypes, float, float]:
    xs = element.x_vertices + [element.x_vertices[0]]
    ys = element.y_vertices + [element.y_vertices[0]]
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

    # Clean up params due to MAD-X quirks
    params = params[:allowed_len_params]

    if np.any(np.array(params[allowed_len_params:]) != 0):
        raise ValueError(
            f"Extra non-zero parameters provided for MAD-X aperture shape "
            f"{shape}. Accepted number of params is {allowed_len_params}; "
            f"provided {params}."
        )

    # If all params are zero, we ignore the aperture
    if np.all(params == 0):
        return None

    return converter(*params)


def _profile_from_madx_circle(radius) -> Circle:
    return Circle(radius=radius)


def _profile_from_madx_rectangle(half_width, half_height) -> Rectangle:
    return Rectangle(half_width=half_width, half_height=half_height)


def _profile_from_madx_ellipse(half_major, half_minor) -> Ellipse:
    return Ellipse(half_major=half_major, half_minor=half_minor)


def _profile_from_madx_rectellipse(half_width, half_height, half_major, half_minor) -> RectEllipse:
    return RectEllipse(
        half_width=half_width,
        half_height=half_height,
        half_major=half_major,
        half_minor=half_minor,
    )


def _profile_from_madx_racetrack(half_width, half_height, half_major, half_minor) -> Racetrack:
    return Racetrack(
        half_width=half_width,
        half_height=half_height,
        half_major=half_major,
        half_minor=half_minor,
    )


def _profile_from_madx_octagon(half_width, half_height, angle_0, angle_1) -> Octagon:
    # TODO: Handle inconsistencies coming from angle_1
    x = 0.5 * half_width * (np.tan(angle_0) + 1)
    diag = np.sqrt(2) * x
    return Octagon(half_width=half_width, half_height=half_height, half_diagonal=diag)