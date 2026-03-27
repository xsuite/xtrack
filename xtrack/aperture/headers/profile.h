#ifndef XTRACK_PROFILE_H
#define XTRACK_PROFILE_H

#include <math.h>
#include <stdlib.h>

#include "base.h"
#include "path.h"

void build_polygon_for_profile(float_type *const, const uint32_t, const Profile);

static inline void build_circle_polygon(Point2D *const, const uint32_t, const Circle);
static inline void build_rectangle_polygon(Point2D *const, const uint32_t, const Rectangle);
static inline void build_ellipse_polygon(Point2D *const, const uint32_t, const Ellipse);
static inline void build_rect_ellipse_polygon(Point2D *const, const uint32_t, const RectEllipse);
static inline void build_racetrack_polygon(Point2D *const, const uint32_t, const Racetrack);
static inline void build_octagon_polygon(Point2D *const, const uint32_t, const Octagon);
static inline void build_polygon_polygon(Point2D *const, const uint32_t, const Polygon);

static inline void build_circle_polygon(Point2D *const points, const uint32_t len_points, const Circle circle)
{
    const float_type radius = Circle_get_radius(circle);

    Segment2D segments[1];
    Path2D path = {.segments = segments, .len_segments = 1};
    segments_from_circle(radius, segments);
    poly_get_n_uniform_points(&path, len_points, points);
}


static inline void build_rectangle_polygon(Point2D *const points, const uint32_t len_points, const Rectangle rectangle)
{
    const float_type half_width = Rectangle_get_half_width(rectangle);
    const float_type half_height = Rectangle_get_half_height(rectangle);

    Segment2D segments[4];
    Path2D path = {.segments = segments, .len_segments = 4};
    segments_from_rectangle(half_width, half_height, segments);
    poly_get_n_uniform_points(&path, len_points, points);
}


static inline void build_ellipse_polygon(Point2D *const points, const uint32_t len_points, const Ellipse ellipse)
{
    const float_type half_major = Ellipse_get_half_major(ellipse);
    const float_type half_minor = Ellipse_get_half_minor(ellipse);

    Segment2D segments[1];
    Path2D path = {.segments = segments, .len_segments = 1};
    segments_from_ellipse(half_major, half_minor, segments);
    poly_get_n_uniform_points(&path, len_points, points);
}


static inline void build_rect_ellipse_polygon(Point2D *const points, const uint32_t len_points, const RectEllipse rect_ellipse)
{
    const float_type half_width = RectEllipse_get_half_width(rect_ellipse);
    const float_type half_height = RectEllipse_get_half_height(rect_ellipse);
    const float_type half_major = RectEllipse_get_half_major(rect_ellipse);
    const float_type half_minor = RectEllipse_get_half_minor(rect_ellipse);

    Segment2D segments[8];
    Path2D path = {.segments = segments, .len_segments = 8};
    segments_from_rectellipse(half_width, half_height, half_major, half_minor, segments, &path.len_segments);
    poly_get_n_uniform_points(&path, len_points, points);
}


static inline void build_racetrack_polygon(Point2D *const points, const uint32_t len_points, const Racetrack racetrack)
{
    const float_type half_width = Racetrack_get_half_width(racetrack);
    const float_type half_height = Racetrack_get_half_height(racetrack);
    const float_type half_major = Racetrack_get_half_major(racetrack);
    const float_type half_minor = Racetrack_get_half_minor(racetrack);

    Segment2D segments[8];
    Path2D path = {.segments = segments, .len_segments = 8};
    segments_from_racetrack(half_width, half_height, half_major, half_minor, segments, &path.len_segments);
    poly_get_n_uniform_points(&path, len_points, points);
}


static inline void build_octagon_polygon(Point2D *const points, const uint32_t len_points, const Octagon octagon)
{
    const float_type half_width = Octagon_get_half_width(octagon);
    const float_type half_height = Octagon_get_half_height(octagon);
    const float_type half_diagonal = Octagon_get_half_diagonal(octagon);

    Segment2D segments[8];
    Path2D path = {.segments = segments, .len_segments = 8};
    segments_from_octagon(half_width, half_height, half_diagonal, segments, &path.len_segments);
    poly_get_n_uniform_points(&path, len_points, points);
}


static inline void build_polygon_polygon(Point2D *const points, const uint32_t len_points, const Polygon polygon)
{
    // TODO: Not yet implemented, requires resampling the polygon
    // TODO: When implemented, change the hardcoded convex=1 flags where needed
}


void build_polygon_for_profile(
    float_type *const points,
    const uint32_t len_points,
    const Profile profile
)
{
    /*
        Convert the logical description of a profile to a polygon, and store it in ``aperture_bounds``.

        The polygons created are expected to be:
        (1) anticlockwise, and
        (2) closed.
    */
    const uint64_t profile_type_id = Profile_typeid_shape(profile);

    switch (profile_type_id)
    {
        case Shape_Circle_t:
        {
            const Circle circle = Profile_member_shape(profile);
            build_circle_polygon((Point2D* const) points, len_points, circle);
            break;
        }
        case Shape_Rectangle_t:
        {
            const Rectangle rectangle = Profile_member_shape(profile);
            build_rectangle_polygon((Point2D* const) points, len_points, rectangle);
            break;
        }
        case Shape_Ellipse_t:
        {
            const Ellipse ellipse = Profile_member_shape(profile);
            build_ellipse_polygon((Point2D* const) points, len_points, ellipse);
            break;
        }
        case Shape_RectEllipse_t:
        {
            const RectEllipse rect_ellipse = Profile_member_shape(profile);
            build_rect_ellipse_polygon((Point2D* const) points, len_points, rect_ellipse);
            break;
        }
        case Shape_Racetrack_t:
        {
            const Racetrack racetrack = Profile_member_shape(profile);
            build_racetrack_polygon((Point2D* const) points, len_points, racetrack);
            break;
        }
        case Shape_Octagon_t:
        {
            const Octagon octagon = Profile_member_shape(profile);
            build_octagon_polygon((Point2D* const) points, len_points, octagon);
            break;
        }
        case Shape_Polygon_t:
        {
            const Polygon polygon = Profile_member_shape(profile);
            build_polygon_polygon((Point2D* const) points, len_points, polygon);
            break;
        }
        case Shape_SVGShape_t:
        {
            const SVGShape svg_shape = Profile_member_shape(profile);
            // TODO: Implement
            break;
        }
    }
}

#endif /* XTRACK_PROFILE_H */
