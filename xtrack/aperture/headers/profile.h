#ifndef XTRACK_PROFILE_H
#define XTRACK_PROFILE_H

#include <stdint.h>

#include "path.h"
#include "polygon_algs.h"


static inline void build_circle_polygon(float_type *const out_points, const uint32_t len_points, const float_type radius)
{
    Point2D *const points = (Point2D*) out_points;

    Segment2D segments[1];
    Path2D path = {.segments = segments, .len_segments = 1};
    segments_from_circle(radius, segments);
    poly_get_n_uniform_points(&path, len_points, points);
}


static inline void build_rectangle_polygon(float_type *const out_points, const uint32_t len_points, const float_type max_x, const float_type max_y)
{
    Point2D *const points = (Point2D*) out_points;

    Segment2D segments[4];
    Path2D path = {.segments = segments, .len_segments = 4};
    segments_from_rectangle(max_x, max_y, segments);
    poly_get_n_uniform_points(&path, len_points, points);
}


static inline void build_ellipse_polygon(float_type *const out_points, const uint32_t len_points, const float_type a, const float_type b)
{
    Point2D *const points = (Point2D*) out_points;

    Segment2D segments[1];
    Path2D path = {.segments = segments, .len_segments = 1};
    segments_from_ellipse(a, b, segments);
    poly_get_n_uniform_points(&path, len_points, points);
}


static inline void build_rect_ellipse_polygon(float_type *const out_points, const uint32_t len_points, const float_type max_x, const float_type max_y, const float_type a, const float_type b)
{
    Point2D *const points = (Point2D*) out_points;

    Segment2D segments[8];
    Path2D path = {.segments = segments, .len_segments = 8};
    segments_from_rectellipse(max_x, max_y, a, b, segments, &path.len_segments);
    poly_get_n_uniform_points(&path, len_points, points);
}


static inline void build_racetrack_polygon(float_type *const out_points, const uint32_t len_points, const float_type max_x, const float_type max_y, const float_type a, const float_type b)
{
    Point2D *const points = (Point2D*) out_points;

    Segment2D segments[8];
    Path2D path = {.segments = segments, .len_segments = 8};
    segments_from_racetrack(max_x, max_y, a, b, segments, &path.len_segments);
    poly_get_n_uniform_points(&path, len_points, points);
}


static inline void build_octagon_polygon(float_type *const out_points, const uint32_t len_points, const float_type max_x, const float_type max_y, const float_type max_diagonal)
{
    Point2D *const points = (Point2D*) out_points;

    Segment2D segments[8];
    Path2D path = {.segments = segments, .len_segments = 8};
    segments_from_octagon(max_x, max_y, max_diagonal, segments, &path.len_segments);
    poly_get_n_uniform_points(&path, len_points, points);
}


static inline int8_t build_polygon_polygon(
    float_type *const out_points,
    const uint32_t len_points,
    const float_type *const vertices,
    const uint32_t len_vertices
)
{
    Point2D *const points = (Point2D*) out_points;
    const Point2D *const polygon_vertices = (const Point2D*) vertices;
    if (len_vertices <= 0)
        return 0;

    const int is_closed = len_vertices > 1
        && point2d_distance(
            polygon_vertices[0].x, polygon_vertices[0].y,
            polygon_vertices[len_vertices - 1].x, polygon_vertices[len_vertices - 1].y
        ) <= APER_PRECISION;
    const int num_unique_vertices = is_closed ? len_vertices - 1 : len_vertices;
    if (num_unique_vertices < 2)
        return 0;

    Point2D ordered_vertices[num_unique_vertices];
    Segment2D segments[num_unique_vertices];
    const float_type signed_area = polygon_signed_area(polygon_vertices, len_vertices);
    for (int i = 0; i < num_unique_vertices; i++)
    {
        const int source_idx = signed_area >= 0 ? i : (num_unique_vertices - 1 - i);
        ordered_vertices[i] = polygon_vertices[source_idx];
    }

    Path2D path = {.segments = segments, .len_segments = num_unique_vertices};
    segments_from_polygon(ordered_vertices, num_unique_vertices, segments, &path.len_segments);
    if (path.len_segments == 0)
        return 0;
    poly_get_n_uniform_points(&path, len_points, points);
    return 1;
}

#endif /* XTRACK_PROFILE_H */
