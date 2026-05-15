#ifndef XTRACK_APERTURE_POLYGON_H
#define XTRACK_APERTURE_POLYGON_H

#include <stdint.h>

#include "base.h"
#include "polygon_algs.h"

static inline void points_from_polygon(const Polygon polygon, Point2D* const out_points)
{
    const int len_points = Polygon_len_vertices(polygon) / 2;
    for (int i = 0; i < len_points; i++)
    {
        out_points[i] = (Point2D){
            .x = Polygon_get_vertices(polygon, i, 0),
            .y = Polygon_get_vertices(polygon, i, 1),
        };
    }
}


char is_point_inside_polygon(const Polygon polygon, const float_type* point)
{
    const int len_vertices = Polygon_len_vertices(polygon) / 2;
    if (len_vertices <= 0)
        return 0;

    Point2D polygon_vertices[len_vertices];
    points_from_polygon(polygon, polygon_vertices);
    return is_point_inside_polygon_points((const Point2D*) point, polygon_vertices, len_vertices);
}


char points_inside_polygon(const Polygon polygon, const float_type* points, const uint32_t len_points)
{
    const int len_poly_points = Polygon_len_vertices(polygon) / 2;
    if (len_poly_points <= 0)
        return 0;

    Point2D polygon_vertices[len_poly_points];
    points_from_polygon(polygon, polygon_vertices);
    return points_inside_polygon_points((const Point2D*) points, polygon_vertices, len_points, len_poly_points);
}

#endif /* XTRACK_APERTURE_POLYGON_H */
