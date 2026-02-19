#include <math.h>
#include <stdlib.h>

#include "base.h"
#include "path.h"


void build_polygon_for_profile(float_type *const, const uint32_t, const Profile);
void polygon_transform_in_type_frame(float_type *const, const uint32_t, const ProfilePosition);
void build_circle_polygon(G2DPoint *const, const uint32_t, const Circle);
void build_rectangle_polygon(G2DPoint *const, const uint32_t, const Rectangle);
void build_ellipse_polygon(G2DPoint *const, const uint32_t, const Ellipse);
void build_rect_ellipse_polygon(G2DPoint *const, const uint32_t, const RectEllipse);
void build_racetrack_polygon(G2DPoint *const, const uint32_t, const Racetrack);
void build_octagon_polygon(G2DPoint *const, const uint32_t, const Octagon);
void build_polygon_polygon(G2DPoint *const, const uint32_t, const Polygon);


void build_profile_polygons(const ApertureModel model, const CrossSections cross_sections)  // TODO: include survey related logic
{
  /*
      Based on the aperture model and cross section location data, generate
     correct polygons, orthogonal to the survey s path.
  */
    const uint32_t num_cross_sections = CrossSections_get_count(cross_sections);

    for (uint32_t idx = 0; idx < num_cross_sections; idx++)
    {
        const uint32_t type_pos_idx = CrossSections_get_type_position_indices(cross_sections, idx);
        const uint32_t profile_pos_idx = CrossSections_get_profile_position_indices(cross_sections, idx);

        const TypePosition type_pos = ApertureModel_getp1_type_positions(model, type_pos_idx);
        const uint32_t type_idx = TypePosition_get_type_index(type_pos);
        const ApertureType aper_type = ApertureModel_getp1_types(model, type_idx);

        const ProfilePosition profile_pos = ApertureType_getp1_positions(aper_type, profile_pos_idx);
        const uint32_t profile_idx = ProfilePosition_get_profile_index(profile_pos);
        const Profile profile = ApertureModel_getp1_profiles(model, profile_idx);

        float_type *const points = CrossSections_getp3_points(cross_sections, idx, 0, 0);
        const uint32_t num_points = CrossSections_get_num_points(cross_sections);

        build_polygon_for_profile(points, num_points, profile);
        polygon_transform_in_type_frame(points, num_points, profile_pos);
    }
}

void build_polygon_for_profile(
    float_type *const points,
    const uint32_t num_points,
    const Profile profile
)
{
    /*
        Convert the logical description of a profile to a polygon, and store it in
        ``cross_sections``.
    */
    const uint64_t profile_type_id = Profile_typeid_shape(profile);

    switch (profile_type_id)
    {
        case Shape_Circle_t:  // LHC
        {
            const Circle circle = Profile_member_shape(profile);
            build_circle_polygon((G2DPoint* const) points, num_points, circle);
            break;
        }
        case Shape_Rectangle_t:
        {
            const Rectangle rectangle = Profile_member_shape(profile);
            build_rectangle_polygon((G2DPoint* const) points, num_points, rectangle);
            break;
        }
        case Shape_Ellipse_t:
        {
            const Ellipse ellipse = Profile_member_shape(profile);
            build_ellipse_polygon((G2DPoint* const) points, num_points, ellipse);
            break;
        }
        case Shape_RectEllipse_t:
        {
            const RectEllipse rect_ellipse = Profile_member_shape(profile);
            build_rect_ellipse_polygon((G2DPoint* const) points, num_points, rect_ellipse);
            break;
        }
        case Shape_Racetrack_t:
        {
            const Racetrack racetrack = Profile_member_shape(profile);
            build_racetrack_polygon((G2DPoint* const) points, num_points, racetrack);
            break;
        }
        case Shape_Octagon_t:
        {
            const Octagon octagon = Profile_member_shape(profile);
            build_octagon_polygon((G2DPoint* const) points, num_points, octagon);
            break;
        }
        case Shape_Polygon_t:
        {
            const Polygon polygon = Profile_member_shape(profile);
            build_polygon_polygon((G2DPoint* const) points, num_points, polygon);
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


void polygon_transform_in_type_frame(
    float_type *const points,
    const uint32_t num_points,
    const ProfilePosition profile_pos
) {
    /*
    Apply the type frame transformation described in ``profile_pos`` to a
    polygon.
    */
    const float_type shift_x = ProfilePosition_get_shift_x(profile_pos);
    const float_type shift_y = ProfilePosition_get_shift_y(profile_pos);

    for (uint32_t i = 0; i < num_points; i++)
    {
        ((G2DPoint* const) points)[i].x += shift_x;
        ((G2DPoint* const) points)[i].y += shift_y;
        // TODO: Apply rotations, will change s (a heuristic for when a profile
        // generates 1 or 2 cross sections needed?)
        // TODO: Also, how will we select if this is the entry or exit in case of 2
        // cross sections?
    }
}


void build_circle_polygon(G2DPoint *const points, const uint32_t num_points, const Circle circle)
{
    const float_type radius = Circle_get_radius(circle);

    G2DSegment segments[1];
    G2DPath path = {.segments = segments, .len_segments = 1};
    geom2d_segments_from_circle(radius, segments);
    geom2d_poly_get_n_uniform_points(&path, num_points, points);
}


void build_rectangle_polygon(G2DPoint *const points, const uint32_t num_points, const Rectangle rectangle)
{
    const float_type half_width = Rectangle_get_half_width(rectangle);
    const float_type half_height = Rectangle_get_half_height(rectangle);

    G2DSegment segments[4];
    G2DPath path = {.segments = segments, .len_segments = 4};
    geom2d_segments_from_rectangle(half_width, half_height, segments);
    geom2d_poly_get_n_uniform_points(&path, num_points, points);
}


void build_ellipse_polygon(G2DPoint *const points, const uint32_t num_points, const Ellipse ellipse)
{
    const float_type half_major = Ellipse_get_half_major(ellipse);
    const float_type half_minor = Ellipse_get_half_minor(ellipse);

    G2DSegment segments[1];
    G2DPath path = {.segments = segments, .len_segments = 1};
    geom2d_segments_from_ellipse(half_major, half_minor, segments);
    geom2d_poly_get_n_uniform_points(&path, num_points, points);
}


void build_rect_ellipse_polygon(G2DPoint *const points, const uint32_t num_points, const RectEllipse rect_ellipse)
{
    const float_type half_width = RectEllipse_get_half_width(rect_ellipse);
    const float_type half_height = RectEllipse_get_half_height(rect_ellipse);
    const float_type half_major = RectEllipse_get_half_major(rect_ellipse);
    const float_type half_minor = RectEllipse_get_half_minor(rect_ellipse);

    G2DSegment segments[8];
    G2DPath path = {.segments = segments, .len_segments = 8};
    geom2d_segments_from_rectellipse(half_width, half_height, half_major, half_minor, segments, &path.len_segments);
    geom2d_poly_get_n_uniform_points(&path, num_points, points);
}


void build_racetrack_polygon(G2DPoint *const points, const uint32_t num_points, const Racetrack racetrack)
{
    const float_type half_width = Racetrack_get_half_width(racetrack);
    const float_type half_height = Racetrack_get_half_height(racetrack);
    const float_type half_major = Racetrack_get_half_major(racetrack);
    const float_type half_minor = Racetrack_get_half_minor(racetrack);

    G2DSegment segments[8];
    G2DPath path = {.segments = segments, .len_segments = 8};
    geom2d_segments_from_racetrack(half_width, half_height, half_major, half_minor, segments, &path.len_segments);
    geom2d_poly_get_n_uniform_points(&path, num_points, points);
}


void build_octagon_polygon(G2DPoint *const points, const uint32_t num_points, const Octagon octagon)
{
    const float_type half_width = Octagon_get_half_width(octagon);
    const float_type half_height = Octagon_get_half_height(octagon);
    const float_type half_diagonal = Octagon_get_half_diagonal(octagon);

    G2DSegment segments[8];
    G2DPath path = {.segments = segments, .len_segments = 8};
    geom2d_segments_from_octagon(half_width, half_height, half_diagonal, segments, &path.len_segments);
    geom2d_poly_get_n_uniform_points(&path, num_points, points);
}


void build_polygon_polygon(G2DPoint *const points, const uint32_t num_points, const Polygon polygon)
{
    // TODO: Not yet implemented, requires resampling the polygon
}
