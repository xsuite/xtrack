#include <math.h>
#include <stdlib.h>

#include "base.h"
#include "path.h"
#include "survey_tools.h"


typedef struct {
    int32_t index;
    int32_t offset;
    int32_t start;
    int32_t lower_bound;
    int32_t upper_bound;
} ZigZagIterator;


ZigZagIterator zigzag_iterator_new(uint32_t start, uint32_t lower_bound, uint32_t upper_bound);
uint8_t zigzag_iterator_next(ZigZagIterator* iter);

void build_polygon_for_profile(float_type *const, const uint32_t, const Profile);
void build_circle_polygon(G2DPoint *const, const uint32_t, const Circle);
void build_rectangle_polygon(G2DPoint *const, const uint32_t, const Rectangle);
void build_ellipse_polygon(G2DPoint *const, const uint32_t, const Ellipse);
void build_rect_ellipse_polygon(G2DPoint *const, const uint32_t, const RectEllipse);
void build_racetrack_polygon(G2DPoint *const, const uint32_t, const Racetrack);
void build_octagon_polygon(G2DPoint *const, const uint32_t, const Octagon);
void build_polygon_polygon(G2DPoint *const, const uint32_t, const Polygon);


ZigZagIterator zigzag_iterator_new(uint32_t start, uint32_t lower_bound, uint32_t upper_bound)
/*
    Return a new zigzag iterator: already pointing to `start`.

    The iterator's `.index` property points to the current index, and iterates within `[lower_bound, upper_bound)`.
*/
{
    return (ZigZagIterator) {
        .index = start,
        .offset = 0,
        .start = start,
        .lower_bound = lower_bound,
        .upper_bound = upper_bound
    };
}


uint8_t zigzag_iterator_next(ZigZagIterator* iter)
/*
    Advance the zigzag iterator to the next index.

    The iterator alternates outward: start, start+1, start-1, start+2, start-2, ...
    When one side reaches a bound, iteration continues only on the remaining side
    until all indices in [lower_bound, upper_bound) are exhausted.

    Returns
    -------
      1 if the iterator was successfully advanced and `iter->index` is valid.
      0 if the iterator is exhausted, and cannot be advanced anymore.
 */
{
    int32_t prev_index = iter->index;

    if (iter->offset == 0) {
        /* Initial condition */
        iter->offset++;
        iter->index++;
    }
    else if (iter->offset > 0) {
        /* Positive side -> negative side */
        iter->index -= 2 * iter->offset;

        if (iter->index < iter->lower_bound) {
            /* Hit the lower bound, continue the positive side */
            iter->index = prev_index + 1;
            iter->offset++;
        }
        else iter->offset *= -1;
    }
    else if (iter->offset < 0) {
        /* Negative side -> positive side + 1 */
        iter->index += -2 * iter->offset + 1;
        if (iter->index >= iter->upper_bound) {
            /* Hit the upper bound, continue the negative side */
            iter->index = prev_index - 1;
            iter->offset--;
        }
        else iter->offset = 1 - iter->offset;
    }

    /* If still not in the bounds, means we've exhausted the iterator */
    if (iter->index < iter->lower_bound || iter->upper_bound <= iter->index) return 0;
    else return 1;
}


static inline float_type survey_s_for_aperture(
    const TypePosition type_pos,
    const ProfilePosition profile_pos,
    const SurveyData survey,
    uint32_t* found_survey_index
)
/*
    Get the survey `s` at which the profile is installed.

    This is found by obtaining the intersection point of the plane on which a profile sits
    and the survey. This will not be accurate when the survey does not actually pass through
    the profile polygon, however in such an unlikely scenario we can clip to bounds.
*/
{
    const float_type eps = APER_PRECISION;
    const uint32_t num_survey_entries = SurveyData_len_s(survey);

    // Transformation from plane (s = 0) -> type frame
    Transform profile_transform = {
        .x = ProfilePosition_get_shift_x(profile_pos),
        .y = ProfilePosition_get_shift_y(profile_pos),
        .s = ProfilePosition_get_s_position(profile_pos),
        .rot_x = ProfilePosition_get_rot_x(profile_pos),
        .rot_y = ProfilePosition_get_rot_y(profile_pos),
        .rot_s = ProfilePosition_get_rot_s(profile_pos),
    };
    Pose plane_in_type = transform_to_matrix(profile_transform);

    // Transformation from type frame -> survey reference point frame
    Pose type_in_survey_ref;
    for (uint8_t i = 0; i < 4; i++)
        for (uint8_t j = 0; j < 4; j++)
            type_in_survey_ref.mat[i][j] = TypePosition_get_transformation(type_pos, i, j);

    // Transformation from survey reference point -> world frame
    const uint32_t survey_idx = TypePosition_get_survey_index(type_pos);
    Pose survey_ref_in_world = pose_matrix_from_survey(survey, survey_idx);

    // Compute survey_ref_in_world @ type_in_survey_ref @ plane_in_type
    Pose plane_in_survey_ref;
    matrix_multiply_4x4(type_in_survey_ref.mat, plane_in_type.mat, plane_in_survey_ref.mat);
    Pose plane_in_world;
    matrix_multiply_4x4(survey_ref_in_world.mat, plane_in_survey_ref.mat, plane_in_world.mat);

    /*
        For data from MAD-X etc. it's likely that it's the type's reference survey point where the profile
        intersects the survey (the true installed s), however, that is not necessarily true. We find the
        s by testing that survey "segment" first, and working our way outwards in both directions.
    */
    float_type found_s = NAN;
    ZigZagIterator it = zigzag_iterator_new(survey_idx, 0, num_survey_entries - 1);
    do
    {
        LineSegment3D segment = survey_segment(survey, it.index);
        const float_type t = dist_along_segment_where_plane_intersects(segment, plane_in_world);
        const float_type type_s = SurveyData_get_s(survey, it.index);

        if (-eps <= t && t <= 1 + eps)
        {
            /* Current survey segment is intersected by the installed profile plane: compute the position. */
            const float_type dist = t * segment_get_length(segment);
            found_s = type_s + dist;
            *found_survey_index = it.index;
            break;
        }
    } while (zigzag_iterator_next(&it));

    return found_s;
}


static inline void bounds_on_s_for_aperture(
    const TypePosition type_pos,
    const ProfilePosition profile_pos,
    const SurveyData survey,
    const G2DPoint* const profile_points,
    const uint32_t num_poly_points,
    const uint32_t installed_survey_index,
    float_type* min_s,
    float_type* max_s
)
/*
    Get the survey s bounds that an installed profile spans.

    For each point of the installed profile in 3D space compute the s coordinate along the survey
    to which the point is closest, and take the minimum and maximum of those across all points.

    We ignore the degenerate cases where the curvature is such that a point is stuck in the centre
    of the arc. We also assume that once the shortest distance hits the middle of a given survey
    segment (as opposed to being clamped to the edge points) we have found the right segment.
    This is a fair assumption as the diameter of a profile << radius of curvature of the survey,
    but if that is not the case, the bounds will not be correct.

    TODO: Handling of arcs, for now the algorithm deals with straight segments only.
*/
{
    const float_type eps = APER_PRECISION;
    const uint32_t num_survey_entries = SurveyData_len_s(survey);

    // Transformation profile local -> type frame
    Transform profile_transform = {
        .x = ProfilePosition_get_shift_x(profile_pos),
        .y = ProfilePosition_get_shift_y(profile_pos),
        .s = ProfilePosition_get_s_position(profile_pos),
        .rot_x = ProfilePosition_get_rot_x(profile_pos),
        .rot_y = ProfilePosition_get_rot_y(profile_pos),
        .rot_s = ProfilePosition_get_rot_s(profile_pos),
    };
    Pose profile_in_type = transform_to_matrix(profile_transform);

    // Transformation from type frame -> survey reference point frame
    Pose type_in_survey_ref;
    for (uint8_t i = 0; i < 4; i++)
        for (uint8_t j = 0; j < 4; j++)
            type_in_survey_ref.mat[i][j] = TypePosition_get_transformation(type_pos, i, j);

    // Transformation from survey reference point -> world frame
    const uint32_t survey_idx = TypePosition_get_survey_index(type_pos);
    Pose survey_ref_in_world = pose_matrix_from_survey(survey, survey_idx);

    // Compute survey_ref_in_world @ type_in_survey_ref @ profile_in_type
    Pose plane_in_survey_ref;
    matrix_multiply_4x4(type_in_survey_ref.mat, profile_in_type.mat, plane_in_survey_ref.mat);
    Pose profile_in_world;
    matrix_multiply_4x4(survey_ref_in_world.mat, plane_in_survey_ref.mat, profile_in_world.mat);

    float_type out_min = INFINITY;
    float_type out_max = -INFINITY;

    for (uint32_t poly_point_idx = 0; poly_point_idx < num_poly_points; poly_point_idx++)
    {
        const Point3D pt_in_profile = (Point3D){
            profile_points[poly_point_idx].x,
            profile_points[poly_point_idx].y,
            0
        };
        const Point3D pt_in_world = pose_apply_point(profile_in_world, pt_in_profile);

        float_type closest_s = NAN;
        /*
            It's likely that survey segment at the installed s is where the bounds are, however this
            might not be always the case. So we test other segments working our way outwards in both directions.
        */
        ZigZagIterator it = zigzag_iterator_new(installed_survey_index, 0, num_survey_entries - 1);
        do
        {
            const LineSegment3D seg = survey_segment(survey, it.index);

            const Point3D a = seg.start;
            const Point3D b = seg.end;

            const float_type t = closest_t_on_segment(pt_in_world, a, b);

            if (-eps < t && t < 1 + eps) {
                /* Candidate s on this segment */
                const float_type seg_s_start = SurveyData_get_s(survey, it.index);
                const float_type seg_len = segment_get_length(seg);
                closest_s = seg_s_start + t * seg_len;
                break;
            }
        } while (zigzag_iterator_next(&it));

        if (closest_s < out_min) out_min = closest_s;
        if (closest_s > out_max) out_max = closest_s;
    }

    if (!isfinite(out_min) || !isfinite(out_max)) {
        *min_s = NAN;
        *max_s = NAN;
    } else {
        *min_s = out_min;
        *max_s = out_max;
    }
}


void build_profile_polygons(
    const ApertureModel model,
    const ProfilePolygons profile_polygons,
    const CrossSections cross_sections,
    const SurveyData survey
)
    /*
      Based on the aperture model and cross section location data, generate
      the bounds for each installed profile.
    */
{
    const uint32_t num_profiles = ProfilePolygons_get_count(profile_polygons);
    const uint32_t num_cross_sections = CrossSections_get_count(cross_sections);
    const uint32_t num_survey_entries = SurveyData_len_s(survey);

    /* First generate polygons for profiles */
    for (uint32_t idx = 0; idx < num_profiles; idx++)
    {
        const Profile profile = ApertureModel_getp1_profiles(model, idx);
        float_type *const points = ProfilePolygons_getp3_points(profile_polygons, idx, 0, 0);
        const uint32_t num_points = ProfilePolygons_get_num_points(profile_polygons);

        build_polygon_for_profile(points, num_points, profile);
    }

    /*
        Pre-process all installed profiles: compute correct s-positions for:
        - the intersection points with the survey,
        - the bounds along the survey s.
     */
    for (uint32_t idx = 0; idx < num_cross_sections; idx++)
    {
        const uint32_t type_pos_idx = CrossSections_get_type_position_indices(cross_sections, idx);
        const uint32_t profile_pos_idx = CrossSections_get_profile_position_indices(cross_sections, idx);

        /* Get the aperture type and type position */
        const TypePosition type_pos = ApertureModel_getp1_type_positions(model, type_pos_idx);
        const uint32_t type_idx = TypePosition_get_type_index(type_pos);
        const ApertureType aper_type = ApertureModel_getp1_types(model, type_idx);

        /* Get the profile position, and the polygon */
        const ProfilePosition profile_pos = ApertureType_getp1_positions(aper_type, profile_pos_idx);
        const uint32_t profile_idx = ProfilePosition_get_profile_index(profile_pos);
        float_type *const poly = ProfilePolygons_getp3_points(profile_polygons, profile_idx, 0, 0);

        /* Copy the points to the cross section */
        float_type *const cross_sec_points = CrossSections_getp3_points(cross_sections, idx, 0, 0);
        const uint32_t num_points = CrossSections_get_num_points(cross_sections);
        memcpy(cross_sec_points, poly, 2 * num_points * sizeof(float_type));

        /* Get the survey s where the aperture actually sits */
        uint32_t installed_survey_index;
        const float_type found_s = survey_s_for_aperture(type_pos, profile_pos, survey, &installed_survey_index);
        CrossSections_set_s_positions(cross_sections, idx, found_s);

        /* Get the bounds in s that the aperture spans */
        float_type min_s, max_s;
        const G2DPoint* const profile_points = (G2DPoint*)poly;
        bounds_on_s_for_aperture(type_pos, profile_pos, survey, profile_points, num_points, installed_survey_index, &min_s, &max_s);
        CrossSections_set_s_start(cross_sections, idx, min_s);
        CrossSections_set_s_end(cross_sections, idx, max_s);
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
