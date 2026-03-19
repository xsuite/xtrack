#ifndef XTRACK_POLYGONS_H
#define XTRACK_POLYGONS_H

#include <math.h>
#include <stdlib.h>

#include "base.h"
#include "path.h"
#include "survey_tools.h"
#include "zigzag_iterate.h"
#include "convert_curvilinear.h"


void build_profile_polygons(const ApertureModel, const ProfilePolygons, ApertureBounds, const SurveyData survey);
void cross_sections_at_s(
    const SurveyData survey_at_s,
    const ApertureModel,
    const ProfilePolygons,
    const ApertureBounds,
    const SurveyData,
    float_type* cross_sections);
void build_polygon_for_profile(float_type *const, const uint32_t, const Profile);
uint32_t find_aperture_info_for_s(const ApertureBounds, const float_type s, const uint32_t lower_bound);

static inline float_type survey_s_for_aperture(const TypePosition, const ProfilePosition, const float_type curvature, const SurveyData, uint32_t*);
static inline void bounds_on_s_for_aperture(
    const TypePosition,
    const ProfilePosition,
    const float_type curvature,
    const SurveyData,
    const Point2D* const,
    const uint32_t num_poly_points,
    const uint32_t installed_survey_index,
    float_type* min_s,
    float_type* max_s);

static inline void build_circle_polygon(Point2D *const, const uint32_t, const Circle);
static inline void build_rectangle_polygon(Point2D *const, const uint32_t, const Rectangle);
static inline void build_ellipse_polygon(Point2D *const, const uint32_t, const Ellipse);
static inline void build_rect_ellipse_polygon(Point2D *const, const uint32_t, const RectEllipse);
static inline void build_racetrack_polygon(Point2D *const, const uint32_t, const Racetrack);
static inline void build_octagon_polygon(Point2D *const, const uint32_t, const Octagon);
static inline void build_polygon_polygon(Point2D *const, const uint32_t, const Polygon);

static inline uint32_t find_aperture_info_bisection(const ApertureBounds, const float_type s);
static inline uint32_t find_aperture_info_linear(const ApertureBounds, const float_type s, const uint32_t lower_bound);
static inline uint32_t find_active_profile_for_s(const ApertureBounds, const float_type s, const uint32_t lower_bound);


static inline Pose pose_from_type_position(const TypePosition type_pos)
{
    Pose type_in_survey_ref;

    for (uint8_t i = 0; i < 4; i++)
        for (uint8_t j = 0; j < 4; j++)
            type_in_survey_ref.mat[i][j] = TypePosition_get_transformation(type_pos, i, j);

    return type_in_survey_ref;
}


void build_profile_polygons(
    const ApertureModel model,
    const ProfilePolygons profile_polygons,
    const ApertureBounds aperture_bounds,
    const SurveyData survey
)
    /*
      Based on the aperture model and cross section location data, generate
      the bounds for each installed profile.
    */
{
    const uint32_t num_profiles = ProfilePolygons_get_count(profile_polygons);
    const uint32_t num_cross_sections = ApertureBounds_get_count(aperture_bounds);
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
        const uint32_t type_pos_idx = ApertureBounds_get_type_position_indices(aperture_bounds, idx);
        const uint32_t profile_pos_idx = ApertureBounds_get_profile_position_indices(aperture_bounds, idx);

        /* Get the aperture type and type position */
        const TypePosition type_pos = ApertureModel_getp1_type_positions(model, type_pos_idx);
        const uint32_t type_idx = TypePosition_get_type_index(type_pos);
        const ApertureType aper_type = ApertureModel_getp1_types(model, type_idx);
        const float_type curvature = ApertureType_get_curvature(aper_type);

        /* Get the profile position, and the polygon */
        const ProfilePosition profile_pos = ApertureType_getp1_positions(aper_type, profile_pos_idx);
        const uint32_t profile_idx = ProfilePosition_get_profile_index(profile_pos);
        float_type *const poly = ProfilePolygons_getp3_points(profile_polygons, profile_idx, 0, 0);

        const uint32_t num_points = ProfilePolygons_get_num_points(profile_polygons);

        /* Get the survey s where the aperture actually sits */
        uint32_t installed_survey_index;
        const float_type found_s = survey_s_for_aperture(type_pos, profile_pos, curvature, survey, &installed_survey_index);

        ApertureBounds_set_s_positions(aperture_bounds, idx, found_s);

        /* Get the bounds in s that the aperture spans */
        float_type min_s, max_s;
        const Point2D* const profile_points = (Point2D*)poly;
        bounds_on_s_for_aperture(type_pos, profile_pos, curvature, survey, profile_points, num_points, installed_survey_index, &min_s, &max_s);
        ApertureBounds_set_s_start(aperture_bounds, idx, min_s);
        ApertureBounds_set_s_end(aperture_bounds, idx, max_s);
    }
}


static inline void aperture_profile_pose_frame(
    const TypePosition type_pos,
    const ProfilePosition profile_pos,
    const float_type curvature,
    const SurveyData survey,
    const Pose target_frame,
    Pose* out_profile_in_target_frame
)
/*
    Compute: profile_in_world = survey_ref_in_world @ type_in_survey_ref @ profile_in_type
*/
{
    // Transformation from local -> type frame
    const float_type ds = ProfilePosition_get_s_position(profile_pos);
    Pose profile_in_axis = transform_to_matrix((Transform) {
        .x = ProfilePosition_get_shift_x(profile_pos),
        .y = ProfilePosition_get_shift_y(profile_pos),
        .s = 0,
        .rot_x = ProfilePosition_get_rot_x(profile_pos),
        .rot_y = ProfilePosition_get_rot_y(profile_pos),
        .rot_s = ProfilePosition_get_rot_s(profile_pos),
    });
    Pose axis_in_type = arc_matrix(ds, ds * curvature, 0);
    Pose profile_in_type = matrix_multiply(axis_in_type, profile_in_axis);

    // Transformation from type frame -> survey reference point frame
    Pose type_in_survey_ref = pose_from_type_position(type_pos);

    // Transformation from survey reference point -> world frame
    const uint32_t survey_idx = TypePosition_get_survey_index(type_pos);
    Pose survey_ref_in_world = pose_matrix_from_survey(survey, survey_idx);

    // Compute survey_ref_in_target_frame = world_in_target_frame @ survey_ref_in_world
    Pose survey_ref_in_target_frame = matrix_multiply(target_frame, survey_ref_in_world);

    // Compute survey_ref_in_target_frame @ type_in_survey_ref @ profile_in_type
    Pose profile_in_survey = matrix_multiply(type_in_survey_ref, profile_in_type);
    *out_profile_in_target_frame = matrix_multiply(survey_ref_in_target_frame, profile_in_survey);
}


static inline Pose aperture_type_pose_in_world(
    const TypePosition type_pos,
    const SurveyData survey
)
{
    Pose type_in_survey_ref = pose_from_type_position(type_pos);
    const uint32_t survey_idx = TypePosition_get_survey_index(type_pos);
    Pose survey_ref_in_world = pose_matrix_from_survey(survey, survey_idx);
    return matrix_multiply(survey_ref_in_world, type_in_survey_ref);
}


static inline char get_aperture_polygon_and_pose(
    const ApertureModel model,
    const ProfilePolygons profile_polygons,
    const ApertureBounds aperture_bounds,
    const SurveyData survey,
    const uint32_t aper_info_idx,
    const Pose target_frame,
    const Point2D** out_poly,
    Pose* out_profile_in_world
)
/*
    Given `aper_info_idx`, the index of the aperture bounds entry corresponding to an installed profile,
    return the 2D polygon of the profile (projected onto the z = 0 of the profile frame), and the pose
    placing the polygon in world coordinates.
*/
{
    const uint32_t num_bounds = ApertureBounds_get_count(aperture_bounds);
    if (aper_info_idx < 0 || aper_info_idx >= num_bounds) return 0;

    const uint32_t type_pos_idx = ApertureBounds_get_type_position_indices(aperture_bounds, aper_info_idx);
    const uint32_t profile_pos_idx = ApertureBounds_get_profile_position_indices(aperture_bounds, aper_info_idx);

    const TypePosition type_pos = ApertureModel_getp1_type_positions(model, type_pos_idx);
    const uint32_t type_idx = TypePosition_get_type_index(type_pos);
    const ApertureType aper_type = ApertureModel_getp1_types(model, type_idx);

    const ProfilePosition profile_pos = ApertureType_getp1_positions(aper_type, profile_pos_idx);
    const uint32_t profile_idx = ProfilePosition_get_profile_index(profile_pos);

    /* Return the polygon */
    *out_poly = (const Point2D* const)ProfilePolygons_getp3_points(profile_polygons, profile_idx, 0, 0);

    /* Return the pose */
    const float_type curvature = ApertureType_get_curvature(aper_type);
    aperture_profile_pose_frame(type_pos, profile_pos, curvature, survey, target_frame, out_profile_in_world);
    return 1;
}


static inline void project_3d_polygon_to_plane(
    const Point2D* poly_local,
    const Pose profile_in_world,
    const Pose world_in_plane,
    const uint32_t num_points,
    Point2D* out_poly_plane
)
/* Project a local 2D polygon (lying on z = 0 in the local frame) into plane frame. */
{
    for (uint32_t j = 0; j < num_points; j++) {
        const Point3D p_local = (Point3D){ poly_local[j].x, poly_local[j].y, 0.f };
        const Point3D p_world = pose_apply_point(profile_in_world, p_local);
        const Point3D p_plane = pose_apply_point(world_in_plane, p_world);
        out_poly_plane[j].x = p_plane.x;
        out_poly_plane[j].y = p_plane.y;
    }
}


static inline uint32_t find_best_cyclic_shift_plane(
    const Point2D* p0_plane,
    const Point2D* p1_plane,
    const uint32_t n
)
/* Find `shift` that minimises sum of squared distances between `p0[j]` and `p1[(j + shift) % n]`. */
{
    float_type best_cost = INFINITY;
    uint32_t best_shift = 0;

    for (uint32_t shift = 0; shift < n; shift++) {
        float_type cost = 0.f;
        for (uint32_t j = 0; j < n; j++) {
            const uint32_t k = (j + shift) % n;
            const float_type dx = p0_plane[j].x - p1_plane[k].x;
            const float_type dy = p0_plane[j].y - p1_plane[k].y;
            cost += dx * dx + dy * dy;
        }
        if (cost < best_cost) {
            best_cost = cost;
            best_shift = shift;
        }
    }

    return best_shift;
}


static inline int intersect_segment_with_plane_and_project_xy(
    Point3D a,
    Point3D b,
    const Pose plane_in_frame,
    const Pose frame_in_plane,
    const float_type curvature,
    Point2D* out_xy_plane
)
/*
    Intersect segment [a, b] with plane (z = 0 in the local `plane_in_frame` frame)
    and project intersection into plane coordinates (x,y). Returns 1 on success, 0 otherwise.
*/
{
    const float_type eps = APER_PRECISION;
    const Point3D a_cart = a;
    const Point3D b_cart = b;
    Point3D plane_point = plane_initial_point(plane_in_frame);
    Point3D normal = plane_normal_vector(plane_in_frame);

    /*
        If there is curvature, we "unbend" it (move to curvilinear frame through
        "locally rigid" transformations). This is a little bit of a hack
        for profiles/planes that are not orthogonal to the type frame s,
        but it will do for now, and avoids complex mathematics for proper
        interpolation along curves (which are in this case not circle arcs!)

        TODO: This is not robust for arcs > 90°, a different method needed for that?
    */
    if (fabs(curvature) > APER_PRECISION)
    {
        a = cartesian_to_curvilinear_point(a, curvature);
        b = cartesian_to_curvilinear_point(b, curvature);
        plane_point = cartesian_to_curvilinear_point(plane_point, curvature);
        normal = cartesian_vector_to_curvilinear_at_point(plane_point, normal, curvature);
    }

    /*
        Intersect in the straight (or straightened) frame
    */
    LineSegment3D seg = (LineSegment3D){ .start = a, .end = b };
    const float_type t = line_segment_plane_intersect(seg, plane_point, normal);

    if (!isfinite(t)) {
        /*
            Near-parallel case: if an endpoint is already on the target plane within tolerance,
            use that endpoint rather than dropping the point.
        */
        const Point3D a_plane = pose_apply_point(frame_in_plane, a_cart);
        if (fabs(a_plane.z) <= eps) {
            out_xy_plane->x = a_plane.x;
            out_xy_plane->y = a_plane.y;
            return 1;
        }

        const Point3D b_plane = pose_apply_point(frame_in_plane, b_cart);
        if (fabs(b_plane.z) <= eps) {
            out_xy_plane->x = b_plane.x;
            out_xy_plane->y = b_plane.y;
            return 1;
        }

        return 0;
    }
    if (t < -eps || (1.f + eps) < t) return 0;

    const float_type tt = clamp_value(t, 0.f, 1.f);
    const Point3D dir = point3d_sub(seg.end, seg.start);
    Point3D hit_frame = point3d_add_scaled(seg.start, dir, tt);

    /* Unbend */
    if (fabs(curvature) > APER_PRECISION)
        hit_frame = curvilinear_to_cartesian_point(hit_frame, curvature);

    const Point3D hit_plane = pose_apply_point(frame_in_plane, hit_frame);
    out_xy_plane->x = hit_plane.x;
    out_xy_plane->y = hit_plane.y;
    return 1;
}


void cross_sections_at_s(
    const SurveyData survey_at_s,
    const ApertureModel model,
    const ProfilePolygons profile_polys,
    const ApertureBounds bounds,
    const SurveyData survey,
    float_type* cross_sections
)
{
    const float_type eps = APER_PRECISION;
    const uint32_t num_points = ProfilePolygons_get_num_points(profile_polys);
    const uint32_t num_unique_points = num_points - 1;
    const uint32_t num_cross_sections = SurveyData_len_s(survey_at_s);
    const uint32_t num_bounds = ApertureBounds_get_count(bounds);

    #ifdef XO_CONTEXT_CPU
        int completed = 0;
    #endif

    uint32_t bound_idx = 0;
    #pragma omp parallel for firstprivate(bound_idx)
    for (uint32_t i = 0; i < num_cross_sections; i++)
    {
        const float_type s = SurveyData_get_s(survey_at_s, i);
        Point2D* poly_at_s = (Point2D*)cross_sections + i * num_points;

        /* Use aperture bounds information to find the relevant profile for this s */
        bound_idx = find_active_profile_for_s(bounds, s, bound_idx);

        if (bound_idx >= num_bounds) {
            /* If requested s is outside of the model, give up and return NANs */
            for (uint32_t j = 0; j < num_points; j++) {
                poly_at_s[j].x = NAN;
                poly_at_s[j].y = NAN;
            }
            continue;
        }

        /*
            Find the bound which either contains s or is immediately to the left of it.
        */
        const float_type s_center = ApertureBounds_get_s_positions(bounds, bound_idx);

        const Point2D* poly_center = NULL;
        const Point2D* poly_left = NULL;
        const Point2D* poly_right = NULL;
        Pose pose_center, pose_left, pose_right;

        /*
            Determine whether the installed profiles on either side are in the same type frame.
            If so, we will need to interpolate on the associated curve.
        */
        float_type curvature_left = 0, curvature_right = 0;

        const uint32_t type_pos_idx = ApertureBounds_get_type_position_indices(bounds, bound_idx);
        const TypePosition type_pos = ApertureModel_getp1_type_positions(model, type_pos_idx);
        const uint32_t type_idx = TypePosition_get_type_index(type_pos);
        const ApertureType aper_type = ApertureModel_getp1_types(model, type_idx);
        const float_type curvature = ApertureType_get_curvature(aper_type);

        if (bound_idx > 0 && type_pos_idx == ApertureBounds_get_type_position_indices(bounds, bound_idx - 1))
            curvature_left = curvature;

        if (bound_idx + 1 < num_bounds && type_pos_idx == ApertureBounds_get_type_position_indices(bounds, bound_idx + 1))
            curvature_right = curvature;

        /*
            We want to move to the type frame, so calculate the transformations
        */
        const Pose plane_in_world = pose_matrix_from_survey(survey_at_s, i);
        const Pose type_in_world = aperture_type_pose_in_world(type_pos, survey);
        const Pose world_in_type = pose_inverse_rigid(type_in_world);
        const Pose plane_in_type = matrix_multiply(world_in_type, plane_in_world);
        const Pose type_in_plane = pose_inverse_rigid(plane_in_type);

        /*
            Get polygons and poses in the correct frame(s).
        */
        get_aperture_polygon_and_pose(model, profile_polys, bounds, survey, bound_idx, world_in_type, &poly_center, &pose_center);
        const char has_left = get_aperture_polygon_and_pose(model, profile_polys, bounds, survey, bound_idx - 1, world_in_type, &poly_left, &pose_left);
        const char has_right = get_aperture_polygon_and_pose(model, profile_polys, bounds, survey, bound_idx + 1, world_in_type, &poly_right, &pose_right);

        /*
            Compute the best cyclic shift between the pairs of profiles for the purposes of interpolation.

            We do this by projecting each installed profile onto the plane orthogonal to the survey at
            the s location where the profile sits. Then we minimise the distances between the points:

                dist2d(j) := \Sigma_j (poly2d[j].x - poly2d[i + j].x)^2 + (poly2d[j].y - poly2d[i + j].y)^2

            Note that this is not the same as minimising

                dist3d(j) := dist2d(j) + (poly[j].z - poly[i + j].z)^2

            However, assuming the z-term is not significantly varying between pairs of points and/or does not
            dominate, the following should be good enough. This is cheaper than a full 3D minimisation.
        */
        Point2D poly_center_plane[num_points];
        Point2D poly_left_plane[num_points];
        Point2D poly_right_plane[num_points];
        project_3d_polygon_to_plane(poly_center, pose_center, type_in_plane, num_points, poly_center_plane);
        if (has_left) project_3d_polygon_to_plane(poly_left, pose_left, type_in_plane, num_points, poly_left_plane);
        if (has_right) project_3d_polygon_to_plane(poly_right, pose_right, type_in_plane, num_points, poly_right_plane);

        if (fabs(s - s_center) < eps) {
            for (uint32_t j = 0; j < num_points; j++) poly_at_s[j] = poly_center_plane[j];
            continue;
        }

        const uint32_t shift_center_left = has_left
            ? find_best_cyclic_shift_plane(poly_center_plane, poly_left_plane, num_unique_points)
            : 0;
        const uint32_t shift_center_right = has_right
            ? find_best_cyclic_shift_plane(poly_center_plane, poly_right_plane, num_unique_points)
            : 0;
        const char prefer_right = (s >= s_center);

        /* Interpolate */
        for (uint32_t j = 0; j < num_unique_points; j++) {
            char has_intersection = 0;
            Point2D hit_point_plane = (Point2D){ .x = NAN, .y = NAN };

            /* Try the geometrically expected side first, then the opposite side */
            for (uint32_t attempt = 0; attempt < 2 && !has_intersection; attempt++)
            {
                const int use_right = (attempt == 0) ? prefer_right : !prefer_right;
                if ((use_right && !has_right) || (!use_right && !has_left)) continue;

                const Pose* pose_a = use_right ? &pose_center : &pose_left;
                const Pose* pose_b = use_right ? &pose_right : &pose_center;
                const Point2D* poly_a = use_right ? poly_center : poly_left;
                const Point2D* poly_b = use_right ? poly_right : poly_center;
                const uint32_t idx_a = use_right ? j : (j + shift_center_left) % num_unique_points;
                const uint32_t idx_b = use_right ? (j + shift_center_right) % num_unique_points : j;
                const float_type curvature = use_right ? curvature_right : curvature_left;

                const Point3D point_a_type = pose_apply_point(
                    *pose_a, (Point3D){ poly_a[idx_a].x, poly_a[idx_a].y, 0.f });
                const Point3D point_b_type = pose_apply_point(
                    *pose_b, (Point3D){ poly_b[idx_b].x, poly_b[idx_b].y, 0.f });

                has_intersection = intersect_segment_with_plane_and_project_xy(
                    point_a_type, point_b_type, plane_in_type, type_in_plane, curvature, &hit_point_plane);
            }

            poly_at_s[j] = has_intersection ? hit_point_plane : poly_center_plane[j];
        }
        poly_at_s[num_points - 1] = poly_at_s[0];

        #ifdef XO_CONTEXT_CPU
            printf("Interpolating profiles: %d%%\r", 100 * (++completed) / num_cross_sections);
            fflush(stdout);
        #endif
    }

    #ifdef XO_CONTEXT_CPU
        printf("\n");
    #endif
}


void build_polygon_for_profile(
    float_type *const points,
    const uint32_t num_points,
    const Profile profile
)
{
    /*
        Convert the logical description of a profile to a polygon, and store it in
        ``aperture_bounds``.
    */
    const uint64_t profile_type_id = Profile_typeid_shape(profile);

    switch (profile_type_id)
    {
        case Shape_Circle_t:  // LHC
        {
            const Circle circle = Profile_member_shape(profile);
            build_circle_polygon((Point2D* const) points, num_points, circle);
            break;
        }
        case Shape_Rectangle_t:
        {
            const Rectangle rectangle = Profile_member_shape(profile);
            build_rectangle_polygon((Point2D* const) points, num_points, rectangle);
            break;
        }
        case Shape_Ellipse_t:
        {
            const Ellipse ellipse = Profile_member_shape(profile);
            build_ellipse_polygon((Point2D* const) points, num_points, ellipse);
            break;
        }
        case Shape_RectEllipse_t:
        {
            const RectEllipse rect_ellipse = Profile_member_shape(profile);
            build_rect_ellipse_polygon((Point2D* const) points, num_points, rect_ellipse);
            break;
        }
        case Shape_Racetrack_t:
        {
            const Racetrack racetrack = Profile_member_shape(profile);
            build_racetrack_polygon((Point2D* const) points, num_points, racetrack);
            break;
        }
        case Shape_Octagon_t:
        {
            const Octagon octagon = Profile_member_shape(profile);
            build_octagon_polygon((Point2D* const) points, num_points, octagon);
            break;
        }
        case Shape_Polygon_t:
        {
            const Polygon polygon = Profile_member_shape(profile);
            build_polygon_polygon((Point2D* const) points, num_points, polygon);
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


uint32_t find_aperture_info_for_s(
    const ApertureBounds aperture_bounds,
    const float_type target_s,
    const uint32_t lower_bound
)
{
    uint32_t found_idx;
    if (lower_bound == 0) {
        // If starting from the left, let's do a bisection because chances are we need to jump to a random point
        found_idx = find_aperture_info_bisection(aperture_bounds, target_s);
    }
    else {
        // If a lower bound is specified, chances are the next point that we search for is close to the right
        found_idx = find_aperture_info_linear(aperture_bounds, target_s, lower_bound);
    }

    return found_idx;
}


static inline float_type survey_s_for_aperture(
    const TypePosition type_pos,
    const ProfilePosition profile_pos,
    const float_type curvature,
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
    const uint32_t survey_idx = TypePosition_get_survey_index(type_pos);

    // Transformation from plane (s = 0) -> world
    Pose plane_in_world;
    aperture_profile_pose_frame(type_pos, profile_pos, curvature, survey, identity(), &plane_in_world);

    /*
        For data from MAD-X etc. it's likely that it's the type's reference survey point where the profile
        intersects the survey (the true installed s), however, that is not necessarily true. We find the
        s by testing that survey "segment" first, and working our way outwards in both directions.
    */
    float_type found_s = NAN;
    float_type last_t_right = NAN;
    float_type last_t_left = NAN;
    ZigZagIterator it = zigzag_iterator_new(survey_idx, num_survey_entries - 1);
    do
    {
        Segment3D segment = survey_segment(survey, it.index);
        const Point3D plane_point = plane_initial_point(plane_in_world);
        const Point3D normal = plane_normal_vector(plane_in_world);
        const float_type t = segment3d_plane_intersect(segment, plane_point, normal);

        const float_type type_s = SurveyData_get_s(survey, it.index);

        if (
            /* Candidate s on this segment, or... */
            (-eps < t && t < 1 + eps) ||
            /*
                ...unfortunately, due to numerical instability, it can happen that for one segment
                we get t > 1 + eps, but for the adjacent one t < -eps (or analogously on the left
                side). Detect if there was a sign change, and if so return the current solution.
            */
            (it.offset > 0 && isfinite(last_t_right) && signbit(t) != signbit(last_t_right)) ||
            (it.offset < 0 && isfinite(last_t_left) && signbit(t) != signbit(last_t_left))
        ) {
            const float_type dist = t * segment3d_get_length(segment);
            found_s = type_s + dist;
            *found_survey_index = it.index;
            break;
        }

        if (it.offset >= 0 && isfinite(t)) last_t_right = t;
        if (it.offset <= 0 && isfinite(t)) last_t_left = t;
    } while (zigzag_iterator_next(&it));

    if (!isfinite(found_s))
        printf("survey_s_for_aperture returning NAN: profile sits outside of the survey, a bug in the aperture model?\n");
    fflush(stdout);
    return found_s;
}


static inline void bounds_on_s_for_aperture(
    const TypePosition type_pos,
    const ProfilePosition profile_pos,
    const float_type curvature,
    const SurveyData survey,
    const Point2D* const profile_points,
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
*/
{
    const float_type eps = APER_PRECISION;
    const uint32_t num_survey_entries = SurveyData_len_s(survey);

    // Transformation profile local -> world frame
    Pose profile_in_world;
    aperture_profile_pose_frame(type_pos, profile_pos, curvature, survey, identity(), &profile_in_world);

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
        float_type last_t_right = NAN;
        float_type last_t_left = NAN;
        /*
            It's likely that survey segment at the installed s is where the bounds are, however this
            might not be always the case. So we test other segments working our way outwards in both directions.
        */
        ZigZagIterator it = zigzag_iterator_new(installed_survey_index, num_survey_entries - 1);
        do
        {
            const Segment3D seg = survey_segment(survey, it.index);
            const float_type t = closest_t_on_segment(pt_in_world, seg);

            if (
                /* Candidate s on this segment, or... */
                (-eps < t && t < 1 + eps) ||
                /*
                    ...unfortunately, due to numerical instability, it can happen that for one segment
                    we get t > 1 + eps, but for the adjacent one t < -eps (or analogously on the left
                    side). Detect if there was a sign change, and if so return the current solution.
                */
                (it.offset > 0 && isfinite(last_t_right) && signbit(t) != signbit(last_t_right)) ||
                (it.offset < 0 && isfinite(last_t_left) && signbit(t) != signbit(last_t_left))
            ) {
                const float_type seg_s_start = SurveyData_get_s(survey, it.index);
                const float_type seg_len = segment3d_get_length(seg);
                closest_s = seg_s_start + t * seg_len;
                break;
            }

            if (it.offset >= 0 && isfinite(t)) last_t_right = t;
            if (it.offset <= 0 && isfinite(t)) last_t_left = t;
        } while (zigzag_iterator_next(&it));

        if (closest_s < out_min) out_min = closest_s;
        if (closest_s > out_max) out_max = closest_s;
    }

    if (!isfinite(out_min) || !isfinite(out_max)) {
        *min_s = NAN;
        *max_s = NAN;
        printf("Warning: returning NAN from the aperture bounds calculation...\n");
        fflush(stdout);
    } else {
        *min_s = out_min;
        *max_s = out_max;
    }
}


static inline void build_circle_polygon(Point2D *const points, const uint32_t num_points, const Circle circle)
{
    const float_type radius = Circle_get_radius(circle);

    Segment2D segments[1];
    Path2D path = {.segments = segments, .len_segments = 1};
    segments_from_circle(radius, segments);
    poly_get_n_uniform_points(&path, num_points, points);
}


static inline void build_rectangle_polygon(Point2D *const points, const uint32_t num_points, const Rectangle rectangle)
{
    const float_type half_width = Rectangle_get_half_width(rectangle);
    const float_type half_height = Rectangle_get_half_height(rectangle);

    Segment2D segments[4];
    Path2D path = {.segments = segments, .len_segments = 4};
    segments_from_rectangle(half_width, half_height, segments);
    poly_get_n_uniform_points(&path, num_points, points);
}


static inline void build_ellipse_polygon(Point2D *const points, const uint32_t num_points, const Ellipse ellipse)
{
    const float_type half_major = Ellipse_get_half_major(ellipse);
    const float_type half_minor = Ellipse_get_half_minor(ellipse);

    Segment2D segments[1];
    Path2D path = {.segments = segments, .len_segments = 1};
    segments_from_ellipse(half_major, half_minor, segments);
    poly_get_n_uniform_points(&path, num_points, points);
}


static inline void build_rect_ellipse_polygon(Point2D *const points, const uint32_t num_points, const RectEllipse rect_ellipse)
{
    const float_type half_width = RectEllipse_get_half_width(rect_ellipse);
    const float_type half_height = RectEllipse_get_half_height(rect_ellipse);
    const float_type half_major = RectEllipse_get_half_major(rect_ellipse);
    const float_type half_minor = RectEllipse_get_half_minor(rect_ellipse);

    Segment2D segments[8];
    Path2D path = {.segments = segments, .len_segments = 8};
    segments_from_rectellipse(half_width, half_height, half_major, half_minor, segments, &path.len_segments);
    poly_get_n_uniform_points(&path, num_points, points);
}


static inline void build_racetrack_polygon(Point2D *const points, const uint32_t num_points, const Racetrack racetrack)
{
    const float_type half_width = Racetrack_get_half_width(racetrack);
    const float_type half_height = Racetrack_get_half_height(racetrack);
    const float_type half_major = Racetrack_get_half_major(racetrack);
    const float_type half_minor = Racetrack_get_half_minor(racetrack);

    Segment2D segments[8];
    Path2D path = {.segments = segments, .len_segments = 8};
    segments_from_racetrack(half_width, half_height, half_major, half_minor, segments, &path.len_segments);
    poly_get_n_uniform_points(&path, num_points, points);
}


static inline void build_octagon_polygon(Point2D *const points, const uint32_t num_points, const Octagon octagon)
{
    const float_type half_width = Octagon_get_half_width(octagon);
    const float_type half_height = Octagon_get_half_height(octagon);
    const float_type half_diagonal = Octagon_get_half_diagonal(octagon);

    Segment2D segments[8];
    Path2D path = {.segments = segments, .len_segments = 8};
    segments_from_octagon(half_width, half_height, half_diagonal, segments, &path.len_segments);
    poly_get_n_uniform_points(&path, num_points, points);
}


static inline void build_polygon_polygon(Point2D *const points, const uint32_t num_points, const Polygon polygon)
{
    // TODO: Not yet implemented, requires resampling the polygon
}


static inline uint32_t find_aperture_info_bisection(
    const ApertureBounds aperture_bounds,
    const float_type target_s
)
{
    const uint32_t num_apertures = ApertureBounds_get_count(aperture_bounds);

    uint32_t lo = 0;
    uint32_t hi = num_apertures;

    while (hi > lo) {
        const uint32_t mid = lo + (hi - lo) / 2;
        float_type current_s = ApertureBounds_get_s_positions(aperture_bounds, mid);

        if (current_s <= target_s) lo = mid + 1;
        else hi = mid;
    }

    const uint32_t found_idx = (lo == 0 ? 0 : lo - 1);
    return found_idx;
}


static inline uint32_t find_aperture_info_linear(
    const ApertureBounds aperture_bounds,
    const float_type target_s,
    const uint32_t lower_bound
)
{
    const uint32_t num_apertures = ApertureBounds_get_count(aperture_bounds);

    uint32_t found_idx = lower_bound;

    for (uint32_t i = lower_bound; i < num_apertures; ++i) {
        float_type current_s = ApertureBounds_get_s_positions(aperture_bounds, i);

        if (current_s <= target_s) {
            found_idx = i;  /* keep advancing, last duplicate wins */
        } else {
            break;
        }
    }
    return found_idx;
}


static inline uint32_t find_active_profile_for_s(
    const ApertureBounds aperture_bounds,
    const float_type target_s,
    const uint32_t lower_bound
)
/*
    Find an anchor profile index for interpolation at target_s.

    Assumes bounds are non-overlapping and ordered by s.

    Returns:
    --------
    If target_s is inside some bound, return that interval's index. If target_s is between intervals,
    return the index of the bound immediately preceding target_s (last one in sequence on s clash).
*/
{
    const uint32_t num_bounds = ApertureBounds_get_count(aperture_bounds);
    uint32_t idx = lower_bound;

    while (idx + 1 < num_bounds && target_s >= ApertureBounds_get_s_start(aperture_bounds, idx + 1)) {
        idx++;
    }

    return idx;
}

#endif /* XTRACK_POLYGONS_H */
