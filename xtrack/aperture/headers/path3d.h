#ifndef XT_APERTURE_PATH_3D_H
#define XT_APERTURE_PATH_3D_H

#include "base.h"

typedef struct {
    Point3D start;
    Point3D end;
} LineSegment3D;


typedef struct {
    Pose start;
    float_type length;
    float_type curvature;
    float_type roll;
} ArcSegment3D;


typedef enum {
    SEGMENT3D_LINE,
    SEGMENT3D_ARC,
} Segment3DType;


typedef struct {
    Segment3DType type;
    union {
        LineSegment3D line;
        ArcSegment3D arc;
    };
} Segment3D;


float_type segment_get_squared_length(LineSegment3D segment) {
    const float_type dx = segment.end.x - segment.start.x;
    const float_type dy = segment.end.y - segment.start.y;
    const float_type dz = segment.end.z - segment.start.z;
    return dx * dx + dy * dy + dz * dz;
}


float_type segment_get_length(LineSegment3D segment) {
    const float_type dx = segment.end.x - segment.start.x;
    const float_type dy = segment.end.y - segment.start.y;
    const float_type dz = segment.end.z - segment.start.z;
    return hypot(hypot(dx, dy), dz);
}


float_type segment3d_get_length(Segment3D segment) {
    switch (segment.type) {
        case SEGMENT3D_LINE:
            return segment_get_length(segment.line);
        case SEGMENT3D_ARC:
            return segment.arc.length;
    }
}


inline Point3D line_segment_point_at(const LineSegment3D segment, const float_type frac_length)
{
    const Point3D direction = point3d_sub(segment.end, segment.start);
    return point3d_add_scaled(segment.start, direction, frac_length);
}


inline Point3D arc_segment_point_at(const ArcSegment3D segment, const float_type frac_length)
{
    const float_type length = frac_length * segment.length;
    const float_type curvature = segment.curvature;
    const float_type roll = segment.roll;

    float_type dx, ds;
    if (fabs(curvature) < APER_PRECISION) {
        dx = 0.f;
        ds = length;
    } else {
        const float_type angle = curvature * length;
        dx = (cos(angle) - 1) / curvature;
        ds = sin(angle) / curvature;
    }

    const float_type c_roll = cos(roll);
    const float_type s_roll = sin(roll);
    const Point3D local = (Point3D){
        .x = c_roll * dx,
        .y = s_roll * dx,
        .z = ds,
    };

    return pose_apply_point(segment.start, local);
}


inline Point3D segment_point_at(const Segment3D segment, const float_type frac_length)
{
    switch (segment.type) {
        case SEGMENT3D_LINE:
            return line_segment_point_at(segment.line, frac_length);
        case SEGMENT3D_ARC:
            return arc_segment_point_at(segment.arc, frac_length);
    }
}


inline Point3D plane_initial_point(const Pose plane)
/*
    Given a pose defining a plane (z = 0 in the local `plane` frame), return
    its initial point.
*/
{
    return (Point3D) {
        .x = plane.mat[0][3],
        .y = plane.mat[1][3],
        .z = plane.mat[2][3]
    };
}


inline Point3D plane_normal_vector(const Pose plane)
/*
    Given a pose defining a plane (z = 0 in the local `plane` frame), return
    its normal point.
*/
{
    return (Point3D) {
        .x = plane.mat[0][2],
        .y = plane.mat[1][2],
        .z = plane.mat[2][2]
    };
}


float_type line_segment_plane_intersect(LineSegment3D segment, const Point3D plane_point, const Point3D normal)
/*
    Return a parameter `t` such that `segment.start + t * (segment.end - segment.start)` is the
    point on the segment at which the plane` intersects the `segment`.
*/
{
    const float_type eps = APER_PRECISION;

    // Let A := segment.start, T := plane_point, compute A - T
    const Point3D ta = point3d_sub(segment.start, plane_point);

    // Let B := segment.end, T := plane_point, compute B - T
    const Point3D tb = point3d_sub(segment.end, plane_point);

    /*
        The equation P(t) = A + t * (B - A), t \in [0, 1], defines the line segment, while
        n * (X - T) = 0 defines the plane, where n := normal.
        
        Substituting P(t) in the latter gives n * (A + t * (B - A) - T) = 0, and after rearranging
        n * (A - T) + tn * (B - A) = 0 and then t = - (n * (A - T)) / (n * (B - A)).
    */

    const float_type n_dot_ta = point3d_dot(normal, ta);  /* n * (A - T) */
    const float_type n_dot_tb = point3d_dot(normal, tb);  /* n * (B - T) */

    /* Handle the degenerate case: if the line segment has no length, only check if point is on the plane */
    const float_type length_sq = segment_get_squared_length(segment);

    if (length_sq < eps * eps) {
        /*
            TA is the vector from the "origin" of the normal n to A.
            If the vectors are orthogonal, A is on the plane described by T and n.
        */
        if (fabs(n_dot_ta) < eps) return 0;
        else return NAN;
    }

    const float_type n_dot_ab = (n_dot_tb - n_dot_ta);  /* n * (B - A), by the distributive property of dot product */

    if (fabs(n_dot_ab) < eps) {
        /*
            Segment is parallel (or near-parallel) to the plane.
            If an endpoint lies on the plane within tolerance, use that endpoint to avoid
            spurious NaNs from numerical noise.
        */
        const int a_on_plane = fabs(n_dot_ta) < eps;
        const int b_on_plane = fabs(n_dot_tb) < eps;

        if (a_on_plane && b_on_plane) return 0.f;
        if (a_on_plane) return 0.f;
        if (b_on_plane) return 1.f;

        return NAN;
    }

    return -n_dot_ta / n_dot_ab;
}


float_type arc_segment_plane_intersect(const ArcSegment3D segment, const Point3D plane_point, const Point3D normal)
/*
    Return a parameter `t` such that the point at `t * segment.angle` is the point on the
    segment at which `plane` intersects the `segment`. The plane is defined as a
    pose, such that if the pose is an identity, the plane lies in the X-Y plane (z=0), or, in
    other words, `plane @ [0, 0, 1]^T` is the plane normal.
*/
{
    const float_type eps = APER_PRECISION;
    const float_type length = segment.length;

    if (length <= eps) return 0.f;

    // Extract translation T from pose (local -> world)
    const float_type t_x = plane_point.x;
    const float_type t_y = plane_point.y;
    const float_type t_z = plane_point.z;

    // Extract 3rd column of rotation R (plane normal in world)
    const float_type n_x = normal.x;
    const float_type n_y = normal.y;
    const float_type n_z = normal.z;

    // Let A := segment start, compute A - T
    const Point3D p_start = arc_segment_point_at(segment, 0);
    const float_type ta_x = p_start.x - t_x;
    const float_type ta_y = p_start.y - t_y;
    const float_type ta_z = p_start.z - t_z;

    // Let B := segment end, compute B - T
    const Point3D p_end = arc_segment_point_at(segment, 1);
    const float_type tb_x = p_end.x - t_x;
    const float_type tb_y = p_end.y - t_y;
    const float_type tb_z = p_end.z - t_z;

    /*
        For numerical stability we will use bisection to find the intersection point.
        In principle there might be two such points, but is this is not expected in
        practice, we go on assuming there is just one.

        We iteratively converge on an interval where n * (A - T) and n * (B - T),
        the signed distances between points A, B and the plane, have different signs.
    */
    float_type n_dot_ta = n_x * ta_x + n_y * ta_y + n_z * ta_z;
    float_type n_dot_tb = n_x * tb_x + n_y * tb_y + n_z * tb_z;

    /* Short-circuit if already on one of the points */
    if (fabs(n_dot_ta) <= eps) return 0;
    if (fabs(n_dot_tb) <= eps) return 1;

    /* If no sign change assume no intersection (we assume max one intersection point) */
    if (signbit(n_dot_ta) == signbit(n_dot_tb)) return NAN;

    /* Bisect */
    float_type d_lo = 0;
    float_type d_hi = 1;
    float_type d_mid = 0.5;

    for (int i = 0; i < 34; i++) {
        d_mid = 0.5f * (d_lo + d_hi);
        const Point3D p_mid = arc_segment_point_at(segment, d_mid);
        const float_type n_dot_t_mid = n_x * (p_mid.x - t_x) + n_y * (p_mid.y - t_y) + n_z * (p_mid.z - t_z);

        /* Solution found within precision */
        if (fabs(n_dot_t_mid) <= eps || (d_hi - d_lo) <= eps) {
            return d_mid;
        }

        if (signbit(n_dot_t_mid) == signbit(n_dot_ta)) {
            /* If projections have the same sign, bisect on [d_mid, d_hi] */
            d_lo = d_mid;
            n_dot_ta = n_dot_t_mid;
        } else {
            /* Otherwise bisect on the interval [d_lo, d_mid] */
            d_hi = d_mid;
        }
    }

    return d_mid;
}


inline float_type segment3d_plane_intersect(const Segment3D segment, const Point3D plane_point, const Point3D normal)
/*
    Given a `segment` (line or arc) and a plane defined with a point and a normal, get a parameter `t` along
    the length of the `segment` at which the plane intersects the segment. If `t` is not in [0, 1] the plane
    does not intersect the segment.
*/
{
    switch (segment.type) {
        case SEGMENT3D_LINE:
            return line_segment_plane_intersect(segment.line, plane_point, normal);
        case SEGMENT3D_ARC:
            return arc_segment_plane_intersect(segment.arc, plane_point, normal);
    }
}


inline float_type closest_t_on_line_segment(const Point3D p, const LineSegment3D segment)
/*
    Closest point parameter `t` on segment [a,b] to point p. Parameter `t` is unconstrained,
    if the point strictly on the segment is needed, clamp to [0, 1].
*/
{
    const Point3D a = segment.start;
    const Point3D b = segment.end;
    const float_type eps = APER_PRECISION;
    const Point3D ab = point3d_sub(b, a);
    const Point3D ap = point3d_sub(p, a);
    const float_type segment_length = point3d_dot(ab, ab);

    if (segment_length < eps) return 0;

    const float_type t = point3d_dot(ap, ab) / segment_length;
    return t;
}


inline float_type closest_t_on_arc_segment(const Point3D p, const ArcSegment3D segment)
/*
    Closest point parameter t on an ArcSegment3D to point p.

    The returned value is the normalized arc parameter, such that
    `arc_segment_point_at(segment, t)` is the closest point on the supporting arc.

    The returned parameter `t` is unconstrained, if the point strictly  on the arc
    segment is needed, clamp to [0, 1].
*/
{
    const float_type eps = APER_PRECISION;
    if (segment.length < eps) return 0.f;

    const float_type h = segment.curvature;

    /* Transform p into the local frame of the arc start */
    const Pose inv_start = pose_inverse_rigid(segment.start);
    const Point3D q = pose_apply_point(inv_start, p);

    /*
        Undo the arc roll to bring the point into the frame in which
        the arc lies in the X-Z plane. This is now a 2D problem, and
        the Y coordinate is irrelevant for the computation.
    */
    const float_type x = cos(segment.roll) * q.x + sin(segment.roll) * q.y;
    const float_type z = q.z;


    /* Shart-circuit if actually a line segment */
    if (fabs(h) < eps) {
        return z / segment.length;
    }

    /*
        In the local X-Z plane the arc is a circle with centre `C = (-R, 0)`
        and radius `R = 1 / h`. The closest point to `q` on the arc is the
        point lying on the ray from `C` to `q`.

        The ray is:
            v = (x - C_x, z - C_z) = (x + 1 / h, z)

        and the corresponding circle angle is then:
            theta = atan2(v_z, v_x) = atan2(h * v_z, h * v_x) = atan2(h * z, 1 + h * x)
    */
    const float_type theta = atan2(h * z, 1.f + h * x);

    return theta / (h * segment.length);
}


inline float_type closest_t_on_segment(const Point3D p, const Segment3D segment)
/*
    Get parameter `t` of the point along the `segment` closest to `p`.
*/
{
    switch (segment.type) {
        case SEGMENT3D_LINE:
            return closest_t_on_line_segment(p, segment.line);
        case SEGMENT3D_ARC:
            return closest_t_on_arc_segment(p, segment.arc);
    }
}


inline Pose arc_matrix(const float_type length, const float_type angle, const float_type tilt)
/*
    Get a transformation to the point at `length` along an arc of `angle`.
*/
{
    if (fabs(angle) < APER_PRECISION) {
        // Just a translation in the straight case
        return transform_to_matrix((Transform){ .s = length });
    }

    const float_type ct = cos(tilt), st = sin(tilt);
    const float_type ca = cos(angle), sa = sin(angle);
    const float_type dx = length * (ca - 1) / angle;
    const float_type ds = length * sa / angle;
    return (Pose) {
        .mat = {
            {ct * ca,  -st, -ct * sa,  ct * dx },
            {st * ca,   ct, -st * sa,  st * dx },
            {     sa,  0.f,       ca,       ds },
            {    0.f,  0.f,      0.f,      1.f }
        }
    };
}

#endif  /* XT_APERTURE_PATH_3D_H */
