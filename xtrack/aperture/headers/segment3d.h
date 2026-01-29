#ifndef XT_APERTURE_SEGMENT3D_H
#define XT_APERTURE_SEGMENT3D_H

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


static inline void plane_to_arc_local(
    const ArcSegment3D segment,
    const Point3D plane_point,
    const Point3D normal,
    Point3D *restrict q,
    Point3D *restrict n_local)
{
    const Pose inv = pose_inverse_rigid(segment.start);

    /* Plane point in arc-local frame */
    *q = pose_apply_point(inv, plane_point);

    /* Plane normal in arc-local frame: R^T n */
    *n_local = (Point3D){
        .x = inv.mat[0][0] * normal.x + inv.mat[0][1] * normal.y + inv.mat[0][2] * normal.z,
        .y = inv.mat[1][0] * normal.x + inv.mat[1][1] * normal.y + inv.mat[1][2] * normal.z,
        .z = inv.mat[2][0] * normal.x + inv.mat[2][1] * normal.y + inv.mat[2][2] * normal.z,
    };
}


float_type arc_segment_plane_intersect(
    const ArcSegment3D segment,
    const Point3D plane_point,
    const Point3D normal)
/*
    Return a parameter `t` such that `arc_segment_point_at(segment, t)` lies on the
    plane defined by `plane_point` and `normal`.

    Return NAN only if the plane does not intersect the supporting circle.
    Otherwise return the solution `t` closest to [0, 1].
*/
{
    const float_type eps = APER_PRECISION;
    const float_type L = segment.length;

    if (L <= eps) return 0.f;

    const float_type h = segment.curvature;

    /* Straight-line limit */
    if (fabs(h) < eps) {
        const LineSegment3D line = {
            .start = arc_segment_point_at(segment, 0.f),
            .end = arc_segment_point_at(segment, 1.f),
        };
        return line_segment_plane_intersect(line, plane_point, normal);
    }

    /*
        Bring the plane into arc-local coordinates.

        In these coordinates, with theta = h L t,
        the arc is

            p(theta) =
            (
                cos(roll) * (cos(theta) - 1) / h,
                sin(roll) * (cos(theta) - 1) / h,
                sin(theta) / h
            )
    */
    Point3D q, n;
    plane_to_arc_local(segment, plane_point, normal, &q, &n);

    /*
        Solve the plane equation in local coordinates:
            n · (p(theta) - q) = 0

        Let:
            alpha = n_x * cos(roll) + n_y * sin(roll)
            beta = n_z
            delta = n \cdot q

        Then we can rewrite the original equation as:
            alpha * (cos(theta) - 1) / h + beta * sin(theta) / h - delta = 0

        Multiply by h:
            alpha * cos(theta) + beta * sin(theta) - (alpha + h * delta) = 0
    */
    const float_type alpha = n.x * cos(segment.roll) + n.y * sin(segment.roll);
    const float_type beta = n.z;
    const float_type delta = point3d_dot(n, q);

    /* Let a := alpha, b := beta, and c := -alpha - h * delta */
    const float_type a = alpha;
    const float_type b = beta;
    const float_type c = -(alpha + h * delta);

    /*
        Using the identity
            a cos(theta) + b sin(theta) = R cos(theta - phi)
        where
            R = hypot(a, b)
            phi = atan2(b, a)

        We can solve the following for phi:
            R cos(theta - phi) + c = 0 => cos(theta - phi) = -c / R
    */
    const float_type R = hypot(a, b);

    if (R < eps) {
        /* Degenerate case */
        if (fabs(c) < eps)
            return 0;  // The whole circle lies in the plane
        else
            return NAN;  // No intersection
    }

    const float_type rhs = -c / R;

    if (rhs < -(1 + eps) || rhs > 1 + eps) {
        // No real solutions
        return NAN;
    }

    const float_type rhs_clamped = clamp_value(rhs, -1, 1);
    const float_type phi = atan2(b, a);
    const float_type gamma = acos(rhs_clamped);

    /*
        Two solution families in theta:
            theta = phi + gamma + 2 * pi * m
            theta = phi - gamma + 2 * pi * m

        Convert with
            theta = h L t
    */
    const float_type theta_scale = h * L;
    const float_type period_t = (2.f * M_PI) / theta_scale;

    const float_type t_base[2] = {
        (phi + gamma) / theta_scale,
        (phi - gamma) / theta_scale
    };

    float_type best_t = NAN;
    float_type best_dist = INFINITY;

    for (int branch = 0; branch < 2; ++branch) {
        const float_type t0 = t_base[branch];

        /* Search copies of this solution family nearest to the interval [0, 1]. */
        const float_type m0 = round((0.f - t0) / period_t);
        const float_type m1 = round((1.f - t0) / period_t);

        const float_type candidates[6] = {
            t0 + (m0 - 1.f) * period_t,
            t0 + m0 * period_t,
            t0 + (m0 + 1.f) * period_t,
            t0 + (m1 - 1.f) * period_t,
            t0 + m1 * period_t,
            t0 + (m1 + 1.f) * period_t
        };

        for (int i = 0; i < 6; ++i) {
            const float_type t = candidates[i];
            float_type dist = 0.f;
            if (t < 0.f) dist = -t;
            else if (t > 1.f) dist = t - 1.f;

            if (dist < best_dist) {
                best_dist = dist;
                best_t = t;
            }
        }
    }

    return best_t;
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

#endif  /* XT_APERTURE_SEGMENT3D_H */
