#ifndef XT_APERTURE_PATH_3D_H
#define XT_APERTURE_PATH_3D_H

#include "base.h"

typedef struct {
    Point3D start;
    Point3D end;
} LineSegment3D;


float_type segment_get_squared_length(LineSegment3D segment) {
    const float_type dx = segment.end.x - segment.start.x;
    const float_type dy = segment.end.y - segment.start.y;
    const float_type dz = segment.end.z - segment.start.z;
    return dx * dx + dy * dy + dz * dz;
}


float_type segment_get_length(LineSegment3D segment) {
    return sqrt(segment_get_squared_length(segment));
}


float_type dist_along_segment_where_plane_intersects(LineSegment3D segment, Pose plane)
/*
    Return a parameter `t` such that `segment.start + t * (segment.end - segment.start)` is the
    point on the segment at which `plane` intersects the `segment`. The plane is defined as a
    pose, such that if the pose is an identity, the plane lies in the X-Y plane (z=0), or, in
    other words, `plane @ [0, 0, 1]^T` is the plane normal.
*/
{
    const float_type eps = APER_PRECISION;

    // Extract translation T from pose (local -> world)
    const float_type t_x = plane.mat[0][3];
    const float_type t_y = plane.mat[1][3];
    const float_type t_z = plane.mat[2][3];

    // Extract 3rd column of rotation R (plane normal in world)
    const float_type n_x = plane.mat[0][2];
    const float_type n_y = plane.mat[1][2];
    const float_type n_z = plane.mat[2][2];

    // Let A := segment.start, compute A - T
    const float_type ta_x = segment.start.x - t_x;
    const float_type ta_y = segment.start.y - t_y;
    const float_type ta_z = segment.start.z - t_z;

    // Let B := segment.end, compute B - T
    const float_type tb_x = segment.end.x - t_x;
    const float_type tb_y = segment.end.y - t_y;
    const float_type tb_z = segment.end.z - t_z;

    /*
        The equation P(t) = A + t * (B - A), t \in [0, 1], defines the line segment, while
        n * (X - T) = 0 defines the plane.
        
        Substituting P(t) in the latter gives n * (A + t * (B - A) - T) = 0, and after rearranging
        n * (A - T) + tn * (B - A) = 0 and then t = - (n * (A - T)) / (n * (B - A)).
    */

    const float_type n_dot_ta = n_x * ta_x + n_y * ta_y + n_z * ta_z;  /* n * (A - T) */
    const float_type n_dot_tb = n_x * tb_x + n_y * tb_y + n_z * tb_z;  /* n * (B - T) */

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


    if (fabs(n_dot_ab) < eps) return NAN;  // The segment and the plane are parallel (co-planar or no intersection)

    return -n_dot_ta / n_dot_ab;
}


inline float_type closest_t_on_segment(const Point3D p, const Point3D a, const Point3D b)
/*
    Closest point parameter t on segment [a,b] to point p.
*/
{
    const float_type eps = APER_PRECISION;
    const Point3D ab = point3d_sub(b, a);
    const Point3D ap = point3d_sub(p, a);
    const float_type segment_length = point3d_dot(ab, ab);

    if (segment_length < eps) return 0;

    const float_type t = point3d_dot(ap, ab) / segment_length;
    return t;
}

#endif  /* XT_APERTURE_PATH_3D_H */