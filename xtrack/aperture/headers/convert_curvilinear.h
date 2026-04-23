#ifndef XTRACK_APERTURE_CONVERT_CURVILINEAR_H
#define XTRACK_APERTURE_CONVERT_CURVILINEAR_H

#include "base.h"

inline Point3D curvilinear_to_cartesian_point(const Point3D p_curv, const float_type h)
/*
    Curvilinear (x, y, s) -> Cartesian (X, Y, Z):

        X = (R + x) cos(h s) - R
        Y = y
        Z = (R + x) sin(h s)

    where R = 1 / h.
*/
{
    if (fabs(h) < APER_PRECISION) {
        return (Point3D){
            .x = p_curv.x,
            .y = p_curv.y,
            .z = p_curv.z
        };
    }

    const float_type R = 1.f / h;
    const float_type theta = h * p_curv.z;
    const float_type c = cos(theta);
    const float_type sn = sin(theta);

    return (Point3D){
        .x = (R + p_curv.x) * c - R,
        .y = p_curv.y,
        .z = (R + p_curv.x) * sn
    };
}


inline Point3D cartesian_to_curvilinear_point(const Point3D p_cart, const float_type h)
/*
    Cartesian (X, Y, Z) -> Curvilinear (x, y, s).

    Inverse of curvilinear_to_cartesian_point for the same convention.
*/
{
    if (fabs(h) < APER_PRECISION) {
        return (Point3D){
            .x = p_cart.x,
            .y = p_cart.y,
            .z = p_cart.z
        };
    }

    const float_type R = 1.f / h;

    /*
        X + R = (R + x) cos(theta)
        Z = (R + x) sin(theta)
    */
    const float_type v_x = p_cart.x + R;
    const float_type v_z = p_cart.z;

    /*
        Keep a signed (R + x) convention that is symmetric for positive/negative curvature.
    */
    const float_type sgn = (h >= 0.f) ? 1.f : -1.f;
    const float_type rho = sgn * hypot(v_x, v_z);
    const float_type theta = atan2(sgn * v_z, sgn * v_x);

    return (Point3D){
        .x = rho - R,
        .y = p_cart.y,
        .z = theta * R
    };
}


inline Point3D cartesian_vector_to_curvilinear_at_point(const Point3D p_curv, const Point3D v_cart, const float_type h)
/*
    Convert an attached Cartesian vector to curvilinear components
    using the local inverse Jacobian for the arc_matrix convention.

    Input:
        p_curv: attachment point in curvilinear coordinates (x, y, s)
        v_cart: vector components in Cartesian basis
        h: curvature

    Output:
        vector components in curvilinear basis/coordinates (vx, vy, vs)
*/
{
    if (fabs(h) < APER_PRECISION) {
        return (Point3D){
            .x = v_cart.x,
            .y = v_cart.y,
            .z = v_cart.z
        };
    }

    const float_type theta = h * p_curv.z;
    const float_type c = cos(theta);
    const float_type sn = sin(theta);
    const float_type scale_s = 1.f + h * p_curv.x;

    /* Jacobian singularity at x = -1/h */
    if (fabs(scale_s) < APER_PRECISION) {
        return (Point3D){
            .x = NAN,
            .y = NAN,
            .z = NAN
        };
    }

    return (Point3D){
        .x = c * v_cart.x + sn * v_cart.z,
        .y = v_cart.y,
        .z = (-sn * v_cart.x + c * v_cart.z) / scale_s
    };
}

#endif /* XTRACK_APERTURE_CONVERT_CURVILINEAR_H */
