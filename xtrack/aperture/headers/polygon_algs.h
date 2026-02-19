#ifndef POLYGON_ALGS
#define POLYGON_ALGS

#include <math.h>
#include <stdlib.h>

#include "xobjects/headers/common.h"
#include "base.h"
#include "path.h"


typedef struct {
    float_type h;
    float_type v;
    float_type a;
    float_type b;
} Racetrack_s;


typedef struct {
    float_type dist;
    int vertex_idx;
} RayHit_s;


static inline Racetrack_s geom2d_add_racetracks(Racetrack_s rt1, Racetrack_s rt2)
/* Minkowski sum of two racetracks */
{
    return (Racetrack_s){
        .h = rt1.h + rt2.h,
        .v = rt1.v + rt2.v,
        .a = rt1.a + rt2.a,
        .b = rt1.b + rt2.b,
    };
}


static inline Racetrack_s geom2d_scale_racetrack(Racetrack_s rt, float_type scale)
/* Scale a racetrack by a factor ``scale``. */
{
    return (Racetrack_s){
        .h = rt.h * scale,
        .v = rt.v * scale,
        .a = rt.a * scale,
        .b = rt.b * scale,
    };
}


float_type geom2d_racetrack_radius_at_angle(float_type theta, Racetrack_s rt)
/* Compute the length of a line segment going from the center of a racetrack to its boundary at angle ``theta``. */
{
    static const float_type eps = 1e-8;
    const float_type h = rt.h;
    const float_type v = rt.v;
    const float_type a = rt.a;
    const float_type b = rt.b;

    if ((h < eps && a < eps) || (v < eps && b < eps)) return 0;

    /* Reduce to first quadrant */
    const float_type sin_ = fabs(sin(theta));
    const float_type cos_ = fabs(cos(theta));

    const float_type tan_ = sin_ / cos_;

    /* Straight sides */
    if (tan_ <= (v - b) / h) {
        return h / cos_;
    }

    if (tan_ >= v / (h - a)) {
        return v / sin_;
    }

    /* Ellipse corner: solve quadratic in radius */
    const float_type x0 = h - a;
    const float_type y0 = v - b;

    const float_type a_sq = a * a;
    const float_type b_sq = b * b;

    const float_type cos_sq = cos_ * cos_;
    const float_type sin_sq = sin_ * sin_;

    const float_type q_a = cos_sq / a_sq + sin_sq / b_sq;
    const float_type q_b = -2.f * (x0 * cos_ / a_sq + y0 * sin_ / b_sq);
    const float_type q_c = (x0 * x0) / a_sq + (y0 * y0) / b_sq - 1.f;

    const float_type disc = q_b * q_b - 4.f * q_a * q_c;
    const float_type sqrt_disc = sqrt(disc);

    return (sqrt_disc - q_b) / (2.f * q_a);
}


RayHit_s geom2d_dist_to_poly_along_ray(
    float_type theta,
    float_type x0,
    float_type y0,
    const G2DPoint *poly,
    int len_poly,
    int convex,
    int start_at
)
/*  Find the distance from (x0, y0) to the polygon boundary along a ray going in direction theta.

    Parameters
    ----------
    theta: the angle (in radians) of the ray along which to check the distance.
    x0: the x coordinate of the ray origin.
    y0: the y coordinate of the ray origin.
    poly: the polygon, defined as a list of vertices, expected to be closed.
    len_poly: number of vertices of ``poly``
    convex: a flag to be set if the ``poly`` is a convex polygon (short circuits the algorithm).
    start_at the index of the vertex to start at (useful for resuming the algorithm in ``convex=True`` case).

    Returns
    -------
    (RayHit_s){distance, vertex_idx} where:
    - `distance` is the minimum distance from (x0, y0),
    - `vertex_idx` is the index of the vertex hit by the ray.
*/
{
    static const float_type eps = 1e-8;

    const G2DPoint ray = (G2DPoint){ .x = cos(theta), .y = sin(theta) };
    const G2DPoint xy0 = (G2DPoint){ .x = x0, .y = y0 };

    float_type best = INFINITY;
    int best_idx = -1;

    const int n_edges = len_poly - 1;

    for (int k = 0; k < n_edges; k++) {
        /* Rotate the indices to start at ``start_at`` */
        const int i = (start_at + k) % n_edges;

        /* Translate so that (x0, y0) is origin */
        const G2DPoint a = geom2d_sub(poly[i], xy0);
        const G2DPoint b = geom2d_sub(poly[i + 1], xy0);

        const G2DPoint edge = geom2d_sub(b, a);

        const float_type ca = geom2d_cross(a, ray);
        const float_type cb = geom2d_cross(b, ray);
        const float_type delta = cb - ca;  // == e × d (by the distributive property)

        if (fabs(delta) > eps) {
            /* Since `delta` is not zero, the ray is not parallel to the edge

               We can parametrise the edge as follows, where u in [0, 1]:
               f(u) = a + u * (b - a) = (1 - u) * a + u * b. We want to find the
               point where the ray hits the edge-line, i.e. a value of u for which
               f(u) × ray = (1 - u) * (a × ray) + u * (b × ray) = 0. Solving for
               u gives us the following:
            */
            const float_type u = ca / (ca - cb);

            if (u < -eps || (1.f + eps) < u) {
                // If `u` is outside [0, 1], the ray is parallel but does not hit the edge.
                continue;
            }

            const G2DPoint hit = (G2DPoint){ .x = a.x + u * edge.x, .y = a.y + u * edge.y };
            const float_type t = geom2d_dot(hit, ray); // signed distance to the polygon along the ray.

            if (t >= 0 && t < best) {
                best = fabs(t);
                best_idx = i;
                if (!convex) break;
            }
        } else {
            // Parallel case
            if (fabs(ca) > eps || fabs(cb) > eps) {
                // If ca or cb is nonzero, the ray is parallel to the edge but
                // not collinear, so no intersection. Nothing to do here.
                continue;
            }

            // Collinear overlap: get signed distances to endpoints
            const float_type ta = geom2d_dot(a, ray);
            const float_type tb = geom2d_dot(b, ray);

            if (signbit(ta) != signbit(tb)) {
                // Vertices of the edge are on opposite sides of the origin, so
                // the origin is on the edge. Distance is zero.
                best = 0.f;
                best_idx = i;
                break;
            }

            const float_type col_best = fmin(fabs(ta), fabs(tb));
            if (col_best < best) {
                best = col_best;
                best_idx = i;
                if (!convex) break;
            }
        }
    }

    return (RayHit_s){ .dist = best, .vertex_idx = best_idx };
}


void geom2d_dist_to_poly_along_rays(
    const float_type *thetas,
    int len_thetas,
    float_type x0,
    float_type y0,
    const G2DPoint *poly,
    int len_poly,
    int convex,
    float_type *out_dists
)
/*  Find the distance from (x0, y0) to the polygon boundary along rays going in directions thetas.

    Parameters
    ----------
    thetas: array of angles (in radians) at which to check the distance, expected to be sorted.
    len_thetas: length of the array ``thetas``.
    x0: the x coordinate of the ray origin.
    y0: the y coordinate of the ray origin.
    poly: the polygon, defined as a list of vertices, expected to be closed.
    len_poly: number of vertices of ``poly``
    convex: is ``poly`` convex? If true the complexity is reduced from O(len_thetas * len_poly) to O(len_poly).
    out_dists: array of length ``len_thetas`` to which to write corresponding distances.
*/
{
    int start_at = 0;

    for (int j = 0; j < len_thetas; ++j) {
        const float_type theta = thetas[j];

        const RayHit_s hit = geom2d_dist_to_poly_along_ray(theta, x0, y0, poly, len_poly, convex, start_at);

        out_dists[j] = hit.dist;

        if (convex && hit.vertex_idx >= 0) {
            start_at = hit.vertex_idx;
        }
    }
}

#endif /* POLYGON_ALGS */
