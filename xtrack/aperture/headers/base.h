#ifndef APERTURE_BASE_H
#define APERTURE_BASE_H

#include <math.h>
#include <stdlib.h>

typedef float float_type;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define APER_PRECISION 1e-6f;

typedef struct {
    float_type x;
    float_type y;
} G2DPoint;

typedef struct {
    float_type x;
    float_type y;
    float_type z;
} Point3D;

typedef struct {
    float_type x;  // shift in x
    float_type y;  // shift in y
    float_type s;  // shift in s
    float_type rot_x;  // rotation around the x-axis, positive s to y (MAD-X phi)
    float_type rot_y;  // rotation around the y-axis, positive s to x (MAD-X theta)
    float_type rot_s;  // rotation around the s-axis, positive y to x (MAD-X psi)
} Transform;

typedef struct {
    float_type mat[4][4];
} Pose;


inline float_type geom2d_dot(G2DPoint, G2DPoint);
inline float_type geom2d_cross(G2DPoint, G2DPoint);
inline G2DPoint geom2d_sub(G2DPoint, G2DPoint);
inline float_type geom2d_clamp(float_type t, float_type lo, float_type hi);
inline float_type geom2d_points_distance(float_type x1, float_type y1, float_type x2, float_type y2);
inline void geom2d_points_translate(float_type dx, float_type dy, G2DPoint* points, const int len_points);

static inline float_type geom2d_elliptic_E_adaptive(float_type a, float_type b, float_type eps, int depth, float_type m);
static inline float_type geom2d_elliptic_E_numeric(float_type phi, float_type k);
float_type geom2d_elliptic_E(float_type phi, float_type k);
float_type geom2d_elliptic_E_complete(float_type k);

inline float_type sinc(float_type);
inline void matrix_multiply_4x4(const float_type a[4][4], const float_type b[4][4], float_type result[4][4]);
inline Pose transform_to_matrix(const Transform);
inline Transform matrix_to_transform(const Pose);
inline Point3D point3d_sub(const Point3D, const Point3D);
inline float_type point3d_dot(const Point3D, const Point3D);
inline Point3D point3d_add_scaled(const Point3D a, const Point3D v, const float_type t);
inline Point3D pose_apply_point(const Pose, const Point3D);


inline float_type geom2d_dot(G2DPoint a, G2DPoint b)
{
    return a.x*b.x + a.y*b.y;
}


inline float_type geom2d_cross(G2DPoint a, G2DPoint b)
{
    return a.x*b.y - a.y*b.x;
}


inline G2DPoint geom2d_sub(G2DPoint a, G2DPoint b)
{
    G2DPoint r = {a.x-b.x, a.y-b.y};
    return r;
}


inline float_type geom2d_clamp(float_type t, float_type lo, float_type hi) {
    return (t < lo) ? lo : (t > hi) ? hi : t;
}


inline float_type geom2d_points_distance(float_type x1, float_type y1, float_type x2, float_type y2)
/* Get the distance between two 2D points */
{
    float_type dx = x2 - x1;
    float_type dy = y2 - y1;
    return hypot(dx, dy);
}


inline void geom2d_points_translate(float_type dx, float_type dy, G2DPoint* points, const int len_points)
/* Translate the `points` by (dx, dy)

Contract: len_points=len(points)
*/
{
    for (int i = 0; i < len_points; i++) {
        points[i].x += dx;
        points[i].y += dy;
    }
}


static inline float_type geom2d_elliptic_E_adaptive(float_type a, float_type b, float_type eps, int depth, float_type m)
/* Incomplete elliptic integral of the second kind E(phi, k)
   Uses libm ellint_2/comp_ellint_2 when available (glibc),
   otherwise falls back to a small adaptive Simpson integrator.
*/
{
    float_type c = 0.5 * (a + b);
    float_type fa = sqrt(1.0 - m * sin(a) * sin(a));
    float_type fb = sqrt(1.0 - m * sin(b) * sin(b));
    float_type fc = sqrt(1.0 - m * sin(c) * sin(c));
    float_type h = b - a;
    float_type S = (fa + 4.0 * fc + fb) * h / 6.0;
    float_type left_c = 0.5 * (a + c);
    float_type right_c = 0.5 * (c + b);
    float_type f_left_c = sqrt(1.0 - m * sin(left_c) * sin(left_c));
    float_type f_right_c = sqrt(1.0 - m * sin(right_c) * sin(right_c));
    float_type Sleft = (fa + 4.0 * f_left_c + fc) * (h / 2.0) / 6.0;
    float_type Sright = (fc + 4.0 * f_right_c + fb) * (h / 2.0) / 6.0;
    if (depth <= 0 || fabs(Sleft + Sright - S) < 15.0 * eps)
        return Sleft + Sright + (Sleft + Sright - S) / 15.0;
    return geom2d_elliptic_E_adaptive(a, c, eps / 2.0, depth - 1, m) +
           geom2d_elliptic_E_adaptive(c, b, eps / 2.0, depth - 1, m);
}


static inline float_type geom2d_elliptic_E_numeric(float_type phi, float_type k)
{
    float_type m = k * k;
    float_type sign = (phi >= 0.0) ? 1.0 : -1.0;
    float_type abs_phi = fabs(phi);
    float_type period = M_PI; /* integrand is periodic in pi */
    float_type half_pi = 0.5 * M_PI;
    float_type base_complete = geom2d_elliptic_E_adaptive(0.0, half_pi, 1e-10, 12, m);
    float_type per_value = 2.0 * base_complete; /* integral over one period */
    long periods = (long)(abs_phi / period);
    float_type remainder = abs_phi - periods * period;
    float_type total = periods * per_value;
    if (remainder > 0.0)
        total += geom2d_elliptic_E_adaptive(0.0, remainder, 1e-10, 12, m);
    return sign * total;
}


float_type geom2d_elliptic_E(float_type phi, float_type k)
{
    if (k < 0.0 || k >= 1.0)
        return NAN;
    return geom2d_elliptic_E_numeric(phi, k);
}


float_type geom2d_elliptic_E_complete(float_type k)
{
    if (k < 0.0 || k >= 1.0)
        return NAN;
    return geom2d_elliptic_E_numeric(0.5 * M_PI, k);
}


inline float_type sinc(float_type x) {
    if (fabs(x) < 1e-8f) return 1.0f;
    return sin(x) / x;
}


inline void matrix_multiply_4x4(const float_type a[4][4], const float_type b[4][4], float_type result[4][4]) {
    // Multiply two 4x4 matrices `a` and `b`, and store the result in `result`.
    for (int i = 0; i < 4; i++) {
        result[i][0] = a[i][0] * b[0][0] + a[i][1] * b[1][0] + a[i][2] * b[2][0] + a[i][3] * b[3][0];
        result[i][1] = a[i][0] * b[0][1] + a[i][1] * b[1][1] + a[i][2] * b[2][1] + a[i][3] * b[3][1];
        result[i][2] = a[i][0] * b[0][2] + a[i][1] * b[1][2] + a[i][2] * b[2][2] + a[i][3] * b[3][2];
        result[i][3] = a[i][0] * b[0][3] + a[i][1] * b[1][3] + a[i][2] * b[2][3] + a[i][3] * b[3][3];
    }
}


static inline Pose pose_inverse_rigid(const Pose p)
/* Invert a rigid transform. */
{
    Pose inv;

    /* Transpose rotation */
    for (uint8_t i = 0; i < 3; i++) {
        for (uint8_t j = 0; j < 3; j++) {
            inv.mat[i][j] = p.mat[j][i];
        }
    }

    /* Compute -R^T t */
    const float_type tx = p.mat[0][3];
    const float_type ty = p.mat[1][3];
    const float_type tz = p.mat[2][3];

    inv.mat[0][3] = -(inv.mat[0][0] * tx + inv.mat[0][1] * ty + inv.mat[0][2] * tz);
    inv.mat[1][3] = -(inv.mat[1][0] * tx + inv.mat[1][1] * ty + inv.mat[1][2] * tz);
    inv.mat[2][3] = -(inv.mat[2][0] * tx + inv.mat[2][1] * ty + inv.mat[2][2] * tz);

    inv.mat[3][0] = 0.f;
    inv.mat[3][1] = 0.f;
    inv.mat[3][2] = 0.f;
    inv.mat[3][3] = 1.f;

    return inv;
}



inline Pose transform_to_matrix(const Transform t)
{
    const float_type s_phi = sin(t.rot_x);
    const float_type c_phi = cos(t.rot_x);
    const float_type s_theta = sin(t.rot_y);
    const float_type c_theta = cos(t.rot_y);
    const float_type s_psi = sin(t.rot_s);
    const float_type c_psi = cos(t.rot_s);

    return (Pose) {
        .mat = {
            {
                -s_phi * s_psi * s_theta + c_psi * c_theta,
                -c_psi * s_phi * s_theta - c_theta * s_psi,
                c_phi * s_theta,
                t.x
            },
            {
                c_phi * s_psi,
                c_phi * c_psi,
                s_phi,
                t.y
            },
            {
                -c_theta * s_phi * s_psi - c_psi * s_theta,
                -c_psi * c_theta * s_phi + s_psi * s_theta,
                c_phi * c_theta,
                t.s
            },
            {0, 0, 0, 1}
        }
    };
}


inline Transform matrix_to_transform(const Pose m) {
    return (Transform) {
        .x = m.mat[0][3],
        .y = m.mat[1][3],
        .s = m.mat[2][3],
        .rot_x = atan2(m.mat[1][2], sqrt(m.mat[1][0] * m.mat[1][0] + m.mat[1][1] * m.mat[1][1])),
        .rot_y = atan2(m.mat[0][2], m.mat[2][2]),
        .rot_s = atan2(m.mat[1][0], m.mat[1][1])
    };
}


inline Point3D point3d_sub(const Point3D a, const Point3D b)
{
    return (Point3D){ a.x - b.x, a.y - b.y, a.z - b.z };
}


inline float_type point3d_dot(const Point3D a, const Point3D b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


inline Point3D point3d_add_scaled(const Point3D a, const Point3D v, const float_type t)
{
    return (Point3D){
        a.x + t * v.x,
        a.y + t * v.y,
        a.z + t * v.z
    };
}


inline Point3D pose_apply_point(const Pose p, const Point3D v)
/* Apply a Pose matrix to a 3D point */
{
    const float_type x = p.mat[0][0] * v.x + p.mat[0][1] * v.y + p.mat[0][2] * v.z + p.mat[0][3];
    const float_type y = p.mat[1][0] * v.x + p.mat[1][1] * v.y + p.mat[1][2] * v.z + p.mat[1][3];
    const float_type z = p.mat[2][0] * v.x + p.mat[2][1] * v.y + p.mat[2][2] * v.z + p.mat[2][3];
    return (Point3D){ x, y, z };
}

#endif /* APERTURE_BASE_H */
