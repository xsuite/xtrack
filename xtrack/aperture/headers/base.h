#ifndef APERTURE_BASE_H
#define APERTURE_BASE_H

#include <math.h>
#include <stdlib.h>

typedef float float_type;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    float_type x;
    float_type y;
} G2DPoint;

typedef struct {
    float_type x;
    float_type y;
    float_type z;
} G3DPoint;

typedef struct {
    float_type mat[4*4];
} G3DTransform;



float_type geom2d_dot(G2DPoint a, G2DPoint b);
float_type geom2d_cross(G2DPoint a, G2DPoint b);
G2DPoint geom2d_sub(G2DPoint a, G2DPoint b);
float_type geom2d_clamp(float_type t, float_type lo, float_type hi);
float_type geom2d_points_distance(float_type x1, float_type y1, float_type x2, float_type y2);
void geom2d_points_translate(float_type dx, float_type dy, G2DPoint* points, const int len_points);
float_type geom2d_elliptic_E(float_type phi, float_type k);
float_type geom2d_elliptic_E_complete(float_type k);


float_type geom2d_dot(G2DPoint a, G2DPoint b)
{
    return a.x*b.x + a.y*b.y;
}


float_type geom2d_cross(G2DPoint a, G2DPoint b)
{
    return a.x*b.y - a.y*b.x;
}


G2DPoint geom2d_sub(G2DPoint a, G2DPoint b)
{
    G2DPoint r = {a.x-b.x, a.y-b.y};
    return r;
}


float_type geom2d_clamp(float_type t, float_type lo, float_type hi) {
    return (t < lo) ? lo : (t > hi) ? hi : t;
}


float_type geom2d_points_distance(float_type x1, float_type y1, float_type x2, float_type y2)
/* Get the distance between two 2D points */
{
    float_type dx = x2 - x1;
    float_type dy = y2 - y1;
    return hypot(dx, dy);
}


void geom2d_points_translate(float_type dx, float_type dy, G2DPoint* points, const int len_points)
/* Translate the `points` by (dx, dy)

Contract: len_points=len(points)
*/
{
    for (int i = 0; i < len_points; i++) {
        points[i].x += dx;
        points[i].y += dy;
    }
}


static float_type geom2d_elliptic_E_adaptive(float_type a, float_type b, float_type eps, int depth, float_type m)
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


static float_type geom2d_elliptic_E_numeric(float_type phi, float_type k)
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


void merge_sorted(const float_type *restrict a, const float_type *restrict b, int len_a, int len_b, float_type *out, int *out_len)
/* Return array with not repeated ascending values from a and b that are assumed to be sorted

Contract: len_a=len(a); len_b=len(b); len(out)=len_a+len_b; postlen(out)=out_len;
*/
{
    int ia = 0, ib = 0, k = 0;
    float_type last = NAN;
    while (ia < len_a && ib < len_b) {
        float_type va = a[ia];
        float_type vb = b[ib];
        float_type v;
        if (va < vb) {
            v = va;
            ia++;
        } else if (vb < va) {
            v = vb;
            ib++;
        } else {
            v = va;
            ia++;
            ib++;
        }
        if (k == 0 || v != last) {
            out[k++] = v;
            last = v;
        }
    }
    while (ia < len_a) {
        float_type v = a[ia++];
        if (k == 0 || v != last) {
            out[k++] = v;
            last = v;
        }
    }
    while (ib < len_b) {
        float_type v = b[ib++];
        if (k == 0 || v != last) {
            out[k++] = v;
            last = v;
        }
    }
    *out_len = k;
}


float_type sinc(float_type x) {
    if (fabs(x) < 1e-8f) return 1.0f;
    return sin(x) / x;
}


void matrix_multiply_4x4(const float_type a[4][4], const float_type b[4][4], float_type result[4][4]) {
    // Multiply two 4x4 matrices `a` and `b`, and store the result in `result`.
    for (int i = 0; i < 4; i++) {
        result[i][0] = a[i][0] * b[0][0] + a[i][1] * b[1][0] + a[i][2] * b[2][0] + a[i][3] * b[3][0];
        result[i][1] = a[i][0] * b[0][1] + a[i][1] * b[1][1] + a[i][2] * b[2][1] + a[i][3] * b[3][1];
        result[i][2] = a[i][0] * b[0][2] + a[i][1] * b[1][2] + a[i][2] * b[2][2] + a[i][3] * b[3][2];
        result[i][3] = a[i][0] * b[0][3] + a[i][1] * b[1][3] + a[i][2] * b[2][3] + a[i][3] * b[3][3];
    }
}

#endif /* APERTURE_BASE_H */
