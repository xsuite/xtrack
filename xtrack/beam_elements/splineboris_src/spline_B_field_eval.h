#include <stddef.h>
#include <stdio.h>
#include <string.h>
#define MAX_DEGREE 4

#ifndef SPLINE_B_FIELD_EVAL_H
#define SPLINE_B_FIELD_EVAL_H

// Auto-generated symbolic field expressions for B
// NOTE:
//   - 's' is the local coordinate within the element: s_local ∈ [0, L].
//   - Hermite coefficients are defined on s_local ∈ [0, L] and are converted
//     internally to polynomials in s_local via hermite_to_polynomial(0, L, ...).
//
// Hermite input layout
// --------------------
//   - bs        : one scalar Hermite polynomial (5 coeffs) for bs(s_local)
//   - by[i]     : Hermite coeffs (5) for polynomial group by_i_*(s_local)
//   - bx[i]     : Hermite coeffs (5) for polynomial group bx_i_*(s_local)
//
// For multipole_order = n (1 ≤ n ≤ 7):
//   - bs:       1 polynomial      → bs_0..bs_4 from bs
//   - by:       n polynomials     → by_i_0..by_i_4 from by[i], i=0..n-1
//   - bx:       n polynomials     → bx_i_0..bx_i_4 from bx[i], i=0..n-1
//
// The symbolic expressions below are unchanged; only the way the bs_*, by_*_*,
// and bx_*_* scalars are populated has been refactored to use Hermite data.
typedef struct {
	double coeffs[MAX_DEGREE + 1]; /* coeffs[i] = coefficient of x^i */
	int degree;
} Poly;

static inline Poly poly_scale(Poly p, double s) {
	for (int i = 0; i <= p.degree; i++) p.coeffs[i] *= s;
	return p;
}

static inline Poly poly_add(Poly a, Poly b) {
	Poly result = {0};
	result.degree = a.degree > b.degree ? a.degree : b.degree;
	for (int i = 0; i <= a.degree; i++) result.coeffs[i] += a.coeffs[i];
	for (int i = 0; i <= b.degree; i++) result.coeffs[i] += b.coeffs[i];
	return result;
}

static inline Poly poly_mul(Poly a, Poly b) {
	Poly result = {0};
	int deg = a.degree + b.degree;
	if (deg > MAX_DEGREE)
		deg = MAX_DEGREE;
	result.degree = deg;
	for (int i = 0; i <= a.degree; i++) {
		for (int j = 0; j <= b.degree; j++) {
			int k = i + j;
			if (k <= MAX_DEGREE)
				result.coeffs[k] += a.coeffs[i] * b.coeffs[j];
		}
	}
	return result;
}

/* Compose f(g(x)) via Horner's method:
   result = f[n] * g^n + ... + f[0]
		  = f[0] + g*(f[1] + g*(f[2] + ... + g*f[n]))  */
static inline Poly poly_compose(Poly f, Poly g) {
	Poly result = {0};
	result.coeffs[0] = f.coeffs[f.degree]; /* start with leading coeff */
	result.degree = 0;
	for (int i = f.degree - 1; i >= 0; i--) {
		result = poly_mul(result, g);       /* result = result * g      */
		if (result.degree < MAX_DEGREE) {
			result.degree++;
		}
		result.coeffs[0] += f.coeffs[i];   /* result = result * g + f[i] */
	}
	return result;
}

static inline Poly hermite_to_polynomial(double s_start, double s_end, const double coeffs[5]) {
	double c1 = coeffs[0], c2 = coeffs[1], c3 = coeffs[2];
	double c4 = coeffs[3], c5 = coeffs[4];
	double L = s_end - s_start;

	/* t(s_local) = s_local / L */
	Poly t = { .coeffs = {0.0, 1.0/L}, .degree = 1 };

	/* Hermite basis polynomials in t on [0,1] */
	Poly b1 = { .coeffs = { 1,  0,  -18,   32,  -15}, .degree = 4 };
	Poly b2 = { .coeffs = { 0,  1, -4.5,    6, -2.5}, .degree = 4 };
	Poly b3 = { .coeffs = { 0,  0,  -12,   28,  -15}, .degree = 4 };
	Poly b4 = { .coeffs = { 0,  0,  1.5,   -4,  2.5}, .degree = 4 };
	Poly b5 = { .coeffs = { 0,  0,   30,  -60,   30}, .degree = 4 };

	/* poly_t = c1*b1 + L*c2*b2 + c3*b3 + L*c4*b4 + c5*b5 */
	Poly poly_t = {0};
	poly_t = poly_add(poly_t, poly_scale(b1, c1));
	poly_t = poly_add(poly_t, poly_scale(b2, L * c2));
	poly_t = poly_add(poly_t, poly_scale(b3, c3));
	poly_t = poly_add(poly_t, poly_scale(b4, L * c4));
	poly_t = poly_add(poly_t, poly_scale(b5, c5));

	/* poly_s(s_local) = poly_t(t(s_local)) */
	return poly_compose(poly_t, t);
}

GPUFUN
void evaluate_B(const double x, const double y, const double s,
                const double *bs,
                const double *const *by,
                const double *const *bx,
                const double L,
                const int multipole_order,
                double *Bx_out, double *By_out, double *Bs_out){

	switch (multipole_order) {
	case 1: {
		// Hermite → polynomial coefficients (order 1)
		const Poly bs_poly = hermite_to_polynomial(0.0, L, bs);
		const double bs_0   = bs_poly.coeffs[0];
		const double bs_1   = bs_poly.coeffs[1];
		const double bs_2   = bs_poly.coeffs[2];
		const double bs_3   = bs_poly.coeffs[3];
		const double bs_4   = bs_poly.coeffs[4];

		const Poly by0_poly = hermite_to_polynomial(0.0, L, by[0]);
		const double by_0_0 = by0_poly.coeffs[0];
		const double by_0_1 = by0_poly.coeffs[1];
		const double by_0_2 = by0_poly.coeffs[2];
		const double by_0_3 = by0_poly.coeffs[3];
		const double by_0_4 = by0_poly.coeffs[4];

		const Poly bx0_poly = hermite_to_polynomial(0.0, L, bx[0]);
		const double bx_0_0 = bx0_poly.coeffs[0];
		const double bx_0_1 = bx0_poly.coeffs[1];
		const double bx_0_2 = bx0_poly.coeffs[2];
		const double bx_0_3 = bx0_poly.coeffs[3];
		const double bx_0_4 = bx0_poly.coeffs[4];

		// Common sub-expressions
		const double x0 = s*s;
		const double x1 = s*s*s;
		const double x2 = s*s*s*s;
		const double x3 = y*y*y*y;
		const double x4 = y*y;
		const double x5 = 3*s;
		const double x6 = 6*x0;
		const double x7 = bx_0_2 + bx_0_3*x5 + bx_0_4*x6;
		const double x8 = y*y*y;
		const double x9 = 4*bs_4;
		const double x10 = 2*s;
		const double x11 = 3*x0;
		const double x12 = 4*s;
		const double x13 = 4*x1;

		// Reduced expressions
		*Bx_out = bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2 + bx_0_4*x3 - x4*x7;
		*By_out = by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 + by_0_4*x3 - x4*(by_0_2 + by_0_3*x5 + by_0_4*x6) + (1.0/6.0)*x8*(6*bs_3 + 24*bx_0_4*x + 6*s*x9) - y*(bs_1 + bs_2*x10 + bs_3*x11 + 2*x*x7 + x1*x9);
		*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2 + bs_4*x3 + x*(bx_0_1 + bx_0_2*x10 + bx_0_3*x11 + bx_0_4*x13) - 1.0/2.0*x4*(2*bs_2 + 2*bs_3*x5 + 2*bs_4*x6 + 6*x*(bx_0_3 + bx_0_4*x12)) - x8*(by_0_3 + by_0_4*x12) + y*(by_0_1 + by_0_2*x10 + by_0_3*x11 + by_0_4*x13);
		return;

	}
	case 2: {
		// Hermite → polynomial coefficients (order 2)
		const Poly bs_poly = hermite_to_polynomial(0.0, L, bs);
		const double bs_0   = bs_poly.coeffs[0];
		const double bs_1   = bs_poly.coeffs[1];
		const double bs_2   = bs_poly.coeffs[2];
		const double bs_3   = bs_poly.coeffs[3];
		const double bs_4   = bs_poly.coeffs[4];

		const Poly by0_poly = hermite_to_polynomial(0.0, L, by[0]);
		const double by_0_0 = by0_poly.coeffs[0];
		const double by_0_1 = by0_poly.coeffs[1];
		const double by_0_2 = by0_poly.coeffs[2];
		const double by_0_3 = by0_poly.coeffs[3];
		const double by_0_4 = by0_poly.coeffs[4];

		const Poly by1_poly = hermite_to_polynomial(0.0, L, by[1]);
		const double by_1_0 = by1_poly.coeffs[0];
		const double by_1_1 = by1_poly.coeffs[1];
		const double by_1_2 = by1_poly.coeffs[2];
		const double by_1_3 = by1_poly.coeffs[3];
		const double by_1_4 = by1_poly.coeffs[4];

		const Poly bx0_poly = hermite_to_polynomial(0.0, L, bx[0]);
		const double bx_0_0 = bx0_poly.coeffs[0];
		const double bx_0_1 = bx0_poly.coeffs[1];
		const double bx_0_2 = bx0_poly.coeffs[2];
		const double bx_0_3 = bx0_poly.coeffs[3];
		const double bx_0_4 = bx0_poly.coeffs[4];

		const Poly bx1_poly = hermite_to_polynomial(0.0, L, bx[1]);
		const double bx_1_0 = bx1_poly.coeffs[0];
		const double bx_1_1 = bx1_poly.coeffs[1];
		const double bx_1_2 = bx1_poly.coeffs[2];
		const double bx_1_3 = bx1_poly.coeffs[3];
		const double bx_1_4 = bx1_poly.coeffs[4];

		// Common sub-expressions
		const double x0 = s*s;
		const double x1 = s*s*s;
		const double x2 = s*s*s*s;
		const double x3 = y*y*y*y*y;
		const double x4 = 48*bx_0_4;
		const double x5 = y*y*y*y;
		const double x6 = (1.0/48.0)*x5;
		const double x7 = y*y*y;
		const double x8 = 3*s;
		const double x9 = 6*x0;
		const double x10 = by_1_2 + by_1_3*x8 + by_1_4*x9;
		const double x11 = bx_1_1*s;
		const double x12 = bx_1_2*x0;
		const double x13 = bx_1_3*x1;
		const double x14 = bx_1_4*x2;
		const double x15 = by_1_0 + by_1_1*s + by_1_2*x0 + by_1_3*x1 + by_1_4*x2;
		const double x16 = bx_0_2 + bx_0_3*x8 + bx_0_4*x9;
		const double x17 = bx_1_2 + bx_1_3*x8 + bx_1_4*x9;
		const double x18 = 4*x;
		const double x19 = y*y;
		const double x20 = (1.0/4.0)*x19;
		const double x21 = x*x;
		const double x22 = 4*s;
		const double x23 = 8*x1;
		const double x24 = bx_1_3 + bx_1_4*x22;
		const double x25 = 2*s;
		const double x26 = 3*x0;
		const double x27 = 4*x1;

		// Reduced expressions
		*Bx_out = bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2 + (1.0/5.0)*by_1_4*x3 + x*(bx_1_0 + x11 + x12 + x13 + x14) - 1.0/3.0*x10*x7 + x15*y - x20*(4*x16 + x17*x18) + x6*(48*bx_1_4*x + x4);
		*By_out = -3.0/5.0*bx_1_4*x3 + by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 + x*x15 - 1.0/2.0*x19*(2*by_0_2 + 2*by_0_3*x8 + 2*by_0_4*x9 + 2*x*x10) + (1.0/24.0)*x5*(24*by_0_4 + 24*by_1_4*x) + (1.0/12.0)*x7*(12*bs_3 + 12*bs_4*x22 + 24*bx_1_4*x21 + x*x4 + 8*x17) - 1.0/2.0*y*(2*bs_1 + bs_2*x22 + bs_3*x9 + bs_4*x23 + 2*bx_1_0 + 2*x11 + 2*x12 + 2*x13 + 2*x14 + x16*x18 + 2*x17*x21);
		*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2 + x*(bx_0_1 + bx_0_2*x25 + bx_0_3*x26 + bx_0_4*x27) - x20*(4*bs_2 + 4*bs_3*x8 + 4*bs_4*x9 + 2*bx_1_1 + bx_1_2*x22 + bx_1_3*x9 + bx_1_4*x23 + 12*x*(bx_0_3 + bx_0_4*x22) + 6*x21*x24) + (1.0/2.0)*x21*(bx_1_1 + bx_1_2*x25 + bx_1_3*x26 + bx_1_4*x27) + x6*(48*bs_4 + 24*x24) - 1.0/6.0*x7*(6*by_0_3 + 6*by_0_4*x22 + 6*x*(by_1_3 + by_1_4*x22)) + y*(by_0_1 + by_0_2*x25 + by_0_3*x26 + by_0_4*x27 + x*(by_1_1 + by_1_2*x25 + by_1_3*x26 + by_1_4*x27));
		return;

	}
	case 3: {
		// Hermite → polynomial coefficients (order 3)
		const Poly bs_poly = hermite_to_polynomial(0.0, L, bs);
		const double bs_0   = bs_poly.coeffs[0];
		const double bs_1   = bs_poly.coeffs[1];
		const double bs_2   = bs_poly.coeffs[2];
		const double bs_3   = bs_poly.coeffs[3];
		const double bs_4   = bs_poly.coeffs[4];

		const Poly by0_poly = hermite_to_polynomial(0.0, L, by[0]);
		const double by_0_0 = by0_poly.coeffs[0];
		const double by_0_1 = by0_poly.coeffs[1];
		const double by_0_2 = by0_poly.coeffs[2];
		const double by_0_3 = by0_poly.coeffs[3];
		const double by_0_4 = by0_poly.coeffs[4];

		const Poly by1_poly = hermite_to_polynomial(0.0, L, by[1]);
		const double by_1_0 = by1_poly.coeffs[0];
		const double by_1_1 = by1_poly.coeffs[1];
		const double by_1_2 = by1_poly.coeffs[2];
		const double by_1_3 = by1_poly.coeffs[3];
		const double by_1_4 = by1_poly.coeffs[4];

		const Poly by2_poly = hermite_to_polynomial(0.0, L, by[2]);
		const double by_2_0 = by2_poly.coeffs[0];
		const double by_2_1 = by2_poly.coeffs[1];
		const double by_2_2 = by2_poly.coeffs[2];
		const double by_2_3 = by2_poly.coeffs[3];
		const double by_2_4 = by2_poly.coeffs[4];

		const Poly bx0_poly = hermite_to_polynomial(0.0, L, bx[0]);
		const double bx_0_0 = bx0_poly.coeffs[0];
		const double bx_0_1 = bx0_poly.coeffs[1];
		const double bx_0_2 = bx0_poly.coeffs[2];
		const double bx_0_3 = bx0_poly.coeffs[3];
		const double bx_0_4 = bx0_poly.coeffs[4];

		const Poly bx1_poly = hermite_to_polynomial(0.0, L, bx[1]);
		const double bx_1_0 = bx1_poly.coeffs[0];
		const double bx_1_1 = bx1_poly.coeffs[1];
		const double bx_1_2 = bx1_poly.coeffs[2];
		const double bx_1_3 = bx1_poly.coeffs[3];
		const double bx_1_4 = bx1_poly.coeffs[4];

		const Poly bx2_poly = hermite_to_polynomial(0.0, L, bx[2]);
		const double bx_2_0 = bx2_poly.coeffs[0];
		const double bx_2_1 = bx2_poly.coeffs[1];
		const double bx_2_2 = bx2_poly.coeffs[2];
		const double bx_2_3 = bx2_poly.coeffs[3];
		const double bx_2_4 = bx2_poly.coeffs[4];

		// Common sub-expressions
		const double x0 = s*s;
		const double x1 = s*s*s;
		const double x2 = s*s*s*s;
		const double x3 = (1.0/10.0)*y*y*y*y*y*y;
		const double x4 = y*y*y*y*y;
		const double x5 = 48*by_1_4;
		const double x6 = bx_1_1*s;
		const double x7 = bx_1_2*x0;
		const double x8 = bx_1_3*x1;
		const double x9 = bx_1_4*x2;
		const double x10 = bx_2_1*s;
		const double x11 = bx_2_2*x0;
		const double x12 = bx_2_3*x1;
		const double x13 = bx_2_4*x2;
		const double x14 = bx_2_0 + x10 + x11 + x12 + x13;
		const double x15 = x*x;
		const double x16 = (1.0/2.0)*x15;
		const double x17 = 144*bx_0_4;
		const double x18 = 72*x15;
		const double x19 = 3*s;
		const double x20 = 6*x0;
		const double x21 = bx_2_2 + bx_2_3*x19 + bx_2_4*x20;
		const double x22 = y*y*y*y;
		const double x23 = (1.0/144.0)*x22;
		const double x24 = by_1_2 + by_1_3*x19 + by_1_4*x20;
		const double x25 = by_2_2 + by_2_3*x19 + by_2_4*x20;
		const double x26 = 4*x;
		const double x27 = y*y*y;
		const double x28 = (1.0/12.0)*x27;
		const double x29 = by_1_1*s;
		const double x30 = by_1_2*x0;
		const double x31 = by_1_3*x1;
		const double x32 = by_1_4*x2;
		const double x33 = by_2_1*s;
		const double x34 = by_2_2*x0;
		const double x35 = by_2_3*x1;
		const double x36 = by_2_4*x2;
		const double x37 = by_2_0 + x33 + x34 + x35 + x36;
		const double x38 = 2*x;
		const double x39 = (1.0/2.0)*y;
		const double x40 = bx_0_2 + bx_0_3*x19 + bx_0_4*x20;
		const double x41 = bx_1_2 + bx_1_3*x19 + bx_1_4*x20;
		const double x42 = 12*x;
		const double x43 = 6*x15;
		const double x44 = y*y;
		const double x45 = (1.0/12.0)*x44;
		const double x46 = x*x*x;
		const double x47 = 4*s;
		const double x48 = 12*s;
		const double x49 = 18*x0;
		const double x50 = 24*x1;
		const double x51 = 6*x;
		const double x52 = by_2_3 + by_2_4*x47;
		const double x53 = 2*s;
		const double x54 = 3*x0;
		const double x55 = 4*x1;
		const double x56 = bx_2_1 + bx_2_2*x53 + bx_2_3*x54 + bx_2_4*x55;
		const double x57 = bx_1_3 + bx_1_4*x47;
		const double x58 = bx_2_3 + bx_2_4*x47;
		const double x59 = 8*x1;

		// Reduced expressions
		*Bx_out = bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2 - bx_2_4*x3 + x*(bx_1_0 + x6 + x7 + x8 + x9) + x14*x16 + x23*(144*bx_1_4*x + bx_2_4*x18 + x17 + 24*x21) - x28*(4*x24 + x25*x26) + x39*(2*by_1_0 + 2*x29 + 2*x30 + 2*x31 + 2*x32 + x37*x38) + (1.0/240.0)*x4*(48*by_2_4*x + x5) - x45*(6*bx_2_0 + 6*x10 + 6*x11 + 6*x12 + 6*x13 + x21*x43 + 12*x40 + x41*x42);
		*By_out = by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 - by_2_4*x3 + x*(by_1_0 + x29 + x30 + x31 + x32) + x16*x37 + (1.0/48.0)*x22*(48*by_0_4 + 24*by_2_4*x15 + x*x5 + 8*x25) + (1.0/36.0)*x27*(36*bs_3 + 36*bs_4*x47 + bx_1_4*x18 + 24*bx_2_4*x46 + x*x17 + 24*x*x21 + 24*x41) - 1.0/720.0*x4*(432*bx_1_4 + 432*bx_2_4*x) - 1.0/4.0*x44*(4*by_0_2 + 4*by_0_3*x19 + 4*by_0_4*x20 + 2*by_2_0 + 2*x15*x25 + x24*x26 + 2*x33 + 2*x34 + 2*x35 + 2*x36) - 1.0/6.0*y*(6*bs_1 + bs_2*x48 + bs_3*x49 + bs_4*x50 + 6*bx_1_0 + x14*x51 + 2*x21*x46 + x40*x42 + x41*x43 + 6*x6 + 6*x7 + 6*x8 + 6*x9);
		*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2 + x*(bx_0_1 + bx_0_2*x53 + bx_0_3*x54 + bx_0_4*x55) + x16*(bx_1_1 + bx_1_2*x53 + bx_1_3*x54 + bx_1_4*x55) + x23*(144*bs_4 + 72*x*x58 + 72*x57) - x28*(12*by_0_3 + 12*by_0_4*x47 + 2*by_2_1 + by_2_2*x47 + by_2_3*x20 + by_2_4*x59 + x42*(by_1_3 + by_1_4*x47) + x43*x52) + x39*(2*by_0_1 + by_0_2*x47 + by_0_3*x20 + by_0_4*x59 + x15*(by_2_1 + by_2_2*x53 + by_2_3*x54 + by_2_4*x55) + x38*(by_1_1 + by_1_2*x53 + by_1_3*x54 + by_1_4*x55)) + (1.0/10.0)*x4*x52 - x45*(12*bs_2 + 12*bs_3*x19 + 12*bs_4*x20 + 6*bx_1_1 + bx_1_2*x48 + bx_1_3*x49 + bx_1_4*x50 + 36*x*(bx_0_3 + bx_0_4*x47) + 18*x15*x57 + 6*x46*x58 + x51*x56) + (1.0/6.0)*x46*x56;
		return;

	}
	case 4: {
		// Hermite → polynomial coefficients (order 4)
		const Poly bs_poly = hermite_to_polynomial(0.0, L, bs);
		const double bs_0   = bs_poly.coeffs[0];
		const double bs_1   = bs_poly.coeffs[1];
		const double bs_2   = bs_poly.coeffs[2];
		const double bs_3   = bs_poly.coeffs[3];
		const double bs_4   = bs_poly.coeffs[4];

		const Poly by0_poly = hermite_to_polynomial(0.0, L, by[0]);
		const double by_0_0 = by0_poly.coeffs[0];
		const double by_0_1 = by0_poly.coeffs[1];
		const double by_0_2 = by0_poly.coeffs[2];
		const double by_0_3 = by0_poly.coeffs[3];
		const double by_0_4 = by0_poly.coeffs[4];

		const Poly by1_poly = hermite_to_polynomial(0.0, L, by[1]);
		const double by_1_0 = by1_poly.coeffs[0];
		const double by_1_1 = by1_poly.coeffs[1];
		const double by_1_2 = by1_poly.coeffs[2];
		const double by_1_3 = by1_poly.coeffs[3];
		const double by_1_4 = by1_poly.coeffs[4];

		const Poly by2_poly = hermite_to_polynomial(0.0, L, by[2]);
		const double by_2_0 = by2_poly.coeffs[0];
		const double by_2_1 = by2_poly.coeffs[1];
		const double by_2_2 = by2_poly.coeffs[2];
		const double by_2_3 = by2_poly.coeffs[3];
		const double by_2_4 = by2_poly.coeffs[4];

		const Poly by3_poly = hermite_to_polynomial(0.0, L, by[3]);
		const double by_3_0 = by3_poly.coeffs[0];
		const double by_3_1 = by3_poly.coeffs[1];
		const double by_3_2 = by3_poly.coeffs[2];
		const double by_3_3 = by3_poly.coeffs[3];
		const double by_3_4 = by3_poly.coeffs[4];

		const Poly bx0_poly = hermite_to_polynomial(0.0, L, bx[0]);
		const double bx_0_0 = bx0_poly.coeffs[0];
		const double bx_0_1 = bx0_poly.coeffs[1];
		const double bx_0_2 = bx0_poly.coeffs[2];
		const double bx_0_3 = bx0_poly.coeffs[3];
		const double bx_0_4 = bx0_poly.coeffs[4];

		const Poly bx1_poly = hermite_to_polynomial(0.0, L, bx[1]);
		const double bx_1_0 = bx1_poly.coeffs[0];
		const double bx_1_1 = bx1_poly.coeffs[1];
		const double bx_1_2 = bx1_poly.coeffs[2];
		const double bx_1_3 = bx1_poly.coeffs[3];
		const double bx_1_4 = bx1_poly.coeffs[4];

		const Poly bx2_poly = hermite_to_polynomial(0.0, L, bx[2]);
		const double bx_2_0 = bx2_poly.coeffs[0];
		const double bx_2_1 = bx2_poly.coeffs[1];
		const double bx_2_2 = bx2_poly.coeffs[2];
		const double bx_2_3 = bx2_poly.coeffs[3];
		const double bx_2_4 = bx2_poly.coeffs[4];

		const Poly bx3_poly = hermite_to_polynomial(0.0, L, bx[3]);
		const double bx_3_0 = bx3_poly.coeffs[0];
		const double bx_3_1 = bx3_poly.coeffs[1];
		const double bx_3_2 = bx3_poly.coeffs[2];
		const double bx_3_3 = bx3_poly.coeffs[3];
		const double bx_3_4 = bx3_poly.coeffs[4];

		// Common sub-expressions
		const double x0 = s*s;
		const double x1 = s*s*s;
		const double x2 = s*s*s*s;
		const double x3 = y*y*y*y*y*y*y;
		const double x4 = y*y*y*y*y*y;
		const double x5 = 1728*bx_2_4;
		const double x6 = bx_1_1*s;
		const double x7 = bx_1_2*x0;
		const double x8 = bx_1_3*x1;
		const double x9 = bx_1_4*x2;
		const double x10 = bx_2_1*s;
		const double x11 = bx_2_2*x0;
		const double x12 = bx_2_3*x1;
		const double x13 = bx_2_4*x2;
		const double x14 = bx_2_0 + x10 + x11 + x12 + x13;
		const double x15 = x*x;
		const double x16 = (1.0/2.0)*x15;
		const double x17 = bx_3_1*s;
		const double x18 = bx_3_2*x0;
		const double x19 = bx_3_3*x1;
		const double x20 = bx_3_4*x2;
		const double x21 = bx_3_0 + x17 + x18 + x19 + x20;
		const double x22 = x*x*x;
		const double x23 = (1.0/6.0)*x22;
		const double x24 = 144*by_1_4;
		const double x25 = 144*x;
		const double x26 = 72*x15;
		const double x27 = 3*s;
		const double x28 = 6*x0;
		const double x29 = by_3_2 + by_3_3*x27 + by_3_4*x28;
		const double x30 = y*y*y*y*y;
		const double x31 = (1.0/720.0)*x30;
		const double x32 = 576*bx_0_4;
		const double x33 = 288*x15;
		const double x34 = 96*x22;
		const double x35 = bx_2_2 + bx_2_3*x27 + bx_2_4*x28;
		const double x36 = bx_3_2 + bx_3_3*x27 + bx_3_4*x28;
		const double x37 = 96*x;
		const double x38 = y*y*y*y;
		const double x39 = (1.0/576.0)*x38;
		const double x40 = by_1_1*s;
		const double x41 = by_1_2*x0;
		const double x42 = by_1_3*x1;
		const double x43 = by_1_4*x2;
		const double x44 = by_2_1*s;
		const double x45 = by_2_2*x0;
		const double x46 = by_2_3*x1;
		const double x47 = by_2_4*x2;
		const double x48 = by_2_0 + x44 + x45 + x46 + x47;
		const double x49 = 6*x;
		const double x50 = by_3_1*s;
		const double x51 = by_3_2*x0;
		const double x52 = by_3_3*x1;
		const double x53 = by_3_4*x2;
		const double x54 = by_3_0 + x50 + x51 + x52 + x53;
		const double x55 = 3*x15;
		const double x56 = (1.0/6.0)*y;
		const double x57 = by_1_2 + by_1_3*x27 + by_1_4*x28;
		const double x58 = by_2_2 + by_2_3*x27 + by_2_4*x28;
		const double x59 = 12*x;
		const double x60 = 6*x15;
		const double x61 = y*y*y;
		const double x62 = (1.0/36.0)*x61;
		const double x63 = bx_0_2 + bx_0_3*x27 + bx_0_4*x28;
		const double x64 = bx_1_2 + bx_1_3*x27 + bx_1_4*x28;
		const double x65 = 48*x;
		const double x66 = 24*x15;
		const double x67 = 8*x22;
		const double x68 = 24*x;
		const double x69 = y*y;
		const double x70 = (1.0/48.0)*x69;
		const double x71 = 24*x22;
		const double x72 = x*x*x*x;
		const double x73 = 4*s;
		const double x74 = 48*s;
		const double x75 = 72*x0;
		const double x76 = 96*x1;
		const double x77 = 12*x15;
		const double x78 = bx_3_3 + bx_3_4*x73;
		const double x79 = 2*s;
		const double x80 = 3*x0;
		const double x81 = 4*x1;
		const double x82 = bx_2_1 + bx_2_2*x79 + bx_2_3*x80 + bx_2_4*x81;
		const double x83 = bx_3_1 + bx_3_2*x79 + bx_3_3*x80 + bx_3_4*x81;
		const double x84 = by_2_3 + by_2_4*x73;
		const double x85 = by_3_3 + by_3_4*x73;
		const double x86 = bx_1_3 + bx_1_4*x73;
		const double x87 = bx_2_3 + bx_2_4*x73;
		const double x88 = 12*s;
		const double x89 = 18*x0;
		const double x90 = 24*x1;
		const double x91 = by_3_1 + by_3_2*x79 + by_3_3*x80 + by_3_4*x81;

		// Reduced expressions
		*Bx_out = bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2 - 1.0/70.0*by_3_4*x3 + x*(bx_1_0 + x6 + x7 + x8 + x9) + x14*x16 + x21*x23 + x31*(by_2_4*x25 + by_3_4*x26 + x24 + 24*x29) + x39*(576*bx_1_4*x + bx_2_4*x33 + bx_3_4*x34 + x32 + 96*x35 + x36*x37) - 1.0/17280.0*x4*(1728*bx_3_4*x + x5) + x56*(6*by_1_0 + 6*x40 + 6*x41 + 6*x42 + 6*x43 + x48*x49 + x54*x55) - x62*(6*by_3_0 + x29*x60 + 6*x50 + 6*x51 + 6*x52 + 6*x53 + 12*x57 + x58*x59) - x70*(24*bx_2_0 + 24*x10 + 24*x11 + 24*x12 + 24*x13 + x21*x68 + x35*x66 + x36*x67 + 48*x63 + x64*x65);
		*By_out = (1.0/35.0)*bx_3_4*x3 + by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 + x*(by_1_0 + x40 + x41 + x42 + x43) + x16*x48 + x23*x54 - 1.0/2880.0*x30*(1728*bx_1_4 + 864*bx_3_4*x15 + x*x5 + 144*x36) + (1.0/144.0)*x38*(144*by_0_4 + by_2_4*x26 + by_3_4*x71 + x*x24 + x29*x68 + 24*x58) - 1.0/4320.0*x4*(432*by_2_4 + 432*by_3_4*x) + (1.0/144.0)*x61*(144*bs_3 + 144*bs_4*x73 + bx_1_4*x33 + bx_2_4*x34 + 24*bx_3_0 + 24*bx_3_4*x72 + x*x32 + 48*x15*x36 + 24*x17 + 24*x18 + 24*x19 + 24*x20 + x35*x37 + 96*x64) - 1.0/12.0*x69*(12*by_0_2 + 12*by_0_3*x27 + 12*by_0_4*x28 + 6*by_2_0 + 2*x22*x29 + 6*x44 + 6*x45 + 6*x46 + 6*x47 + x49*x54 + x57*x59 + x58*x60) - 1.0/24.0*y*(24*bs_1 + bs_2*x74 + bs_3*x75 + bs_4*x76 + 24*bx_1_0 + x14*x68 + x21*x77 + x35*x67 + 2*x36*x72 + 24*x6 + x63*x65 + x64*x66 + 24*x7 + 24*x8 + 24*x9);
		*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2 + x*(bx_0_1 + bx_0_2*x79 + bx_0_3*x80 + bx_0_4*x81) + x16*(bx_1_1 + bx_1_2*x79 + bx_1_3*x80 + bx_1_4*x81) + x23*x82 + x31*(72*x*x85 + 72*x84) + x39*(576*bs_4 + 24*bx_3_1 + bx_3_2*x74 + bx_3_3*x75 + bx_3_4*x76 + 288*x*x87 + 144*x15*x78 + 288*x86) - 1.0/40.0*x4*x78 + x56*(6*by_0_1 + by_0_2*x88 + by_0_3*x89 + by_0_4*x90 + x22*x91 + x49*(by_1_1 + by_1_2*x79 + by_1_3*x80 + by_1_4*x81) + x55*(by_2_1 + by_2_2*x79 + by_2_3*x80 + by_2_4*x81)) - x62*(36*by_0_3 + 36*by_0_4*x73 + 6*by_2_1 + by_2_2*x88 + by_2_3*x89 + by_2_4*x90 + 36*x*(by_1_3 + by_1_4*x73) + 18*x15*x84 + 6*x22*x85 + x49*x91) - x70*(48*bs_2 + 48*bs_3*x27 + 48*bs_4*x28 + 24*bx_1_1 + bx_1_2*x74 + bx_1_3*x75 + bx_1_4*x76 + x25*(bx_0_3 + bx_0_4*x73) + x26*x86 + x68*x82 + x71*x87 + 6*x72*x78 + x77*x83) + (1.0/24.0)*x72*x83;
		return;

	}
	case 5: {
		// Hermite → polynomial coefficients (order 5)
		const Poly bs_poly = hermite_to_polynomial(0.0, L, bs);
		const double bs_0   = bs_poly.coeffs[0];
		const double bs_1   = bs_poly.coeffs[1];
		const double bs_2   = bs_poly.coeffs[2];
		const double bs_3   = bs_poly.coeffs[3];
		const double bs_4   = bs_poly.coeffs[4];

		const Poly by0_poly = hermite_to_polynomial(0.0, L, by[0]);
		const double by_0_0 = by0_poly.coeffs[0];
		const double by_0_1 = by0_poly.coeffs[1];
		const double by_0_2 = by0_poly.coeffs[2];
		const double by_0_3 = by0_poly.coeffs[3];
		const double by_0_4 = by0_poly.coeffs[4];

		const Poly by1_poly = hermite_to_polynomial(0.0, L, by[1]);
		const double by_1_0 = by1_poly.coeffs[0];
		const double by_1_1 = by1_poly.coeffs[1];
		const double by_1_2 = by1_poly.coeffs[2];
		const double by_1_3 = by1_poly.coeffs[3];
		const double by_1_4 = by1_poly.coeffs[4];

		const Poly by2_poly = hermite_to_polynomial(0.0, L, by[2]);
		const double by_2_0 = by2_poly.coeffs[0];
		const double by_2_1 = by2_poly.coeffs[1];
		const double by_2_2 = by2_poly.coeffs[2];
		const double by_2_3 = by2_poly.coeffs[3];
		const double by_2_4 = by2_poly.coeffs[4];

		const Poly by3_poly = hermite_to_polynomial(0.0, L, by[3]);
		const double by_3_0 = by3_poly.coeffs[0];
		const double by_3_1 = by3_poly.coeffs[1];
		const double by_3_2 = by3_poly.coeffs[2];
		const double by_3_3 = by3_poly.coeffs[3];
		const double by_3_4 = by3_poly.coeffs[4];

		const Poly by4_poly = hermite_to_polynomial(0.0, L, by[4]);
		const double by_4_0 = by4_poly.coeffs[0];
		const double by_4_1 = by4_poly.coeffs[1];
		const double by_4_2 = by4_poly.coeffs[2];
		const double by_4_3 = by4_poly.coeffs[3];
		const double by_4_4 = by4_poly.coeffs[4];

		const Poly bx0_poly = hermite_to_polynomial(0.0, L, bx[0]);
		const double bx_0_0 = bx0_poly.coeffs[0];
		const double bx_0_1 = bx0_poly.coeffs[1];
		const double bx_0_2 = bx0_poly.coeffs[2];
		const double bx_0_3 = bx0_poly.coeffs[3];
		const double bx_0_4 = bx0_poly.coeffs[4];

		const Poly bx1_poly = hermite_to_polynomial(0.0, L, bx[1]);
		const double bx_1_0 = bx1_poly.coeffs[0];
		const double bx_1_1 = bx1_poly.coeffs[1];
		const double bx_1_2 = bx1_poly.coeffs[2];
		const double bx_1_3 = bx1_poly.coeffs[3];
		const double bx_1_4 = bx1_poly.coeffs[4];

		const Poly bx2_poly = hermite_to_polynomial(0.0, L, bx[2]);
		const double bx_2_0 = bx2_poly.coeffs[0];
		const double bx_2_1 = bx2_poly.coeffs[1];
		const double bx_2_2 = bx2_poly.coeffs[2];
		const double bx_2_3 = bx2_poly.coeffs[3];
		const double bx_2_4 = bx2_poly.coeffs[4];

		const Poly bx3_poly = hermite_to_polynomial(0.0, L, bx[3]);
		const double bx_3_0 = bx3_poly.coeffs[0];
		const double bx_3_1 = bx3_poly.coeffs[1];
		const double bx_3_2 = bx3_poly.coeffs[2];
		const double bx_3_3 = bx3_poly.coeffs[3];
		const double bx_3_4 = bx3_poly.coeffs[4];

		const Poly bx4_poly = hermite_to_polynomial(0.0, L, bx[4]);
		const double bx_4_0 = bx4_poly.coeffs[0];
		const double bx_4_1 = bx4_poly.coeffs[1];
		const double bx_4_2 = bx4_poly.coeffs[2];
		const double bx_4_3 = bx4_poly.coeffs[3];
		const double bx_4_4 = bx4_poly.coeffs[4];

		// Common sub-expressions
		const double x0 = s*s;
		const double x1 = s*s*s;
		const double x2 = s*s*s*s;
		const double x3 = (1.0/280.0)*y*y*y*y*y*y*y*y;
		const double x4 = y*y*y*y*y*y*y;
		const double x5 = 1728*by_3_4;
		const double x6 = bx_1_1*s;
		const double x7 = bx_1_2*x0;
		const double x8 = bx_1_3*x1;
		const double x9 = bx_1_4*x2;
		const double x10 = bx_2_1*s;
		const double x11 = bx_2_2*x0;
		const double x12 = bx_2_3*x1;
		const double x13 = bx_2_4*x2;
		const double x14 = bx_2_0 + x10 + x11 + x12 + x13;
		const double x15 = x*x;
		const double x16 = (1.0/2.0)*x15;
		const double x17 = bx_3_1*s;
		const double x18 = bx_3_2*x0;
		const double x19 = bx_3_3*x1;
		const double x20 = bx_3_4*x2;
		const double x21 = bx_3_0 + x17 + x18 + x19 + x20;
		const double x22 = x*x*x;
		const double x23 = (1.0/6.0)*x22;
		const double x24 = bx_4_1*s;
		const double x25 = bx_4_2*x0;
		const double x26 = bx_4_3*x1;
		const double x27 = bx_4_4*x2;
		const double x28 = bx_4_0 + x24 + x25 + x26 + x27;
		const double x29 = x*x*x*x;
		const double x30 = (1.0/24.0)*x29;
		const double x31 = 8640*bx_2_4;
		const double x32 = 4320*x15;
		const double x33 = 3*s;
		const double x34 = 6*x0;
		const double x35 = bx_4_2 + bx_4_3*x33 + bx_4_4*x34;
		const double x36 = y*y*y*y*y*y;
		const double x37 = (1.0/86400.0)*x36;
		const double x38 = 576*by_1_4;
		const double x39 = 288*x15;
		const double x40 = 96*x22;
		const double x41 = by_3_2 + by_3_3*x33 + by_3_4*x34;
		const double x42 = by_4_2 + by_4_3*x33 + by_4_4*x34;
		const double x43 = 96*x;
		const double x44 = y*y*y*y*y;
		const double x45 = (1.0/2880.0)*x44;
		const double x46 = by_1_1*s;
		const double x47 = by_1_2*x0;
		const double x48 = by_1_3*x1;
		const double x49 = by_1_4*x2;
		const double x50 = by_2_1*s;
		const double x51 = by_2_2*x0;
		const double x52 = by_2_3*x1;
		const double x53 = by_2_4*x2;
		const double x54 = by_2_0 + x50 + x51 + x52 + x53;
		const double x55 = 24*x;
		const double x56 = by_3_1*s;
		const double x57 = by_3_2*x0;
		const double x58 = by_3_3*x1;
		const double x59 = by_3_4*x2;
		const double x60 = by_3_0 + x56 + x57 + x58 + x59;
		const double x61 = 12*x15;
		const double x62 = by_4_1*s;
		const double x63 = by_4_2*x0;
		const double x64 = by_4_3*x1;
		const double x65 = by_4_4*x2;
		const double x66 = by_4_0 + x62 + x63 + x64 + x65;
		const double x67 = 4*x22;
		const double x68 = (1.0/24.0)*y;
		const double x69 = 2880*bx_0_4;
		const double x70 = 1440*x15;
		const double x71 = 480*x22;
		const double x72 = 120*x29;
		const double x73 = bx_2_2 + bx_2_3*x33 + bx_2_4*x34;
		const double x74 = bx_3_2 + bx_3_3*x33 + bx_3_4*x34;
		const double x75 = 480*x;
		const double x76 = 240*x15;
		const double x77 = y*y*y*y;
		const double x78 = (1.0/2880.0)*x77;
		const double x79 = by_1_2 + by_1_3*x33 + by_1_4*x34;
		const double x80 = by_2_2 + by_2_3*x33 + by_2_4*x34;
		const double x81 = 48*x;
		const double x82 = 24*x15;
		const double x83 = 8*x22;
		const double x84 = y*y*y;
		const double x85 = (1.0/144.0)*x84;
		const double x86 = bx_0_2 + bx_0_3*x33 + bx_0_4*x34;
		const double x87 = bx_1_2 + bx_1_3*x33 + bx_1_4*x34;
		const double x88 = 240*x;
		const double x89 = 120*x15;
		const double x90 = 40*x22;
		const double x91 = 10*x29;
		const double x92 = 120*x;
		const double x93 = 60*x15;
		const double x94 = y*y;
		const double x95 = (1.0/240.0)*x94;
		const double x96 = 720*x;
		const double x97 = x*x*x*x*x;
		const double x98 = 4*s;
		const double x99 = 240*s;
		const double x100 = 360*x0;
		const double x101 = 480*x1;
		const double x102 = 20*x22;
		const double x103 = by_4_3 + by_4_4*x98;
		const double x104 = 2*s;
		const double x105 = 3*x0;
		const double x106 = 4*x1;
		const double x107 = bx_2_1 + bx_2_2*x104 + bx_2_3*x105 + bx_2_4*x106;
		const double x108 = bx_3_1 + bx_3_2*x104 + bx_3_3*x105 + bx_3_4*x106;
		const double x109 = bx_4_1 + bx_4_2*x104 + bx_4_3*x105 + bx_4_4*x106;
		const double x110 = bx_3_3 + bx_3_4*x98;
		const double x111 = bx_4_3 + bx_4_4*x98;
		const double x112 = 48*s;
		const double x113 = 72*x0;
		const double x114 = 96*x1;
		const double x115 = by_2_3 + by_2_4*x98;
		const double x116 = by_3_3 + by_3_4*x98;
		const double x117 = bx_1_3 + bx_1_4*x98;
		const double x118 = bx_2_3 + bx_2_4*x98;
		const double x119 = by_4_1 + by_4_2*x104 + by_4_3*x105 + by_4_4*x106;
		const double x120 = by_3_1 + by_3_2*x104 + by_3_3*x105 + by_3_4*x106;

		// Reduced expressions
		*Bx_out = bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2 + bx_4_4*x3 + x*(bx_1_0 + x6 + x7 + x8 + x9) + x14*x16 + x21*x23 + x28*x30 - x37*(8640*bx_3_4*x + bx_4_4*x32 + x31 + 720*x35) - 1.0/120960.0*x4*(1728*by_4_4*x + x5) + x45*(576*by_2_4*x + by_3_4*x39 + by_4_4*x40 + x38 + 96*x41 + x42*x43) + x68*(24*by_1_0 + 24*x46 + 24*x47 + 24*x48 + 24*x49 + x54*x55 + x60*x61 + x66*x67) + x78*(2880*bx_1_4*x + bx_2_4*x70 + bx_3_4*x71 + 120*bx_4_0 + bx_4_4*x72 + 120*x24 + 120*x25 + 120*x26 + 120*x27 + x35*x76 + x69 + 480*x73 + x74*x75) - x85*(24*by_3_0 + x41*x82 + x42*x83 + x55*x66 + 24*x56 + 24*x57 + 24*x58 + 24*x59 + 48*x79 + x80*x81) - x95*(120*bx_2_0 + 120*x10 + 120*x11 + 120*x12 + 120*x13 + x21*x92 + x28*x93 + x35*x91 + x73*x89 + x74*x90 + 240*x86 + x87*x88);
		*By_out = by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 + by_4_4*x3 + x*(by_1_0 + x46 + x47 + x48 + x49) + x16*x54 + x23*x60 + x30*x66 - 1.0/17280.0*x36*(1728*by_2_4 + 864*by_4_4*x15 + x*x5 + 144*x42) + (1.0/604800.0)*x4*(17280*bx_3_4 + 17280*bx_4_4*x) - 1.0/14400.0*x44*(8640*bx_1_4 + bx_3_4*x32 + 1440*bx_4_4*x22 + x*x31 + x35*x96 + 720*x74) + (1.0/576.0)*x77*(576*by_0_4 + by_2_4*x39 + by_3_4*x40 + 24*by_4_0 + 24*by_4_4*x29 + x*x38 + 48*x15*x42 + x41*x43 + 24*x62 + 24*x63 + 24*x64 + 24*x65 + 96*x80) + (1.0/720.0)*x84*(720*bs_3 + 720*bs_4*x98 + bx_1_4*x70 + bx_2_4*x71 + 120*bx_3_0 + bx_3_4*x72 + 24*bx_4_4*x97 + x*x69 + 120*x17 + 120*x18 + 120*x19 + 120*x20 + 80*x22*x35 + x28*x92 + x73*x75 + x74*x76 + 480*x87) - 1.0/48.0*x94*(48*by_0_2 + 48*by_0_3*x33 + 48*by_0_4*x34 + 24*by_2_0 + 2*x29*x42 + x41*x83 + 24*x50 + 24*x51 + 24*x52 + 24*x53 + x55*x60 + x61*x66 + x79*x81 + x80*x82) - 1.0/120.0*y*(120*bs_1 + bs_2*x99 + bs_3*x100 + bs_4*x101 + 120*bx_1_0 + x102*x28 + x14*x92 + x21*x93 + 2*x35*x97 + 120*x6 + 120*x7 + x73*x90 + x74*x91 + 120*x8 + x86*x88 + x87*x89 + 120*x9);
		*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2 + x*(bx_0_1 + bx_0_2*x104 + bx_0_3*x105 + bx_0_4*x106) - 1.0/280.0*x103*x4 + x107*x23 + x108*x30 + (1.0/120.0)*x109*x97 + x16*(bx_1_1 + bx_1_2*x104 + bx_1_3*x105 + bx_1_4*x106) - x37*(2160*x*x111 + 2160*x110) + x45*(24*by_4_1 + by_4_2*x112 + by_4_3*x113 + by_4_4*x114 + 288*x*x116 + 144*x103*x15 + 288*x115) + x68*(24*by_0_1 + by_0_2*x112 + by_0_3*x113 + by_0_4*x114 + x119*x29 + x120*x67 + x55*(by_1_1 + by_1_2*x104 + by_1_3*x105 + by_1_4*x106) + x61*(by_2_1 + by_2_2*x104 + by_2_3*x105 + by_2_4*x106)) + x78*(2880*bs_4 + 120*bx_3_1 + bx_3_2*x99 + bx_3_3*x100 + bx_3_4*x101 + 1440*x*x118 + x109*x92 + 720*x110*x15 + 240*x111*x22 + 1440*x117) - x85*(144*by_0_3 + 144*by_0_4*x98 + 24*by_2_1 + by_2_2*x112 + by_2_3*x113 + by_2_4*x114 + 144*x*(by_1_3 + by_1_4*x98) + 6*x103*x29 + 72*x115*x15 + 24*x116*x22 + x119*x61 + x120*x55) - x95*(240*bs_2 + 240*bs_3*x33 + 240*bs_4*x34 + 120*bx_1_1 + bx_1_2*x99 + bx_1_3*x100 + bx_1_4*x101 + x102*x109 + x107*x92 + x108*x93 + 30*x110*x29 + 6*x111*x97 + 360*x117*x15 + 120*x118*x22 + x96*(bx_0_3 + bx_0_4*x98));
		return;

	}
	case 6: {
		// Hermite → polynomial coefficients (order 6)
		const Poly bs_poly = hermite_to_polynomial(0.0, L, bs);
		const double bs_0   = bs_poly.coeffs[0];
		const double bs_1   = bs_poly.coeffs[1];
		const double bs_2   = bs_poly.coeffs[2];
		const double bs_3   = bs_poly.coeffs[3];
		const double bs_4   = bs_poly.coeffs[4];

		const Poly by0_poly = hermite_to_polynomial(0.0, L, by[0]);
		const double by_0_0 = by0_poly.coeffs[0];
		const double by_0_1 = by0_poly.coeffs[1];
		const double by_0_2 = by0_poly.coeffs[2];
		const double by_0_3 = by0_poly.coeffs[3];
		const double by_0_4 = by0_poly.coeffs[4];

		const Poly by1_poly = hermite_to_polynomial(0.0, L, by[1]);
		const double by_1_0 = by1_poly.coeffs[0];
		const double by_1_1 = by1_poly.coeffs[1];
		const double by_1_2 = by1_poly.coeffs[2];
		const double by_1_3 = by1_poly.coeffs[3];
		const double by_1_4 = by1_poly.coeffs[4];

		const Poly by2_poly = hermite_to_polynomial(0.0, L, by[2]);
		const double by_2_0 = by2_poly.coeffs[0];
		const double by_2_1 = by2_poly.coeffs[1];
		const double by_2_2 = by2_poly.coeffs[2];
		const double by_2_3 = by2_poly.coeffs[3];
		const double by_2_4 = by2_poly.coeffs[4];

		const Poly by3_poly = hermite_to_polynomial(0.0, L, by[3]);
		const double by_3_0 = by3_poly.coeffs[0];
		const double by_3_1 = by3_poly.coeffs[1];
		const double by_3_2 = by3_poly.coeffs[2];
		const double by_3_3 = by3_poly.coeffs[3];
		const double by_3_4 = by3_poly.coeffs[4];

		const Poly by4_poly = hermite_to_polynomial(0.0, L, by[4]);
		const double by_4_0 = by4_poly.coeffs[0];
		const double by_4_1 = by4_poly.coeffs[1];
		const double by_4_2 = by4_poly.coeffs[2];
		const double by_4_3 = by4_poly.coeffs[3];
		const double by_4_4 = by4_poly.coeffs[4];

		const Poly by5_poly = hermite_to_polynomial(0.0, L, by[5]);
		const double by_5_0 = by5_poly.coeffs[0];
		const double by_5_1 = by5_poly.coeffs[1];
		const double by_5_2 = by5_poly.coeffs[2];
		const double by_5_3 = by5_poly.coeffs[3];
		const double by_5_4 = by5_poly.coeffs[4];

		const Poly bx0_poly = hermite_to_polynomial(0.0, L, bx[0]);
		const double bx_0_0 = bx0_poly.coeffs[0];
		const double bx_0_1 = bx0_poly.coeffs[1];
		const double bx_0_2 = bx0_poly.coeffs[2];
		const double bx_0_3 = bx0_poly.coeffs[3];
		const double bx_0_4 = bx0_poly.coeffs[4];

		const Poly bx1_poly = hermite_to_polynomial(0.0, L, bx[1]);
		const double bx_1_0 = bx1_poly.coeffs[0];
		const double bx_1_1 = bx1_poly.coeffs[1];
		const double bx_1_2 = bx1_poly.coeffs[2];
		const double bx_1_3 = bx1_poly.coeffs[3];
		const double bx_1_4 = bx1_poly.coeffs[4];

		const Poly bx2_poly = hermite_to_polynomial(0.0, L, bx[2]);
		const double bx_2_0 = bx2_poly.coeffs[0];
		const double bx_2_1 = bx2_poly.coeffs[1];
		const double bx_2_2 = bx2_poly.coeffs[2];
		const double bx_2_3 = bx2_poly.coeffs[3];
		const double bx_2_4 = bx2_poly.coeffs[4];

		const Poly bx3_poly = hermite_to_polynomial(0.0, L, bx[3]);
		const double bx_3_0 = bx3_poly.coeffs[0];
		const double bx_3_1 = bx3_poly.coeffs[1];
		const double bx_3_2 = bx3_poly.coeffs[2];
		const double bx_3_3 = bx3_poly.coeffs[3];
		const double bx_3_4 = bx3_poly.coeffs[4];

		const Poly bx4_poly = hermite_to_polynomial(0.0, L, bx[4]);
		const double bx_4_0 = bx4_poly.coeffs[0];
		const double bx_4_1 = bx4_poly.coeffs[1];
		const double bx_4_2 = bx4_poly.coeffs[2];
		const double bx_4_3 = bx4_poly.coeffs[3];
		const double bx_4_4 = bx4_poly.coeffs[4];

		const Poly bx5_poly = hermite_to_polynomial(0.0, L, bx[5]);
		const double bx_5_0 = bx5_poly.coeffs[0];
		const double bx_5_1 = bx5_poly.coeffs[1];
		const double bx_5_2 = bx5_poly.coeffs[2];
		const double bx_5_3 = bx5_poly.coeffs[3];
		const double bx_5_4 = bx5_poly.coeffs[4];

		// Common sub-expressions
		const double x0 = s*s;
		const double x1 = s*s*s;
		const double x2 = s*s*s*s;
		const double x3 = y*y*y*y*y*y*y*y*y;
		const double x4 = y*y*y*y*y*y*y*y;
		const double x5 = 103680*bx_4_4;
		const double x6 = bx_1_1*s;
		const double x7 = bx_1_2*x0;
		const double x8 = bx_1_3*x1;
		const double x9 = bx_1_4*x2;
		const double x10 = bx_2_1*s;
		const double x11 = bx_2_2*x0;
		const double x12 = bx_2_3*x1;
		const double x13 = bx_2_4*x2;
		const double x14 = bx_2_0 + x10 + x11 + x12 + x13;
		const double x15 = x*x;
		const double x16 = (1.0/2.0)*x15;
		const double x17 = bx_3_1*s;
		const double x18 = bx_3_2*x0;
		const double x19 = bx_3_3*x1;
		const double x20 = bx_3_4*x2;
		const double x21 = bx_3_0 + x17 + x18 + x19 + x20;
		const double x22 = x*x*x;
		const double x23 = (1.0/6.0)*x22;
		const double x24 = bx_4_1*s;
		const double x25 = bx_4_2*x0;
		const double x26 = bx_4_3*x1;
		const double x27 = bx_4_4*x2;
		const double x28 = bx_4_0 + x24 + x25 + x26 + x27;
		const double x29 = x*x*x*x;
		const double x30 = (1.0/24.0)*x29;
		const double x31 = bx_5_1*s;
		const double x32 = bx_5_2*x0;
		const double x33 = bx_5_3*x1;
		const double x34 = bx_5_4*x2;
		const double x35 = bx_5_0 + x31 + x32 + x33 + x34;
		const double x36 = x*x*x*x*x;
		const double x37 = (1.0/120.0)*x36;
		const double x38 = 8640*by_3_4;
		const double x39 = 8640*x;
		const double x40 = 4320*x15;
		const double x41 = 3*s;
		const double x42 = 6*x0;
		const double x43 = by_5_2 + by_5_3*x41 + by_5_4*x42;
		const double x44 = y*y*y*y*y*y*y;
		const double x45 = (1.0/604800.0)*x44;
		const double x46 = 51840*bx_2_4;
		const double x47 = 25920*x15;
		const double x48 = 8640*x22;
		const double x49 = bx_4_2 + bx_4_3*x41 + bx_4_4*x42;
		const double x50 = bx_5_2 + bx_5_3*x41 + bx_5_4*x42;
		const double x51 = 4320*x;
		const double x52 = y*y*y*y*y*y;
		const double x53 = (1.0/518400.0)*x52;
		const double x54 = 2880*by_1_4;
		const double x55 = 2880*x;
		const double x56 = by_5_1*s;
		const double x57 = 1440*x15;
		const double x58 = 480*x22;
		const double x59 = by_5_2*x0;
		const double x60 = by_5_3*x1;
		const double x61 = by_5_4*x2;
		const double x62 = 120*x29;
		const double x63 = by_3_2 + by_3_3*x41 + by_3_4*x42;
		const double x64 = by_4_2 + by_4_3*x41 + by_4_4*x42;
		const double x65 = 480*x;
		const double x66 = 240*x15;
		const double x67 = y*y*y*y*y;
		const double x68 = (1.0/14400.0)*x67;
		const double x69 = by_1_1*s;
		const double x70 = by_1_2*x0;
		const double x71 = by_1_3*x1;
		const double x72 = by_1_4*x2;
		const double x73 = by_2_1*s;
		const double x74 = by_2_2*x0;
		const double x75 = by_2_3*x1;
		const double x76 = by_2_4*x2;
		const double x77 = by_2_0 + x73 + x74 + x75 + x76;
		const double x78 = 120*x;
		const double x79 = by_3_1*s;
		const double x80 = by_3_2*x0;
		const double x81 = by_3_3*x1;
		const double x82 = by_3_4*x2;
		const double x83 = by_3_0 + x79 + x80 + x81 + x82;
		const double x84 = 60*x15;
		const double x85 = by_4_1*s;
		const double x86 = by_4_2*x0;
		const double x87 = by_4_3*x1;
		const double x88 = by_4_4*x2;
		const double x89 = by_4_0 + x85 + x86 + x87 + x88;
		const double x90 = 20*x22;
		const double x91 = by_5_0 + x56 + x59 + x60 + x61;
		const double x92 = 5*x29;
		const double x93 = (1.0/120.0)*y;
		const double x94 = 17280*bx_0_4;
		const double x95 = 17280*x;
		const double x96 = 8640*x15;
		const double x97 = 2880*x22;
		const double x98 = 720*x29;
		const double x99 = 144*x36;
		const double x100 = bx_2_2 + bx_2_3*x41 + bx_2_4*x42;
		const double x101 = bx_3_2 + bx_3_3*x41 + bx_3_4*x42;
		const double x102 = 720*x;
		const double x103 = y*y*y*y;
		const double x104 = (1.0/17280.0)*x103;
		const double x105 = by_1_2 + by_1_3*x41 + by_1_4*x42;
		const double x106 = by_2_2 + by_2_3*x41 + by_2_4*x42;
		const double x107 = 240*x;
		const double x108 = 120*x15;
		const double x109 = 40*x22;
		const double x110 = 10*x29;
		const double x111 = y*y*y;
		const double x112 = (1.0/720.0)*x111;
		const double x113 = bx_0_2 + bx_0_3*x41 + bx_0_4*x42;
		const double x114 = bx_1_2 + bx_1_3*x41 + bx_1_4*x42;
		const double x115 = 1440*x;
		const double x116 = 720*x15;
		const double x117 = 240*x22;
		const double x118 = 60*x29;
		const double x119 = 12*x36;
		const double x120 = 360*x15;
		const double x121 = 120*x22;
		const double x122 = y*y;
		const double x123 = (1.0/1440.0)*x122;
		const double x124 = 1440*x22;
		const double x125 = 2160*x15;
		const double x126 = x*x*x*x*x*x;
		const double x127 = 4*s;
		const double x128 = 1440*s;
		const double x129 = 2160*x0;
		const double x130 = 2880*x1;
		const double x131 = 30*x29;
		const double x132 = bx_5_3 + bx_5_4*x127;
		const double x133 = 2*s;
		const double x134 = 3*x0;
		const double x135 = 4*x1;
		const double x136 = bx_2_1 + bx_2_2*x133 + bx_2_3*x134 + bx_2_4*x135;
		const double x137 = bx_3_1 + bx_3_2*x133 + bx_3_3*x134 + bx_3_4*x135;
		const double x138 = bx_4_1 + bx_4_2*x133 + bx_4_3*x134 + bx_4_4*x135;
		const double x139 = bx_5_1 + bx_5_2*x133 + bx_5_3*x134 + bx_5_4*x135;
		const double x140 = by_4_3 + by_4_4*x127;
		const double x141 = by_5_3 + by_5_4*x127;
		const double x142 = bx_3_3 + bx_3_4*x127;
		const double x143 = bx_4_3 + bx_4_4*x127;
		const double x144 = 240*s;
		const double x145 = 360*x0;
		const double x146 = 480*x1;
		const double x147 = by_2_3 + by_2_4*x127;
		const double x148 = by_3_3 + by_3_4*x127;
		const double x149 = by_5_1 + by_5_2*x133 + by_5_3*x134 + by_5_4*x135;
		const double x150 = bx_1_3 + bx_1_4*x127;
		const double x151 = bx_2_3 + bx_2_4*x127;
		const double x152 = by_3_1 + by_3_2*x133 + by_3_3*x134 + by_3_4*x135;
		const double x153 = by_4_1 + by_4_2*x133 + by_4_3*x134 + by_4_4*x135;

		// Reduced expressions
		*Bx_out = bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2 + (1.0/2520.0)*by_5_4*x3 + x*(bx_1_0 + x6 + x7 + x8 + x9) + x104*(bx_1_4*x95 + bx_2_4*x96 + bx_3_4*x97 + 720*bx_4_0 + bx_4_4*x98 + bx_5_4*x99 + 2880*x100 + x101*x55 + x102*x35 + 720*x24 + 720*x25 + 720*x26 + 720*x27 + x49*x57 + x50*x58 + x94) - x112*(120*by_3_0 + 240*x105 + x106*x107 + x108*x63 + x109*x64 + x110*x43 + x78*x89 + 120*x79 + 120*x80 + 120*x81 + 120*x82 + x84*x91) - x123*(720*bx_2_0 + 720*x10 + x100*x116 + x101*x117 + x102*x21 + 720*x11 + 1440*x113 + x114*x115 + x118*x49 + x119*x50 + 720*x12 + x120*x28 + x121*x35 + 720*x13) + x14*x16 + x21*x23 + x28*x30 + x35*x37 + (1.0/29030400.0)*x4*(103680*bx_5_4*x + x5) - x45*(by_4_4*x39 + by_5_4*x40 + x38 + 720*x43) - x53*(51840*bx_3_4*x + bx_4_4*x47 + bx_5_4*x48 + x46 + 4320*x49 + x50*x51) + x68*(by_2_4*x55 + by_3_4*x57 + by_4_4*x58 + 120*by_5_0 + by_5_4*x62 + x43*x66 + x54 + 120*x56 + 120*x59 + 120*x60 + 120*x61 + 480*x63 + x64*x65) + x93*(120*by_1_0 + 120*x69 + 120*x70 + 120*x71 + 120*x72 + x77*x78 + x83*x84 + x89*x90 + x91*x92);
		*By_out = -1.0/1512.0*bx_5_4*x3 + by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 + x*(by_1_0 + x69 + x70 + x71 + x72) + (1.0/2880.0)*x103*(2880*by_0_4 + by_2_4*x57 + by_3_4*x58 + 120*by_4_0 + by_4_4*x62 + 24*by_5_4*x36 + x*x54 + 480*x106 + 80*x22*x43 + x63*x65 + x64*x66 + x78*x91 + 120*x85 + 120*x86 + 120*x87 + 120*x88) + (1.0/4320.0)*x111*(4320*bs_3 + 4320*bs_4*x127 + bx_1_4*x96 + bx_2_4*x97 + 720*bx_3_0 + bx_3_4*x98 + bx_4_4*x99 + 24*bx_5_4*x126 + x*x94 + x100*x55 + x101*x57 + x102*x28 + 2880*x114 + x120*x35 + 720*x17 + 720*x18 + 720*x19 + 720*x20 + x49*x58 + x50*x62) - 1.0/240.0*x122*(240*by_0_2 + 240*by_0_3*x41 + 240*by_0_4*x42 + 120*by_2_0 + x105*x107 + x106*x108 + x109*x63 + x110*x64 + 2*x36*x43 + 120*x73 + 120*x74 + 120*x75 + 120*x76 + x78*x83 + x84*x89 + x90*x91) + x16*x77 + x23*x83 + x30*x89 + x37*x91 + (1.0/4838400.0)*x4*(17280*by_4_4 + by_5_4*x95) + (1.0/3628800.0)*x44*(103680*bx_3_4 + 51840*bx_5_4*x15 + x*x5 + 5760*x50) - 1.0/86400.0*x52*(8640*by_2_4 + by_4_4*x40 + by_5_4*x124 + x*x38 + x102*x43 + 720*x64) - 1.0/86400.0*x67*(51840*bx_1_4 + bx_3_4*x47 + bx_4_4*x48 + 720*bx_5_0 + 2160*bx_5_4*x29 + x*x46 + 4320*x101 + x125*x50 + 720*x31 + 720*x32 + 720*x33 + 720*x34 + x49*x51) - 1.0/720.0*y*(720*bs_1 + bs_2*x128 + bs_3*x129 + bs_4*x130 + 720*bx_1_0 + x100*x117 + x101*x118 + x102*x14 + x113*x115 + x114*x116 + x119*x49 + x120*x21 + x121*x28 + 2*x126*x50 + x131*x35 + 720*x6 + 720*x7 + 720*x8 + 720*x9);
		*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2 + x*(bx_0_1 + bx_0_2*x133 + bx_0_3*x134 + bx_0_4*x135) + x104*(17280*bs_4 + 720*bx_3_1 + bx_3_2*x128 + bx_3_3*x129 + bx_3_4*x130 + x102*x138 + x120*x139 + x124*x143 + 360*x132*x29 + x142*x40 + 8640*x150 + x151*x39) - x112*(720*by_0_3 + 720*by_0_4*x127 + 120*by_2_1 + by_2_2*x144 + by_2_3*x145 + by_2_4*x146 + x102*(by_1_3 + by_1_4*x127) + x120*x147 + x121*x148 + x131*x140 + 6*x141*x36 + x149*x90 + x152*x78 + x153*x84) - x123*(1440*bs_2 + 1440*bs_3*x41 + 1440*bs_4*x42 + 720*bx_1_1 + bx_1_2*x128 + bx_1_3*x129 + bx_1_4*x130 + x102*x136 + x120*x137 + x121*x138 + x125*x150 + 6*x126*x132 + x131*x139 + 180*x142*x29 + 36*x143*x36 + 720*x151*x22 + x51*(bx_0_3 + bx_0_4*x127)) + (1.0/720.0)*x126*x139 + (1.0/1680.0)*x132*x4 + x136*x23 + x137*x30 + x138*x37 + x16*(bx_1_1 + bx_1_2*x133 + bx_1_3*x134 + bx_1_4*x135) - x45*(2160*x*x141 + 2160*x140) - x53*(720*bx_5_1 + bx_5_2*x128 + bx_5_3*x129 + bx_5_4*x130 + 12960*x*x143 + 6480*x132*x15 + 12960*x142) + x68*(120*by_4_1 + by_4_2*x144 + by_4_3*x145 + by_4_4*x146 + x115*x148 + x116*x140 + x117*x141 + 1440*x147 + x149*x78) + x93*(120*by_0_1 + by_0_2*x144 + by_0_3*x145 + by_0_4*x146 + x149*x36 + x152*x90 + x153*x92 + x78*(by_1_1 + by_1_2*x133 + by_1_3*x134 + by_1_4*x135) + x84*(by_2_1 + by_2_2*x133 + by_2_3*x134 + by_2_4*x135));
		return;

	}
	case 7: {
		// Hermite → polynomial coefficients (order 7)
		const Poly bs_poly = hermite_to_polynomial(0.0, L, bs);
		const double bs_0   = bs_poly.coeffs[0];
		const double bs_1   = bs_poly.coeffs[1];
		const double bs_2   = bs_poly.coeffs[2];
		const double bs_3   = bs_poly.coeffs[3];
		const double bs_4   = bs_poly.coeffs[4];

		const Poly by0_poly = hermite_to_polynomial(0.0, L, by[0]);
		const double by_0_0 = by0_poly.coeffs[0];
		const double by_0_1 = by0_poly.coeffs[1];
		const double by_0_2 = by0_poly.coeffs[2];
		const double by_0_3 = by0_poly.coeffs[3];
		const double by_0_4 = by0_poly.coeffs[4];

		const Poly by1_poly = hermite_to_polynomial(0.0, L, by[1]);
		const double by_1_0 = by1_poly.coeffs[0];
		const double by_1_1 = by1_poly.coeffs[1];
		const double by_1_2 = by1_poly.coeffs[2];
		const double by_1_3 = by1_poly.coeffs[3];
		const double by_1_4 = by1_poly.coeffs[4];

		const Poly by2_poly = hermite_to_polynomial(0.0, L, by[2]);
		const double by_2_0 = by2_poly.coeffs[0];
		const double by_2_1 = by2_poly.coeffs[1];
		const double by_2_2 = by2_poly.coeffs[2];
		const double by_2_3 = by2_poly.coeffs[3];
		const double by_2_4 = by2_poly.coeffs[4];

		const Poly by3_poly = hermite_to_polynomial(0.0, L, by[3]);
		const double by_3_0 = by3_poly.coeffs[0];
		const double by_3_1 = by3_poly.coeffs[1];
		const double by_3_2 = by3_poly.coeffs[2];
		const double by_3_3 = by3_poly.coeffs[3];
		const double by_3_4 = by3_poly.coeffs[4];

		const Poly by4_poly = hermite_to_polynomial(0.0, L, by[4]);
		const double by_4_0 = by4_poly.coeffs[0];
		const double by_4_1 = by4_poly.coeffs[1];
		const double by_4_2 = by4_poly.coeffs[2];
		const double by_4_3 = by4_poly.coeffs[3];
		const double by_4_4 = by4_poly.coeffs[4];

		const Poly by5_poly = hermite_to_polynomial(0.0, L, by[5]);
		const double by_5_0 = by5_poly.coeffs[0];
		const double by_5_1 = by5_poly.coeffs[1];
		const double by_5_2 = by5_poly.coeffs[2];
		const double by_5_3 = by5_poly.coeffs[3];
		const double by_5_4 = by5_poly.coeffs[4];

		const Poly by6_poly = hermite_to_polynomial(0.0, L, by[6]);
		const double by_6_0 = by6_poly.coeffs[0];
		const double by_6_1 = by6_poly.coeffs[1];
		const double by_6_2 = by6_poly.coeffs[2];
		const double by_6_3 = by6_poly.coeffs[3];
		const double by_6_4 = by6_poly.coeffs[4];

		const Poly bx0_poly = hermite_to_polynomial(0.0, L, bx[0]);
		const double bx_0_0 = bx0_poly.coeffs[0];
		const double bx_0_1 = bx0_poly.coeffs[1];
		const double bx_0_2 = bx0_poly.coeffs[2];
		const double bx_0_3 = bx0_poly.coeffs[3];
		const double bx_0_4 = bx0_poly.coeffs[4];

		const Poly bx1_poly = hermite_to_polynomial(0.0, L, bx[1]);
		const double bx_1_0 = bx1_poly.coeffs[0];
		const double bx_1_1 = bx1_poly.coeffs[1];
		const double bx_1_2 = bx1_poly.coeffs[2];
		const double bx_1_3 = bx1_poly.coeffs[3];
		const double bx_1_4 = bx1_poly.coeffs[4];

		const Poly bx2_poly = hermite_to_polynomial(0.0, L, bx[2]);
		const double bx_2_0 = bx2_poly.coeffs[0];
		const double bx_2_1 = bx2_poly.coeffs[1];
		const double bx_2_2 = bx2_poly.coeffs[2];
		const double bx_2_3 = bx2_poly.coeffs[3];
		const double bx_2_4 = bx2_poly.coeffs[4];

		const Poly bx3_poly = hermite_to_polynomial(0.0, L, bx[3]);
		const double bx_3_0 = bx3_poly.coeffs[0];
		const double bx_3_1 = bx3_poly.coeffs[1];
		const double bx_3_2 = bx3_poly.coeffs[2];
		const double bx_3_3 = bx3_poly.coeffs[3];
		const double bx_3_4 = bx3_poly.coeffs[4];

		const Poly bx4_poly = hermite_to_polynomial(0.0, L, bx[4]);
		const double bx_4_0 = bx4_poly.coeffs[0];
		const double bx_4_1 = bx4_poly.coeffs[1];
		const double bx_4_2 = bx4_poly.coeffs[2];
		const double bx_4_3 = bx4_poly.coeffs[3];
		const double bx_4_4 = bx4_poly.coeffs[4];

		const Poly bx5_poly = hermite_to_polynomial(0.0, L, bx[5]);
		const double bx_5_0 = bx5_poly.coeffs[0];
		const double bx_5_1 = bx5_poly.coeffs[1];
		const double bx_5_2 = bx5_poly.coeffs[2];
		const double bx_5_3 = bx5_poly.coeffs[3];
		const double bx_5_4 = bx5_poly.coeffs[4];

		const Poly bx6_poly = hermite_to_polynomial(0.0, L, bx[6]);
		const double bx_6_0 = bx6_poly.coeffs[0];
		const double bx_6_1 = bx6_poly.coeffs[1];
		const double bx_6_2 = bx6_poly.coeffs[2];
		const double bx_6_3 = bx6_poly.coeffs[3];
		const double bx_6_4 = bx6_poly.coeffs[4];

		// Common sub-expressions
		const double x0 = s*s;
		const double x1 = s*s*s;
		const double x2 = s*s*s*s;
		const double x3 = (1.0/15120.0)*y*y*y*y*y*y*y*y*y*y;
		const double x4 = y*y*y*y*y*y*y*y*y;
		const double x5 = 103680*by_5_4;
		const double x6 = bx_1_1*s;
		const double x7 = bx_1_2*x0;
		const double x8 = bx_1_3*x1;
		const double x9 = bx_1_4*x2;
		const double x10 = bx_2_1*s;
		const double x11 = bx_2_2*x0;
		const double x12 = bx_2_3*x1;
		const double x13 = bx_2_4*x2;
		const double x14 = bx_2_0 + x10 + x11 + x12 + x13;
		const double x15 = x*x;
		const double x16 = (1.0/2.0)*x15;
		const double x17 = bx_3_1*s;
		const double x18 = bx_3_2*x0;
		const double x19 = bx_3_3*x1;
		const double x20 = bx_3_4*x2;
		const double x21 = bx_3_0 + x17 + x18 + x19 + x20;
		const double x22 = x*x*x;
		const double x23 = (1.0/6.0)*x22;
		const double x24 = bx_4_1*s;
		const double x25 = bx_4_2*x0;
		const double x26 = bx_4_3*x1;
		const double x27 = bx_4_4*x2;
		const double x28 = bx_4_0 + x24 + x25 + x26 + x27;
		const double x29 = x*x*x*x;
		const double x30 = (1.0/24.0)*x29;
		const double x31 = bx_5_1*s;
		const double x32 = bx_5_2*x0;
		const double x33 = bx_5_3*x1;
		const double x34 = bx_5_4*x2;
		const double x35 = bx_5_0 + x31 + x32 + x33 + x34;
		const double x36 = x*x*x*x*x;
		const double x37 = (1.0/120.0)*x36;
		const double x38 = bx_6_1*s;
		const double x39 = bx_6_2*x0;
		const double x40 = bx_6_3*x1;
		const double x41 = bx_6_4*x2;
		const double x42 = bx_6_0 + x38 + x39 + x40 + x41;
		const double x43 = x*x*x*x*x*x;
		const double x44 = (1.0/720.0)*x43;
		const double x45 = 725760*bx_4_4;
		const double x46 = 362880*x15;
		const double x47 = 3*s;
		const double x48 = 6*x0;
		const double x49 = bx_6_2 + bx_6_3*x47 + bx_6_4*x48;
		const double x50 = y*y*y*y*y*y*y*y;
		const double x51 = (1.0/203212800.0)*x50;
		const double x52 = 51840*by_3_4;
		const double x53 = 25920*x15;
		const double x54 = 8640*x22;
		const double x55 = by_5_2 + by_5_3*x47 + by_5_4*x48;
		const double x56 = by_6_2 + by_6_3*x47 + by_6_4*x48;
		const double x57 = 4320*x;
		const double x58 = y*y*y*y*y*y*y;
		const double x59 = (1.0/3628800.0)*x58;
		const double x60 = 362880*bx_2_4;
		const double x61 = 181440*x15;
		const double x62 = 60480*x22;
		const double x63 = 15120*x29;
		const double x64 = bx_4_2 + bx_4_3*x47 + bx_4_4*x48;
		const double x65 = bx_5_2 + bx_5_3*x47 + bx_5_4*x48;
		const double x66 = 30240*x;
		const double x67 = 15120*x15;
		const double x68 = y*y*y*y*y*y;
		const double x69 = (1.0/3628800.0)*x68;
		const double x70 = 17280*by_1_4;
		const double x71 = by_5_1*s;
		const double x72 = 8640*x15;
		const double x73 = 2880*x22;
		const double x74 = by_5_2*x0;
		const double x75 = by_5_3*x1;
		const double x76 = by_5_4*x2;
		const double x77 = 720*x29;
		const double x78 = 144*x36;
		const double x79 = by_3_2 + by_3_3*x47 + by_3_4*x48;
		const double x80 = by_4_2 + by_4_3*x47 + by_4_4*x48;
		const double x81 = 2880*x;
		const double x82 = 1440*x15;
		const double x83 = 480*x22;
		const double x84 = by_6_1*s;
		const double x85 = by_6_2*x0;
		const double x86 = by_6_3*x1;
		const double x87 = by_6_4*x2;
		const double x88 = by_6_0 + x84 + x85 + x86 + x87;
		const double x89 = 720*x;
		const double x90 = y*y*y*y*y;
		const double x91 = (1.0/86400.0)*x90;
		const double x92 = by_1_1*s;
		const double x93 = by_1_2*x0;
		const double x94 = by_1_3*x1;
		const double x95 = by_1_4*x2;
		const double x96 = by_2_1*s;
		const double x97 = by_2_2*x0;
		const double x98 = by_2_3*x1;
		const double x99 = by_2_4*x2;
		const double x100 = by_2_0 + x96 + x97 + x98 + x99;
		const double x101 = by_3_1*s;
		const double x102 = by_3_2*x0;
		const double x103 = by_3_3*x1;
		const double x104 = by_3_4*x2;
		const double x105 = by_3_0 + x101 + x102 + x103 + x104;
		const double x106 = 360*x15;
		const double x107 = by_4_1*s;
		const double x108 = by_4_2*x0;
		const double x109 = by_4_3*x1;
		const double x110 = by_4_4*x2;
		const double x111 = by_4_0 + x107 + x108 + x109 + x110;
		const double x112 = 120*x22;
		const double x113 = by_5_0 + x71 + x74 + x75 + x76;
		const double x114 = 30*x29;
		const double x115 = 6*x36;
		const double x116 = (1.0/720.0)*y;
		const double x117 = 120960*bx_0_4;
		const double x118 = 120960*x;
		const double x119 = 60480*x15;
		const double x120 = 20160*x22;
		const double x121 = 5040*x29;
		const double x122 = 1008*x36;
		const double x123 = 168*x43;
		const double x124 = bx_2_2 + bx_2_3*x47 + bx_2_4*x48;
		const double x125 = bx_3_2 + bx_3_3*x47 + bx_3_4*x48;
		const double x126 = 20160*x;
		const double x127 = 10080*x15;
		const double x128 = 3360*x22;
		const double x129 = 840*x29;
		const double x130 = 5040*x;
		const double x131 = 2520*x15;
		const double x132 = y*y*y*y;
		const double x133 = (1.0/120960.0)*x132;
		const double x134 = by_1_2 + by_1_3*x47 + by_1_4*x48;
		const double x135 = by_2_2 + by_2_3*x47 + by_2_4*x48;
		const double x136 = 1440*x;
		const double x137 = 720*x15;
		const double x138 = 240*x22;
		const double x139 = 60*x29;
		const double x140 = 12*x36;
		const double x141 = y*y*y;
		const double x142 = (1.0/4320.0)*x141;
		const double x143 = bx_0_2 + bx_0_3*x47 + bx_0_4*x48;
		const double x144 = bx_1_2 + bx_1_3*x47 + bx_1_4*x48;
		const double x145 = 10080*x;
		const double x146 = 5040*x15;
		const double x147 = 1680*x22;
		const double x148 = 420*x29;
		const double x149 = 84*x36;
		const double x150 = 14*x43;
		const double x151 = 840*x22;
		const double x152 = 210*x29;
		const double x153 = y*y;
		const double x154 = (1.0/10080.0)*x153;
		const double x155 = 2160*x15;
		const double x156 = 5040*x22;
		const double x157 = x*x*x*x*x*x*x;
		const double x158 = 4*s;
		const double x159 = 10080*s;
		const double x160 = 15120*x0;
		const double x161 = 20160*x1;
		const double x162 = 42*x36;
		const double x163 = by_6_3 + by_6_4*x158;
		const double x164 = 2*s;
		const double x165 = 3*x0;
		const double x166 = 4*x1;
		const double x167 = bx_2_1 + bx_2_2*x164 + bx_2_3*x165 + bx_2_4*x166;
		const double x168 = bx_3_1 + bx_3_2*x164 + bx_3_3*x165 + bx_3_4*x166;
		const double x169 = bx_4_1 + bx_4_2*x164 + bx_4_3*x165 + bx_4_4*x166;
		const double x170 = bx_5_1 + bx_5_2*x164 + bx_5_3*x165 + bx_5_4*x166;
		const double x171 = bx_6_1 + bx_6_2*x164 + bx_6_3*x165 + bx_6_4*x166;
		const double x172 = bx_5_3 + bx_5_4*x158;
		const double x173 = bx_6_3 + bx_6_4*x158;
		const double x174 = 1440*s;
		const double x175 = 2160*x0;
		const double x176 = 2880*x1;
		const double x177 = by_4_3 + by_4_4*x158;
		const double x178 = by_5_3 + by_5_4*x158;
		const double x179 = bx_3_3 + bx_3_4*x158;
		const double x180 = bx_4_3 + bx_4_4*x158;
		const double x181 = by_2_3 + by_2_4*x158;
		const double x182 = by_3_3 + by_3_4*x158;
		const double x183 = by_5_1 + by_5_2*x164 + by_5_3*x165 + by_5_4*x166;
		const double x184 = by_6_1 + by_6_2*x164 + by_6_3*x165 + by_6_4*x166;
		const double x185 = bx_1_3 + bx_1_4*x158;
		const double x186 = bx_2_3 + bx_2_4*x158;
		const double x187 = by_3_1 + by_3_2*x164 + by_3_3*x165 + by_3_4*x166;
		const double x188 = by_4_1 + by_4_2*x164 + by_4_3*x165 + by_4_4*x166;

		// Reduced expressions
		*Bx_out = bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2 - bx_6_4*x3 + x*(bx_1_0 + x6 + x7 + x8 + x9) + x116*(720*by_1_0 + x100*x89 + x105*x106 + x111*x112 + x113*x114 + x115*x88 + 720*x92 + 720*x93 + 720*x94 + 720*x95) + x133*(bx_1_4*x118 + bx_2_4*x119 + bx_3_4*x120 + 5040*bx_4_0 + bx_4_4*x121 + bx_5_4*x122 + bx_6_4*x123 + x117 + 20160*x124 + x125*x126 + x127*x64 + x128*x65 + x129*x49 + x130*x35 + x131*x42 + 5040*x24 + 5040*x25 + 5040*x26 + 5040*x27) + x14*x16 - x142*(720*by_3_0 + 720*x101 + 720*x102 + 720*x103 + 720*x104 + x106*x113 + x111*x89 + x112*x88 + 1440*x134 + x135*x136 + x137*x79 + x138*x80 + x139*x55 + x140*x56) - x154*(5040*bx_2_0 + 5040*x10 + 5040*x11 + 5040*x12 + x124*x146 + x125*x147 + 5040*x13 + x130*x21 + x131*x28 + 10080*x143 + x144*x145 + x148*x64 + x149*x65 + x150*x49 + x151*x35 + x152*x42) + x21*x23 + x28*x30 + x35*x37 + (1.0/261273600.0)*x4*(103680*by_6_4*x + x5) + x42*x44 + x51*(725760*bx_5_4*x + bx_6_4*x46 + x45 + 40320*x49) - x59*(51840*by_4_4*x + by_5_4*x53 + by_6_4*x54 + x52 + 4320*x55 + x56*x57) - x69*(362880*bx_3_4*x + bx_4_4*x61 + bx_5_4*x62 + 5040*bx_6_0 + bx_6_4*x63 + 5040*x38 + 5040*x39 + 5040*x40 + 5040*x41 + x49*x67 + x60 + 30240*x64 + x65*x66) + x91*(17280*by_2_4*x + by_3_4*x72 + by_4_4*x73 + 720*by_5_0 + by_5_4*x77 + by_6_4*x78 + x55*x82 + x56*x83 + x70 + 720*x71 + 720*x74 + 720*x75 + 720*x76 + 2880*x79 + x80*x81 + x88*x89);
		*By_out = by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 - by_6_4*x3 + x*(by_1_0 + x92 + x93 + x94 + x95) + x100*x16 + x105*x23 + x111*x30 + x113*x37 + (1.0/17280.0)*x132*(17280*by_0_4 + by_2_4*x72 + by_3_4*x73 + 720*by_4_0 + by_4_4*x77 + by_5_4*x78 + 24*by_6_4*x43 + x*x70 + x106*x88 + 720*x107 + 720*x108 + 720*x109 + 720*x110 + x113*x89 + 2880*x135 + 120*x29*x56 + x55*x83 + x79*x81 + x80*x82) + (1.0/30240.0)*x141*(30240*bs_3 + 30240*bs_4*x158 + bx_1_4*x119 + bx_2_4*x120 + 5040*bx_3_0 + bx_3_4*x121 + bx_4_4*x122 + bx_5_4*x123 + 24*bx_6_4*x157 + x*x117 + x124*x126 + x125*x127 + x128*x64 + x129*x65 + x130*x28 + x131*x35 + 20160*x144 + x151*x42 + 5040*x17 + 5040*x18 + 5040*x19 + 5040*x20 + 168*x36*x49) - 1.0/1440.0*x153*(1440*by_0_2 + 1440*by_0_3*x47 + 1440*by_0_4*x48 + 720*by_2_0 + x105*x89 + x106*x111 + x112*x113 + x114*x88 + x134*x136 + x135*x137 + x138*x79 + x139*x80 + x140*x55 + 2*x43*x56 + 720*x96 + 720*x97 + 720*x98 + 720*x99) - 1.0/1828915200.0*x4*(1209600*bx_5_4 + 1209600*bx_6_4*x) + x44*x88 + (1.0/29030400.0)*x50*(103680*by_4_4 + 51840*by_6_4*x15 + x*x5 + 5760*x56) + (1.0/25401600.0)*x58*(725760*bx_3_4 + bx_5_4*x46 + 120960*bx_6_4*x22 + x*x45 + 40320*x*x49 + 40320*x65) - 1.0/518400.0*x68*(51840*by_2_4 + by_4_4*x53 + by_5_4*x54 + 720*by_6_0 + 2160*by_6_4*x29 + x*x52 + x155*x56 + x55*x57 + 4320*x80 + 720*x84 + 720*x85 + 720*x86 + 720*x87) - 1.0/604800.0*x90*(362880*bx_1_4 + bx_3_4*x61 + bx_4_4*x62 + 5040*bx_5_0 + bx_5_4*x63 + 3024*bx_6_4*x36 + x*x60 + 30240*x125 + x130*x42 + x156*x49 + 5040*x31 + 5040*x32 + 5040*x33 + 5040*x34 + x64*x66 + x65*x67) - 1.0/5040.0*y*(5040*bs_1 + bs_2*x159 + bs_3*x160 + bs_4*x161 + 5040*bx_1_0 + x124*x147 + x125*x148 + x130*x14 + x131*x21 + x143*x145 + x144*x146 + x149*x64 + x150*x65 + x151*x28 + x152*x35 + 2*x157*x49 + x162*x42 + 5040*x6 + 5040*x7 + 5040*x8 + 5040*x9);
		*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2 + x*(bx_0_1 + bx_0_2*x164 + bx_0_3*x165 + bx_0_4*x166) + x116*(720*by_0_1 + by_0_2*x174 + by_0_3*x175 + by_0_4*x176 + x106*(by_2_1 + by_2_2*x164 + by_2_3*x165 + by_2_4*x166) + x112*x187 + x114*x188 + x115*x183 + x184*x43 + x89*(by_1_1 + by_1_2*x164 + by_1_3*x165 + by_1_4*x166)) + x133*(120960*bs_4 + 5040*bx_3_1 + bx_3_2*x159 + bx_3_3*x160 + bx_3_4*x161 + 60480*x*x186 + x130*x169 + x131*x170 + 30240*x15*x179 + x151*x171 + 2520*x172*x29 + 504*x173*x36 + 10080*x180*x22 + 60480*x185) - x142*(4320*by_0_3 + 4320*by_0_4*x158 + 720*by_2_1 + by_2_2*x174 + by_2_3*x175 + by_2_4*x176 + x106*x188 + x112*x183 + x114*x184 + x155*x181 + 6*x163*x43 + 180*x177*x29 + 36*x178*x36 + 720*x182*x22 + x187*x89 + x57*(by_1_3 + by_1_4*x158)) - x154*(10080*bs_2 + 10080*bs_3*x47 + 10080*bs_4*x48 + 5040*bx_1_1 + bx_1_2*x159 + bx_1_3*x160 + bx_1_4*x161 + x130*x167 + x131*x168 + x151*x169 + x152*x170 + x156*x186 + 6*x157*x173 + x162*x171 + 42*x172*x43 + 1260*x179*x29 + 252*x180*x36 + x185*x67 + x66*(bx_0_3 + bx_0_4*x158)) + (1.0/5040.0)*x157*x171 + x16*(bx_1_1 + bx_1_2*x164 + bx_1_3*x165 + bx_1_4*x166) + (1.0/15120.0)*x163*x4 + x167*x23 + x168*x30 + x169*x37 + x170*x44 + x51*(x118*x173 + 120960*x172) - x59*(720*by_6_1 + by_6_2*x174 + by_6_3*x175 + by_6_4*x176 + 12960*x*x178 + 6480*x15*x163 + 12960*x177) - x69*(5040*bx_5_1 + bx_5_2*x159 + bx_5_3*x160 + bx_5_4*x161 + 90720*x*x180 + x130*x171 + 45360*x15*x172 + 15120*x173*x22 + 90720*x179) + x91*(720*by_4_1 + by_4_2*x174 + by_4_3*x175 + by_4_4*x176 + 8640*x*x182 + x106*x184 + 4320*x15*x177 + 360*x163*x29 + 1440*x178*x22 + 8640*x181 + x183*x89);
		return;

	}
	default: {
		printf("Error: Unsupported multipole order %d\n", multipole_order);
		printf("Supported orders are 1 to 7\n");
		printf("Setting field values to zero.\n");
		// Reduced expressions
		*Bx_out = 0;
		*By_out = 0;
		*Bs_out = 0;
		return;
	}
	}
}

#endif // SPLINE_B_FIELD_EVAL_H
