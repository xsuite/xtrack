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
GPUFUN
void evaluate_B(const double x, const double y, const double s,
                const double *bs,
                const double *const *by,
                const double *const *bx,
                const double L,
                const int multipole_order,
                double *Bx_out, double *By_out, double *Bs_out){

		typedef struct {
			double coeffs[MAX_DEGREE + 1]; /* coeffs[i] = coefficient of x^i */
			int degree;
		} Poly;

		static Poly poly_scale(Poly p, double s) {
			for (int i = 0; i <= p.degree; i++) p.coeffs[i] *= s;
			return p;
		}

		static Poly poly_add(Poly a, Poly b) {
			Poly result = {0};
			result.degree = a.degree > b.degree ? a.degree : b.degree;
			for (int i = 0; i <= a.degree; i++) result.coeffs[i] += a.coeffs[i];
			for (int i = 0; i <= b.degree; i++) result.coeffs[i] += b.coeffs[i];
			return result;
		}

		static Poly poly_mul(Poly a, Poly b) {
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
		static Poly poly_compose(Poly f, Poly g) {
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

		Poly hermite_to_polynomial(double s_start, double s_end, const double coeffs[5]) {
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

		/* Evaluate polynomial at x via Horner's method */
		double poly_eval(Poly p, double x) {
			double result = p.coeffs[p.degree];
			for (int i = p.degree - 1; i >= 0; i--)
				result = result * x + p.coeffs[i];
			return result;
		}

	switch (multipole_order) {
	case 1{
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
		const double x3 = y*y;
		const double x4 = 3*bx_0_3;
		const double x5 = 6*x0;
		const double x6 = bx_0_2 + bx_0_4*x5 + s*x4;
		const double x7 = 2*s;
		const double x8 = 3*x0;
		const double x9 = 4*x1;

		// Reduced expressions
		*Bx_out = bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2 - x3*x6;
		*By_out = by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 - y*(bs_1 + bs_2*x7 + bs_3*x8 + bs_4*x9 + 2*x*x6);
		*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2 + x*(bx_0_1 + bx_0_2*x7 + bx_0_4*x9 + x0*x4) - 1.0/2.0*x3*(2*bs_2 + 6*bs_3*s + 2*bs_4*x5 + 6*x*(bx_0_3 + 4*bx_0_4*s)) + y*(by_0_1 + by_0_2*x7 + by_0_3*x8 + by_0_4*x9);
		return;

	}
	case 2{
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
		const double x3 = y*y*y;
		const double x4 = 3*s;
		const double x5 = 6*x0;
		const double x6 = by_1_2 + by_1_3*x4 + by_1_4*x5;
		const double x7 = bx_1_1*s;
		const double x8 = bx_1_2*x0;
		const double x9 = bx_1_3*x1;
		const double x10 = bx_1_4*x2;
		const double x11 = by_1_0 + by_1_1*s + by_1_2*x0 + by_1_3*x1 + by_1_4*x2;
		const double x12 = bx_0_2 + bx_0_3*x4 + bx_0_4*x5;
		const double x13 = bx_1_2 + bx_1_3*x4 + bx_1_4*x5;
		const double x14 = 4*x;
		const double x15 = y*y;
		const double x16 = (1.0/4.0)*x15;
		const double x17 = 4*s;
		const double x18 = 8*x1;
		const double x19 = x*x;
		const double x20 = 2*s;
		const double x21 = 3*x0;
		const double x22 = 4*x1;

		// Reduced expressions
		*Bx_out = bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2 + x*(bx_1_0 + x10 + x7 + x8 + x9) + x11*y - x16*(4*x12 + x13*x14) - 1.0/3.0*x3*x6;
		*By_out = by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 + x*x11 - 1.0/2.0*x15*(2*by_0_2 + 2*by_0_3*x4 + 2*by_0_4*x5 + 2*x*x6) - 1.0/2.0*y*(2*bs_1 + bs_2*x17 + bs_3*x5 + bs_4*x18 + 2*bx_1_0 + 2*x10 + x12*x14 + 2*x13*x19 + 2*x7 + 2*x8 + 2*x9);
		*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2 + x*(bx_0_1 + bx_0_2*x20 + bx_0_3*x21 + bx_0_4*x22) - x16*(4*bs_2 + 4*bs_3*x4 + 4*bs_4*x5 + 2*bx_1_1 + bx_1_2*x17 + bx_1_3*x5 + bx_1_4*x18 + 12*x*(bx_0_3 + bx_0_4*x17) + 6*x19*(bx_1_3 + bx_1_4*x17)) + (1.0/2.0)*x19*(bx_1_1 + bx_1_2*x20 + bx_1_3*x21 + bx_1_4*x22) - 1.0/6.0*x3*(6*by_0_3 + 6*by_0_4*x17 + 6*x*(by_1_3 + by_1_4*x17)) + y*(by_0_1 + by_0_2*x20 + by_0_3*x21 + by_0_4*x22 + x*(by_1_1 + by_1_2*x20 + by_1_3*x21 + by_1_4*x22));
		return;

	}
	case 3{
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
		const double x3 = bx_1_1*s;
		const double x4 = bx_1_2*x0;
		const double x5 = bx_1_3*x1;
		const double x6 = bx_1_4*x2;
		const double x7 = bx_2_1*s;
		const double x8 = bx_2_2*x0;
		const double x9 = bx_2_3*x1;
		const double x10 = bx_2_4*x2;
		const double x11 = bx_2_0 + x10 + x7 + x8 + x9;
		const double x12 = x*x;
		const double x13 = (1.0/2.0)*x12;
		const double x14 = 144*bx_0_4;
		const double x15 = 72*x12;
		const double x16 = 3*s;
		const double x17 = 6*x0;
		const double x18 = bx_2_2 + bx_2_3*x16 + bx_2_4*x17;
		const double x19 = (1.0/144.0)*y*y*y*y;
		const double x20 = by_1_2 + by_1_3*x16 + by_1_4*x17;
		const double x21 = by_2_2 + by_2_3*x16 + by_2_4*x17;
		const double x22 = 4*x;
		const double x23 = y*y*y;
		const double x24 = (1.0/12.0)*x23;
		const double x25 = by_1_1*s;
		const double x26 = by_1_2*x0;
		const double x27 = by_1_3*x1;
		const double x28 = by_1_4*x2;
		const double x29 = by_2_1*s;
		const double x30 = by_2_2*x0;
		const double x31 = by_2_3*x1;
		const double x32 = by_2_4*x2;
		const double x33 = by_2_0 + x29 + x30 + x31 + x32;
		const double x34 = 2*x;
		const double x35 = (1.0/2.0)*y;
		const double x36 = bx_0_2 + bx_0_3*x16 + bx_0_4*x17;
		const double x37 = bx_1_2 + bx_1_3*x16 + bx_1_4*x17;
		const double x38 = 12*x;
		const double x39 = 6*x12;
		const double x40 = y*y;
		const double x41 = (1.0/12.0)*x40;
		const double x42 = x*x*x;
		const double x43 = 4*s;
		const double x44 = 12*s;
		const double x45 = 18*x0;
		const double x46 = 24*x1;
		const double x47 = 6*x;
		const double x48 = 2*s;
		const double x49 = 3*x0;
		const double x50 = 4*x1;
		const double x51 = bx_2_1 + bx_2_2*x48 + bx_2_3*x49 + bx_2_4*x50;
		const double x52 = bx_1_3 + bx_1_4*x43;
		const double x53 = bx_2_3 + bx_2_4*x43;
		const double x54 = 8*x1;

		// Reduced expressions
		*Bx_out = bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2 + x*(bx_1_0 + x3 + x4 + x5 + x6) + x11*x13 + x19*(144*bx_1_4*x + bx_2_4*x15 + x14 + 24*x18) - x24*(4*x20 + x21*x22) + x35*(2*by_1_0 + 2*x25 + 2*x26 + 2*x27 + 2*x28 + x33*x34) - x41*(6*bx_2_0 + 6*x10 + x18*x39 + 12*x36 + x37*x38 + 6*x7 + 6*x8 + 6*x9);
		*By_out = by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 + x*(by_1_0 + x25 + x26 + x27 + x28) + x13*x33 + (1.0/36.0)*x23*(36*bs_3 + 36*bs_4*x43 + bx_1_4*x15 + 24*bx_2_4*x42 + x*x14 + 24*x*x18 + 24*x37) - 1.0/4.0*x40*(4*by_0_2 + 4*by_0_3*x16 + 4*by_0_4*x17 + 2*by_2_0 + 2*x12*x21 + x20*x22 + 2*x29 + 2*x30 + 2*x31 + 2*x32) - 1.0/6.0*y*(6*bs_1 + bs_2*x44 + bs_3*x45 + bs_4*x46 + 6*bx_1_0 + x11*x47 + 2*x18*x42 + 6*x3 + x36*x38 + x37*x39 + 6*x4 + 6*x5 + 6*x6);
		*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2 + x*(bx_0_1 + bx_0_2*x48 + bx_0_3*x49 + bx_0_4*x50) + x13*(bx_1_1 + bx_1_2*x48 + bx_1_3*x49 + bx_1_4*x50) + x19*(144*bs_4 + 72*x*x53 + 72*x52) - x24*(12*by_0_3 + 12*by_0_4*x43 + 2*by_2_1 + by_2_2*x43 + by_2_3*x17 + by_2_4*x54 + x38*(by_1_3 + by_1_4*x43) + x39*(by_2_3 + by_2_4*x43)) + x35*(2*by_0_1 + by_0_2*x43 + by_0_3*x17 + by_0_4*x54 + x12*(by_2_1 + by_2_2*x48 + by_2_3*x49 + by_2_4*x50) + x34*(by_1_1 + by_1_2*x48 + by_1_3*x49 + by_1_4*x50)) - x41*(12*bs_2 + 12*bs_3*x16 + 12*bs_4*x17 + 6*bx_1_1 + bx_1_2*x44 + bx_1_3*x45 + bx_1_4*x46 + 36*x*(bx_0_3 + bx_0_4*x43) + 18*x12*x52 + 6*x42*x53 + x47*x51) + (1.0/6.0)*x42*x51;
		return;

	}
	case 4{
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
		const double x3 = bx_1_1*s;
		const double x4 = bx_1_2*x0;
		const double x5 = bx_1_3*x1;
		const double x6 = bx_1_4*x2;
		const double x7 = bx_2_1*s;
		const double x8 = bx_2_2*x0;
		const double x9 = bx_2_3*x1;
		const double x10 = bx_2_4*x2;
		const double x11 = bx_2_0 + x10 + x7 + x8 + x9;
		const double x12 = x*x;
		const double x13 = (1.0/2.0)*x12;
		const double x14 = bx_3_1*s;
		const double x15 = bx_3_2*x0;
		const double x16 = bx_3_3*x1;
		const double x17 = bx_3_4*x2;
		const double x18 = bx_3_0 + x14 + x15 + x16 + x17;
		const double x19 = x*x*x;
		const double x20 = (1.0/6.0)*x19;
		const double x21 = 144*by_1_4;
		const double x22 = 144*x;
		const double x23 = 72*x12;
		const double x24 = 3*s;
		const double x25 = 6*x0;
		const double x26 = by_3_2 + by_3_3*x24 + by_3_4*x25;
		const double x27 = (1.0/720.0)*y*y*y*y*y;
		const double x28 = 576*bx_0_4;
		const double x29 = 288*x12;
		const double x30 = 96*x19;
		const double x31 = bx_2_2 + bx_2_3*x24 + bx_2_4*x25;
		const double x32 = bx_3_2 + bx_3_3*x24 + bx_3_4*x25;
		const double x33 = 96*x;
		const double x34 = y*y*y*y;
		const double x35 = (1.0/576.0)*x34;
		const double x36 = by_1_1*s;
		const double x37 = by_1_2*x0;
		const double x38 = by_1_3*x1;
		const double x39 = by_1_4*x2;
		const double x40 = by_2_1*s;
		const double x41 = by_2_2*x0;
		const double x42 = by_2_3*x1;
		const double x43 = by_2_4*x2;
		const double x44 = by_2_0 + x40 + x41 + x42 + x43;
		const double x45 = 6*x;
		const double x46 = by_3_1*s;
		const double x47 = by_3_2*x0;
		const double x48 = by_3_3*x1;
		const double x49 = by_3_4*x2;
		const double x50 = by_3_0 + x46 + x47 + x48 + x49;
		const double x51 = 3*x12;
		const double x52 = (1.0/6.0)*y;
		const double x53 = by_1_2 + by_1_3*x24 + by_1_4*x25;
		const double x54 = by_2_2 + by_2_3*x24 + by_2_4*x25;
		const double x55 = 12*x;
		const double x56 = 6*x12;
		const double x57 = y*y*y;
		const double x58 = (1.0/36.0)*x57;
		const double x59 = bx_0_2 + bx_0_3*x24 + bx_0_4*x25;
		const double x60 = bx_1_2 + bx_1_3*x24 + bx_1_4*x25;
		const double x61 = 48*x;
		const double x62 = 24*x12;
		const double x63 = 8*x19;
		const double x64 = 24*x;
		const double x65 = y*y;
		const double x66 = (1.0/48.0)*x65;
		const double x67 = 24*x19;
		const double x68 = x*x*x*x;
		const double x69 = 4*s;
		const double x70 = 48*s;
		const double x71 = 72*x0;
		const double x72 = 96*x1;
		const double x73 = 12*x12;
		const double x74 = 2*s;
		const double x75 = 3*x0;
		const double x76 = 4*x1;
		const double x77 = bx_2_1 + bx_2_2*x74 + bx_2_3*x75 + bx_2_4*x76;
		const double x78 = bx_3_1 + bx_3_2*x74 + bx_3_3*x75 + bx_3_4*x76;
		const double x79 = by_2_3 + by_2_4*x69;
		const double x80 = by_3_3 + by_3_4*x69;
		const double x81 = bx_1_3 + bx_1_4*x69;
		const double x82 = bx_2_3 + bx_2_4*x69;
		const double x83 = bx_3_3 + bx_3_4*x69;
		const double x84 = 12*s;
		const double x85 = 18*x0;
		const double x86 = 24*x1;
		const double x87 = by_3_1 + by_3_2*x74 + by_3_3*x75 + by_3_4*x76;

		// Reduced expressions
		*Bx_out = bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2 + x*(bx_1_0 + x3 + x4 + x5 + x6) + x11*x13 + x18*x20 + x27*(by_2_4*x22 + by_3_4*x23 + x21 + 24*x26) + x35*(576*bx_1_4*x + bx_2_4*x29 + bx_3_4*x30 + x28 + 96*x31 + x32*x33) + x52*(6*by_1_0 + 6*x36 + 6*x37 + 6*x38 + 6*x39 + x44*x45 + x50*x51) - x58*(6*by_3_0 + x26*x56 + 6*x46 + 6*x47 + 6*x48 + 6*x49 + 12*x53 + x54*x55) - x66*(24*bx_2_0 + 24*x10 + x18*x64 + x31*x62 + x32*x63 + 48*x59 + x60*x61 + 24*x7 + 24*x8 + 24*x9);
		*By_out = by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 + x*(by_1_0 + x36 + x37 + x38 + x39) + x13*x44 + x20*x50 + (1.0/144.0)*x34*(144*by_0_4 + by_2_4*x23 + by_3_4*x67 + x*x21 + x26*x64 + 24*x54) + (1.0/144.0)*x57*(144*bs_3 + 144*bs_4*x69 + bx_1_4*x29 + bx_2_4*x30 + 24*bx_3_0 + 24*bx_3_4*x68 + x*x28 + 48*x12*x32 + 24*x14 + 24*x15 + 24*x16 + 24*x17 + x31*x33 + 96*x60) - 1.0/12.0*x65*(12*by_0_2 + 12*by_0_3*x24 + 12*by_0_4*x25 + 6*by_2_0 + 2*x19*x26 + 6*x40 + 6*x41 + 6*x42 + 6*x43 + x45*x50 + x53*x55 + x54*x56) - 1.0/24.0*y*(24*bs_1 + bs_2*x70 + bs_3*x71 + bs_4*x72 + 24*bx_1_0 + x11*x64 + x18*x73 + 24*x3 + x31*x63 + 2*x32*x68 + 24*x4 + 24*x5 + x59*x61 + 24*x6 + x60*x62);
		*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2 + x*(bx_0_1 + bx_0_2*x74 + bx_0_3*x75 + bx_0_4*x76) + x13*(bx_1_1 + bx_1_2*x74 + bx_1_3*x75 + bx_1_4*x76) + x20*x77 + x27*(72*x*x80 + 72*x79) + x35*(576*bs_4 + 24*bx_3_1 + bx_3_2*x70 + bx_3_3*x71 + bx_3_4*x72 + 288*x*x82 + 144*x12*x83 + 288*x81) + x52*(6*by_0_1 + by_0_2*x84 + by_0_3*x85 + by_0_4*x86 + x19*x87 + x45*(by_1_1 + by_1_2*x74 + by_1_3*x75 + by_1_4*x76) + x51*(by_2_1 + by_2_2*x74 + by_2_3*x75 + by_2_4*x76)) - x58*(36*by_0_3 + 36*by_0_4*x69 + 6*by_2_1 + by_2_2*x84 + by_2_3*x85 + by_2_4*x86 + 36*x*(by_1_3 + by_1_4*x69) + 18*x12*x79 + 6*x19*x80 + x45*x87) - x66*(48*bs_2 + 48*bs_3*x24 + 48*bs_4*x25 + 24*bx_1_1 + bx_1_2*x70 + bx_1_3*x71 + bx_1_4*x72 + x22*(bx_0_3 + bx_0_4*x69) + x23*x81 + x64*x77 + x67*x82 + 6*x68*x83 + x73*x78) + (1.0/24.0)*x68*x78;
		return;

	}
	case 5{
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
		const double x3 = bx_1_1*s;
		const double x4 = bx_1_2*x0;
		const double x5 = bx_1_3*x1;
		const double x6 = bx_1_4*x2;
		const double x7 = bx_2_1*s;
		const double x8 = bx_2_2*x0;
		const double x9 = bx_2_3*x1;
		const double x10 = bx_2_4*x2;
		const double x11 = bx_2_0 + x10 + x7 + x8 + x9;
		const double x12 = x*x;
		const double x13 = (1.0/2.0)*x12;
		const double x14 = bx_3_1*s;
		const double x15 = bx_3_2*x0;
		const double x16 = bx_3_3*x1;
		const double x17 = bx_3_4*x2;
		const double x18 = bx_3_0 + x14 + x15 + x16 + x17;
		const double x19 = x*x*x;
		const double x20 = (1.0/6.0)*x19;
		const double x21 = bx_4_1*s;
		const double x22 = bx_4_2*x0;
		const double x23 = bx_4_3*x1;
		const double x24 = bx_4_4*x2;
		const double x25 = bx_4_0 + x21 + x22 + x23 + x24;
		const double x26 = x*x*x*x;
		const double x27 = (1.0/24.0)*x26;
		const double x28 = 8640*bx_2_4;
		const double x29 = 4320*x12;
		const double x30 = 3*s;
		const double x31 = 6*x0;
		const double x32 = bx_4_2 + bx_4_3*x30 + bx_4_4*x31;
		const double x33 = (1.0/86400.0)*y*y*y*y*y*y;
		const double x34 = 576*by_1_4;
		const double x35 = 288*x12;
		const double x36 = 96*x19;
		const double x37 = by_3_2 + by_3_3*x30 + by_3_4*x31;
		const double x38 = by_4_2 + by_4_3*x30 + by_4_4*x31;
		const double x39 = 96*x;
		const double x40 = y*y*y*y*y;
		const double x41 = (1.0/2880.0)*x40;
		const double x42 = by_1_1*s;
		const double x43 = by_1_2*x0;
		const double x44 = by_1_3*x1;
		const double x45 = by_1_4*x2;
		const double x46 = by_2_1*s;
		const double x47 = by_2_2*x0;
		const double x48 = by_2_3*x1;
		const double x49 = by_2_4*x2;
		const double x50 = by_2_0 + x46 + x47 + x48 + x49;
		const double x51 = 24*x;
		const double x52 = by_3_1*s;
		const double x53 = by_3_2*x0;
		const double x54 = by_3_3*x1;
		const double x55 = by_3_4*x2;
		const double x56 = by_3_0 + x52 + x53 + x54 + x55;
		const double x57 = 12*x12;
		const double x58 = by_4_1*s;
		const double x59 = by_4_2*x0;
		const double x60 = by_4_3*x1;
		const double x61 = by_4_4*x2;
		const double x62 = by_4_0 + x58 + x59 + x60 + x61;
		const double x63 = 4*x19;
		const double x64 = (1.0/24.0)*y;
		const double x65 = 2880*bx_0_4;
		const double x66 = 1440*x12;
		const double x67 = 480*x19;
		const double x68 = 120*x26;
		const double x69 = bx_2_2 + bx_2_3*x30 + bx_2_4*x31;
		const double x70 = bx_3_2 + bx_3_3*x30 + bx_3_4*x31;
		const double x71 = 480*x;
		const double x72 = 240*x12;
		const double x73 = y*y*y*y;
		const double x74 = (1.0/2880.0)*x73;
		const double x75 = by_1_2 + by_1_3*x30 + by_1_4*x31;
		const double x76 = by_2_2 + by_2_3*x30 + by_2_4*x31;
		const double x77 = 48*x;
		const double x78 = 24*x12;
		const double x79 = 8*x19;
		const double x80 = y*y*y;
		const double x81 = (1.0/144.0)*x80;
		const double x82 = bx_0_2 + bx_0_3*x30 + bx_0_4*x31;
		const double x83 = bx_1_2 + bx_1_3*x30 + bx_1_4*x31;
		const double x84 = 240*x;
		const double x85 = 120*x12;
		const double x86 = 40*x19;
		const double x87 = 10*x26;
		const double x88 = 120*x;
		const double x89 = 60*x12;
		const double x90 = y*y;
		const double x91 = (1.0/240.0)*x90;
		const double x92 = 720*x;
		const double x93 = x*x*x*x*x;
		const double x94 = 4*s;
		const double x95 = 240*s;
		const double x96 = 360*x0;
		const double x97 = 480*x1;
		const double x98 = 20*x19;
		const double x99 = 2*s;
		const double x100 = 3*x0;
		const double x101 = 4*x1;
		const double x102 = bx_2_1 + bx_2_2*x99 + bx_2_3*x100 + bx_2_4*x101;
		const double x103 = bx_3_1 + bx_3_2*x99 + bx_3_3*x100 + bx_3_4*x101;
		const double x104 = bx_4_1 + bx_4_2*x99 + bx_4_3*x100 + bx_4_4*x101;
		const double x105 = bx_3_3 + bx_3_4*x94;
		const double x106 = bx_4_3 + bx_4_4*x94;
		const double x107 = 48*s;
		const double x108 = 72*x0;
		const double x109 = 96*x1;
		const double x110 = by_2_3 + by_2_4*x94;
		const double x111 = by_3_3 + by_3_4*x94;
		const double x112 = by_4_3 + by_4_4*x94;
		const double x113 = bx_1_3 + bx_1_4*x94;
		const double x114 = bx_2_3 + bx_2_4*x94;
		const double x115 = by_4_1 + by_4_2*x99 + by_4_3*x100 + by_4_4*x101;
		const double x116 = by_3_1 + by_3_2*x99 + by_3_3*x100 + by_3_4*x101;

		// Reduced expressions
		*Bx_out = bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2 + x*(bx_1_0 + x3 + x4 + x5 + x6) + x11*x13 + x18*x20 + x25*x27 - x33*(8640*bx_3_4*x + bx_4_4*x29 + x28 + 720*x32) + x41*(576*by_2_4*x + by_3_4*x35 + by_4_4*x36 + x34 + 96*x37 + x38*x39) + x64*(24*by_1_0 + 24*x42 + 24*x43 + 24*x44 + 24*x45 + x50*x51 + x56*x57 + x62*x63) + x74*(2880*bx_1_4*x + bx_2_4*x66 + bx_3_4*x67 + 120*bx_4_0 + bx_4_4*x68 + 120*x21 + 120*x22 + 120*x23 + 120*x24 + x32*x72 + x65 + 480*x69 + x70*x71) - x81*(24*by_3_0 + x37*x78 + x38*x79 + x51*x62 + 24*x52 + 24*x53 + 24*x54 + 24*x55 + 48*x75 + x76*x77) - x91*(120*bx_2_0 + 120*x10 + x18*x88 + x25*x89 + x32*x87 + x69*x85 + 120*x7 + x70*x86 + 120*x8 + 240*x82 + x83*x84 + 120*x9);
		*By_out = by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 + x*(by_1_0 + x42 + x43 + x44 + x45) + x13*x50 + x20*x56 + x27*x62 - 1.0/14400.0*x40*(8640*bx_1_4 + bx_3_4*x29 + 1440*bx_4_4*x19 + x*x28 + x32*x92 + 720*x70) + (1.0/576.0)*x73*(576*by_0_4 + by_2_4*x35 + by_3_4*x36 + 24*by_4_0 + 24*by_4_4*x26 + x*x34 + 48*x12*x38 + x37*x39 + 24*x58 + 24*x59 + 24*x60 + 24*x61 + 96*x76) + (1.0/720.0)*x80*(720*bs_3 + 720*bs_4*x94 + bx_1_4*x66 + bx_2_4*x67 + 120*bx_3_0 + bx_3_4*x68 + 24*bx_4_4*x93 + x*x65 + 120*x14 + 120*x15 + 120*x16 + 120*x17 + 80*x19*x32 + x25*x88 + x69*x71 + x70*x72 + 480*x83) - 1.0/48.0*x90*(48*by_0_2 + 48*by_0_3*x30 + 48*by_0_4*x31 + 24*by_2_0 + 2*x26*x38 + x37*x79 + 24*x46 + 24*x47 + 24*x48 + 24*x49 + x51*x56 + x57*x62 + x75*x77 + x76*x78) - 1.0/120.0*y*(120*bs_1 + bs_2*x95 + bs_3*x96 + bs_4*x97 + 120*bx_1_0 + x11*x88 + x18*x89 + x25*x98 + 120*x3 + 2*x32*x93 + 120*x4 + 120*x5 + 120*x6 + x69*x86 + x70*x87 + x82*x84 + x83*x85);
		*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2 + x*(bx_0_1 + bx_0_2*x99 + bx_0_3*x100 + bx_0_4*x101) + x102*x20 + x103*x27 + (1.0/120.0)*x104*x93 + x13*(bx_1_1 + bx_1_2*x99 + bx_1_3*x100 + bx_1_4*x101) - x33*(2160*x*x106 + 2160*x105) + x41*(24*by_4_1 + by_4_2*x107 + by_4_3*x108 + by_4_4*x109 + 288*x*x111 + 288*x110 + 144*x112*x12) + x64*(24*by_0_1 + by_0_2*x107 + by_0_3*x108 + by_0_4*x109 + x115*x26 + x116*x63 + x51*(by_1_1 + by_1_2*x99 + by_1_3*x100 + by_1_4*x101) + x57*(by_2_1 + by_2_2*x99 + by_2_3*x100 + by_2_4*x101)) + x74*(2880*bs_4 + 120*bx_3_1 + bx_3_2*x95 + bx_3_3*x96 + bx_3_4*x97 + 1440*x*x114 + x104*x88 + 720*x105*x12 + 240*x106*x19 + 1440*x113) - x81*(144*by_0_3 + 144*by_0_4*x94 + 24*by_2_1 + by_2_2*x107 + by_2_3*x108 + by_2_4*x109 + 144*x*(by_1_3 + by_1_4*x94) + 72*x110*x12 + 24*x111*x19 + 6*x112*x26 + x115*x57 + x116*x51) - x91*(240*bs_2 + 240*bs_3*x30 + 240*bs_4*x31 + 120*bx_1_1 + bx_1_2*x95 + bx_1_3*x96 + bx_1_4*x97 + x102*x88 + x103*x89 + x104*x98 + 30*x105*x26 + 6*x106*x93 + 360*x113*x12 + 120*x114*x19 + x92*(bx_0_3 + bx_0_4*x94));
		return;

	}
	case 6{
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
		const double x3 = bx_1_1*s;
		const double x4 = bx_1_2*x0;
		const double x5 = bx_1_3*x1;
		const double x6 = bx_1_4*x2;
		const double x7 = bx_2_1*s;
		const double x8 = bx_2_2*x0;
		const double x9 = bx_2_3*x1;
		const double x10 = bx_2_4*x2;
		const double x11 = bx_2_0 + x10 + x7 + x8 + x9;
		const double x12 = x*x;
		const double x13 = (1.0/2.0)*x12;
		const double x14 = bx_3_1*s;
		const double x15 = bx_3_2*x0;
		const double x16 = bx_3_3*x1;
		const double x17 = bx_3_4*x2;
		const double x18 = bx_3_0 + x14 + x15 + x16 + x17;
		const double x19 = x*x*x;
		const double x20 = (1.0/6.0)*x19;
		const double x21 = bx_4_1*s;
		const double x22 = bx_4_2*x0;
		const double x23 = bx_4_3*x1;
		const double x24 = bx_4_4*x2;
		const double x25 = bx_4_0 + x21 + x22 + x23 + x24;
		const double x26 = x*x*x*x;
		const double x27 = (1.0/24.0)*x26;
		const double x28 = bx_5_1*s;
		const double x29 = bx_5_2*x0;
		const double x30 = bx_5_3*x1;
		const double x31 = bx_5_4*x2;
		const double x32 = bx_5_0 + x28 + x29 + x30 + x31;
		const double x33 = x*x*x*x*x;
		const double x34 = (1.0/120.0)*x33;
		const double x35 = 8640*by_3_4;
		const double x36 = 8640*x;
		const double x37 = 4320*x12;
		const double x38 = 3*s;
		const double x39 = 6*x0;
		const double x40 = by_5_2 + by_5_3*x38 + by_5_4*x39;
		const double x41 = (1.0/604800.0)*y*y*y*y*y*y*y;
		const double x42 = 51840*bx_2_4;
		const double x43 = 25920*x12;
		const double x44 = 8640*x19;
		const double x45 = bx_4_2 + bx_4_3*x38 + bx_4_4*x39;
		const double x46 = bx_5_2 + bx_5_3*x38 + bx_5_4*x39;
		const double x47 = 4320*x;
		const double x48 = y*y*y*y*y*y;
		const double x49 = (1.0/518400.0)*x48;
		const double x50 = 2880*by_1_4;
		const double x51 = 2880*x;
		const double x52 = by_5_1*s;
		const double x53 = 1440*x12;
		const double x54 = 480*x19;
		const double x55 = by_5_2*x0;
		const double x56 = by_5_3*x1;
		const double x57 = by_5_4*x2;
		const double x58 = 120*x26;
		const double x59 = by_3_2 + by_3_3*x38 + by_3_4*x39;
		const double x60 = by_4_2 + by_4_3*x38 + by_4_4*x39;
		const double x61 = 480*x;
		const double x62 = 240*x12;
		const double x63 = y*y*y*y*y;
		const double x64 = (1.0/14400.0)*x63;
		const double x65 = by_1_1*s;
		const double x66 = by_1_2*x0;
		const double x67 = by_1_3*x1;
		const double x68 = by_1_4*x2;
		const double x69 = by_2_1*s;
		const double x70 = by_2_2*x0;
		const double x71 = by_2_3*x1;
		const double x72 = by_2_4*x2;
		const double x73 = by_2_0 + x69 + x70 + x71 + x72;
		const double x74 = 120*x;
		const double x75 = by_3_1*s;
		const double x76 = by_3_2*x0;
		const double x77 = by_3_3*x1;
		const double x78 = by_3_4*x2;
		const double x79 = by_3_0 + x75 + x76 + x77 + x78;
		const double x80 = 60*x12;
		const double x81 = by_4_1*s;
		const double x82 = by_4_2*x0;
		const double x83 = by_4_3*x1;
		const double x84 = by_4_4*x2;
		const double x85 = by_4_0 + x81 + x82 + x83 + x84;
		const double x86 = 20*x19;
		const double x87 = by_5_0 + x52 + x55 + x56 + x57;
		const double x88 = 5*x26;
		const double x89 = (1.0/120.0)*y;
		const double x90 = 17280*bx_0_4;
		const double x91 = 8640*x12;
		const double x92 = 2880*x19;
		const double x93 = 720*x26;
		const double x94 = 144*x33;
		const double x95 = bx_2_2 + bx_2_3*x38 + bx_2_4*x39;
		const double x96 = bx_3_2 + bx_3_3*x38 + bx_3_4*x39;
		const double x97 = 720*x;
		const double x98 = y*y*y*y;
		const double x99 = (1.0/17280.0)*x98;
		const double x100 = by_1_2 + by_1_3*x38 + by_1_4*x39;
		const double x101 = by_2_2 + by_2_3*x38 + by_2_4*x39;
		const double x102 = 240*x;
		const double x103 = 120*x12;
		const double x104 = 40*x19;
		const double x105 = 10*x26;
		const double x106 = y*y*y;
		const double x107 = (1.0/720.0)*x106;
		const double x108 = bx_0_2 + bx_0_3*x38 + bx_0_4*x39;
		const double x109 = bx_1_2 + bx_1_3*x38 + bx_1_4*x39;
		const double x110 = 1440*x;
		const double x111 = 720*x12;
		const double x112 = 240*x19;
		const double x113 = 60*x26;
		const double x114 = 12*x33;
		const double x115 = 360*x12;
		const double x116 = 120*x19;
		const double x117 = y*y;
		const double x118 = (1.0/1440.0)*x117;
		const double x119 = 1440*x19;
		const double x120 = 2160*x12;
		const double x121 = x*x*x*x*x*x;
		const double x122 = 4*s;
		const double x123 = 1440*s;
		const double x124 = 2160*x0;
		const double x125 = 2880*x1;
		const double x126 = 30*x26;
		const double x127 = 2*s;
		const double x128 = 3*x0;
		const double x129 = 4*x1;
		const double x130 = bx_2_1 + bx_2_2*x127 + bx_2_3*x128 + bx_2_4*x129;
		const double x131 = bx_3_1 + bx_3_2*x127 + bx_3_3*x128 + bx_3_4*x129;
		const double x132 = bx_4_1 + bx_4_2*x127 + bx_4_3*x128 + bx_4_4*x129;
		const double x133 = bx_5_1 + bx_5_2*x127 + bx_5_3*x128 + bx_5_4*x129;
		const double x134 = by_4_3 + by_4_4*x122;
		const double x135 = by_5_3 + by_5_4*x122;
		const double x136 = bx_3_3 + bx_3_4*x122;
		const double x137 = bx_4_3 + bx_4_4*x122;
		const double x138 = bx_5_3 + bx_5_4*x122;
		const double x139 = 240*s;
		const double x140 = 360*x0;
		const double x141 = 480*x1;
		const double x142 = by_2_3 + by_2_4*x122;
		const double x143 = by_3_3 + by_3_4*x122;
		const double x144 = by_5_1 + by_5_2*x127 + by_5_3*x128 + by_5_4*x129;
		const double x145 = bx_1_3 + bx_1_4*x122;
		const double x146 = bx_2_3 + bx_2_4*x122;
		const double x147 = by_3_1 + by_3_2*x127 + by_3_3*x128 + by_3_4*x129;
		const double x148 = by_4_1 + by_4_2*x127 + by_4_3*x128 + by_4_4*x129;

		// Reduced expressions
		*Bx_out = bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2 + x*(bx_1_0 + x3 + x4 + x5 + x6) - x107*(120*by_3_0 + 240*x100 + x101*x102 + x103*x59 + x104*x60 + x105*x40 + x74*x85 + 120*x75 + 120*x76 + 120*x77 + 120*x78 + x80*x87) + x11*x13 - x118*(720*bx_2_0 + 720*x10 + 1440*x108 + x109*x110 + x111*x95 + x112*x96 + x113*x45 + x114*x46 + x115*x25 + x116*x32 + x18*x97 + 720*x7 + 720*x8 + 720*x9) + x18*x20 + x25*x27 + x32*x34 - x41*(by_4_4*x36 + by_5_4*x37 + x35 + 720*x40) - x49*(51840*bx_3_4*x + bx_4_4*x43 + bx_5_4*x44 + x42 + 4320*x45 + x46*x47) + x64*(by_2_4*x51 + by_3_4*x53 + by_4_4*x54 + 120*by_5_0 + by_5_4*x58 + x40*x62 + x50 + 120*x52 + 120*x55 + 120*x56 + 120*x57 + 480*x59 + x60*x61) + x89*(120*by_1_0 + 120*x65 + 120*x66 + 120*x67 + 120*x68 + x73*x74 + x79*x80 + x85*x86 + x87*x88) + x99*(17280*bx_1_4*x + bx_2_4*x91 + bx_3_4*x92 + 720*bx_4_0 + bx_4_4*x93 + bx_5_4*x94 + 720*x21 + 720*x22 + 720*x23 + 720*x24 + x32*x97 + x45*x53 + x46*x54 + x51*x96 + x90 + 2880*x95);
		*By_out = by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 + x*(by_1_0 + x65 + x66 + x67 + x68) + (1.0/4320.0)*x106*(4320*bs_3 + 4320*bs_4*x122 + bx_1_4*x91 + bx_2_4*x92 + 720*bx_3_0 + bx_3_4*x93 + bx_4_4*x94 + 24*bx_5_4*x121 + x*x90 + 2880*x109 + x115*x32 + 720*x14 + 720*x15 + 720*x16 + 720*x17 + x25*x97 + x45*x54 + x46*x58 + x51*x95 + x53*x96) - 1.0/240.0*x117*(240*by_0_2 + 240*by_0_3*x38 + 240*by_0_4*x39 + 120*by_2_0 + x100*x102 + x101*x103 + x104*x59 + x105*x60 + 2*x33*x40 + 120*x69 + 120*x70 + 120*x71 + 120*x72 + x74*x79 + x80*x85 + x86*x87) + x13*x73 + x20*x79 + x27*x85 + x34*x87 - 1.0/86400.0*x48*(8640*by_2_4 + by_4_4*x37 + by_5_4*x119 + x*x35 + x40*x97 + 720*x60) - 1.0/86400.0*x63*(51840*bx_1_4 + bx_3_4*x43 + bx_4_4*x44 + 720*bx_5_0 + 2160*bx_5_4*x26 + x*x42 + x120*x46 + 720*x28 + 720*x29 + 720*x30 + 720*x31 + x45*x47 + 4320*x96) + (1.0/2880.0)*x98*(2880*by_0_4 + by_2_4*x53 + by_3_4*x54 + 120*by_4_0 + by_4_4*x58 + 24*by_5_4*x33 + x*x50 + 480*x101 + 80*x19*x40 + x59*x61 + x60*x62 + x74*x87 + 120*x81 + 120*x82 + 120*x83 + 120*x84) - 1.0/720.0*y*(720*bs_1 + bs_2*x123 + bs_3*x124 + bs_4*x125 + 720*bx_1_0 + x108*x110 + x109*x111 + x11*x97 + x112*x95 + x113*x96 + x114*x45 + x115*x18 + x116*x25 + 2*x121*x46 + x126*x32 + 720*x3 + 720*x4 + 720*x5 + 720*x6);
		*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2 + x*(bx_0_1 + bx_0_2*x127 + bx_0_3*x128 + bx_0_4*x129) - x107*(720*by_0_3 + 720*by_0_4*x122 + 120*by_2_1 + by_2_2*x139 + by_2_3*x140 + by_2_4*x141 + x115*x142 + x116*x143 + x126*x134 + 6*x135*x33 + x144*x86 + x147*x74 + x148*x80 + x97*(by_1_3 + by_1_4*x122)) - x118*(1440*bs_2 + 1440*bs_3*x38 + 1440*bs_4*x39 + 720*bx_1_1 + bx_1_2*x123 + bx_1_3*x124 + bx_1_4*x125 + x115*x131 + x116*x132 + x120*x145 + 6*x121*x138 + x126*x133 + x130*x97 + 180*x136*x26 + 36*x137*x33 + 720*x146*x19 + x47*(bx_0_3 + bx_0_4*x122)) + (1.0/720.0)*x121*x133 + x13*(bx_1_1 + bx_1_2*x127 + bx_1_3*x128 + bx_1_4*x129) + x130*x20 + x131*x27 + x132*x34 - x41*(2160*x*x135 + 2160*x134) - x49*(720*bx_5_1 + bx_5_2*x123 + bx_5_3*x124 + bx_5_4*x125 + 12960*x*x137 + 6480*x12*x138 + 12960*x136) + x64*(120*by_4_1 + by_4_2*x139 + by_4_3*x140 + by_4_4*x141 + x110*x143 + x111*x134 + x112*x135 + 1440*x142 + x144*x74) + x89*(120*by_0_1 + by_0_2*x139 + by_0_3*x140 + by_0_4*x141 + x144*x33 + x147*x86 + x148*x88 + x74*(by_1_1 + by_1_2*x127 + by_1_3*x128 + by_1_4*x129) + x80*(by_2_1 + by_2_2*x127 + by_2_3*x128 + by_2_4*x129)) + x99*(17280*bs_4 + 720*bx_3_1 + bx_3_2*x123 + bx_3_3*x124 + bx_3_4*x125 + x115*x133 + x119*x137 + x132*x97 + x136*x37 + 360*x138*x26 + 8640*x145 + x146*x36);
		return;

	}
	case 7{
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
		const double x3 = bx_1_1*s;
		const double x4 = bx_1_2*x0;
		const double x5 = bx_1_3*x1;
		const double x6 = bx_1_4*x2;
		const double x7 = bx_2_1*s;
		const double x8 = bx_2_2*x0;
		const double x9 = bx_2_3*x1;
		const double x10 = bx_2_4*x2;
		const double x11 = bx_2_0 + x10 + x7 + x8 + x9;
		const double x12 = x*x;
		const double x13 = (1.0/2.0)*x12;
		const double x14 = bx_3_1*s;
		const double x15 = bx_3_2*x0;
		const double x16 = bx_3_3*x1;
		const double x17 = bx_3_4*x2;
		const double x18 = bx_3_0 + x14 + x15 + x16 + x17;
		const double x19 = x*x*x;
		const double x20 = (1.0/6.0)*x19;
		const double x21 = bx_4_1*s;
		const double x22 = bx_4_2*x0;
		const double x23 = bx_4_3*x1;
		const double x24 = bx_4_4*x2;
		const double x25 = bx_4_0 + x21 + x22 + x23 + x24;
		const double x26 = x*x*x*x;
		const double x27 = (1.0/24.0)*x26;
		const double x28 = bx_5_1*s;
		const double x29 = bx_5_2*x0;
		const double x30 = bx_5_3*x1;
		const double x31 = bx_5_4*x2;
		const double x32 = bx_5_0 + x28 + x29 + x30 + x31;
		const double x33 = x*x*x*x*x;
		const double x34 = (1.0/120.0)*x33;
		const double x35 = bx_6_1*s;
		const double x36 = bx_6_2*x0;
		const double x37 = bx_6_3*x1;
		const double x38 = bx_6_4*x2;
		const double x39 = bx_6_0 + x35 + x36 + x37 + x38;
		const double x40 = x*x*x*x*x*x;
		const double x41 = (1.0/720.0)*x40;
		const double x42 = 725760*bx_4_4;
		const double x43 = 362880*x12;
		const double x44 = 3*s;
		const double x45 = 6*x0;
		const double x46 = bx_6_2 + bx_6_3*x44 + bx_6_4*x45;
		const double x47 = (1.0/203212800.0)*y*y*y*y*y*y*y*y;
		const double x48 = 51840*by_3_4;
		const double x49 = 25920*x12;
		const double x50 = 8640*x19;
		const double x51 = by_5_2 + by_5_3*x44 + by_5_4*x45;
		const double x52 = by_6_2 + by_6_3*x44 + by_6_4*x45;
		const double x53 = 4320*x;
		const double x54 = y*y*y*y*y*y*y;
		const double x55 = (1.0/3628800.0)*x54;
		const double x56 = 362880*bx_2_4;
		const double x57 = 181440*x12;
		const double x58 = 60480*x19;
		const double x59 = 15120*x26;
		const double x60 = bx_4_2 + bx_4_3*x44 + bx_4_4*x45;
		const double x61 = bx_5_2 + bx_5_3*x44 + bx_5_4*x45;
		const double x62 = 30240*x;
		const double x63 = 15120*x12;
		const double x64 = y*y*y*y*y*y;
		const double x65 = (1.0/3628800.0)*x64;
		const double x66 = 17280*by_1_4;
		const double x67 = by_5_1*s;
		const double x68 = 8640*x12;
		const double x69 = 2880*x19;
		const double x70 = by_5_2*x0;
		const double x71 = by_5_3*x1;
		const double x72 = by_5_4*x2;
		const double x73 = 720*x26;
		const double x74 = 144*x33;
		const double x75 = by_3_2 + by_3_3*x44 + by_3_4*x45;
		const double x76 = by_4_2 + by_4_3*x44 + by_4_4*x45;
		const double x77 = 2880*x;
		const double x78 = 1440*x12;
		const double x79 = 480*x19;
		const double x80 = by_6_1*s;
		const double x81 = by_6_2*x0;
		const double x82 = by_6_3*x1;
		const double x83 = by_6_4*x2;
		const double x84 = by_6_0 + x80 + x81 + x82 + x83;
		const double x85 = 720*x;
		const double x86 = y*y*y*y*y;
		const double x87 = (1.0/86400.0)*x86;
		const double x88 = by_1_1*s;
		const double x89 = by_1_2*x0;
		const double x90 = by_1_3*x1;
		const double x91 = by_1_4*x2;
		const double x92 = by_2_1*s;
		const double x93 = by_2_2*x0;
		const double x94 = by_2_3*x1;
		const double x95 = by_2_4*x2;
		const double x96 = by_2_0 + x92 + x93 + x94 + x95;
		const double x97 = by_3_1*s;
		const double x98 = by_3_2*x0;
		const double x99 = by_3_3*x1;
		const double x100 = by_3_4*x2;
		const double x101 = by_3_0 + x100 + x97 + x98 + x99;
		const double x102 = 360*x12;
		const double x103 = by_4_1*s;
		const double x104 = by_4_2*x0;
		const double x105 = by_4_3*x1;
		const double x106 = by_4_4*x2;
		const double x107 = by_4_0 + x103 + x104 + x105 + x106;
		const double x108 = 120*x19;
		const double x109 = by_5_0 + x67 + x70 + x71 + x72;
		const double x110 = 30*x26;
		const double x111 = 6*x33;
		const double x112 = (1.0/720.0)*y;
		const double x113 = 120960*bx_0_4;
		const double x114 = 120960*x;
		const double x115 = 60480*x12;
		const double x116 = 20160*x19;
		const double x117 = 5040*x26;
		const double x118 = 1008*x33;
		const double x119 = 168*x40;
		const double x120 = bx_2_2 + bx_2_3*x44 + bx_2_4*x45;
		const double x121 = bx_3_2 + bx_3_3*x44 + bx_3_4*x45;
		const double x122 = 20160*x;
		const double x123 = 10080*x12;
		const double x124 = 3360*x19;
		const double x125 = 840*x26;
		const double x126 = 5040*x;
		const double x127 = 2520*x12;
		const double x128 = y*y*y*y;
		const double x129 = (1.0/120960.0)*x128;
		const double x130 = by_1_2 + by_1_3*x44 + by_1_4*x45;
		const double x131 = by_2_2 + by_2_3*x44 + by_2_4*x45;
		const double x132 = 1440*x;
		const double x133 = 720*x12;
		const double x134 = 240*x19;
		const double x135 = 60*x26;
		const double x136 = 12*x33;
		const double x137 = y*y*y;
		const double x138 = (1.0/4320.0)*x137;
		const double x139 = bx_0_2 + bx_0_3*x44 + bx_0_4*x45;
		const double x140 = bx_1_2 + bx_1_3*x44 + bx_1_4*x45;
		const double x141 = 10080*x;
		const double x142 = 5040*x12;
		const double x143 = 1680*x19;
		const double x144 = 420*x26;
		const double x145 = 84*x33;
		const double x146 = 14*x40;
		const double x147 = 840*x19;
		const double x148 = 210*x26;
		const double x149 = y*y;
		const double x150 = (1.0/10080.0)*x149;
		const double x151 = 2160*x12;
		const double x152 = 5040*x19;
		const double x153 = x*x*x*x*x*x*x;
		const double x154 = 4*s;
		const double x155 = 10080*s;
		const double x156 = 15120*x0;
		const double x157 = 20160*x1;
		const double x158 = 42*x33;
		const double x159 = 2*s;
		const double x160 = 3*x0;
		const double x161 = 4*x1;
		const double x162 = bx_2_1 + bx_2_2*x159 + bx_2_3*x160 + bx_2_4*x161;
		const double x163 = bx_3_1 + bx_3_2*x159 + bx_3_3*x160 + bx_3_4*x161;
		const double x164 = bx_4_1 + bx_4_2*x159 + bx_4_3*x160 + bx_4_4*x161;
		const double x165 = bx_5_1 + bx_5_2*x159 + bx_5_3*x160 + bx_5_4*x161;
		const double x166 = bx_6_1 + bx_6_2*x159 + bx_6_3*x160 + bx_6_4*x161;
		const double x167 = bx_5_3 + bx_5_4*x154;
		const double x168 = bx_6_3 + bx_6_4*x154;
		const double x169 = 1440*s;
		const double x170 = 2160*x0;
		const double x171 = 2880*x1;
		const double x172 = by_4_3 + by_4_4*x154;
		const double x173 = by_5_3 + by_5_4*x154;
		const double x174 = by_6_3 + by_6_4*x154;
		const double x175 = bx_3_3 + bx_3_4*x154;
		const double x176 = bx_4_3 + bx_4_4*x154;
		const double x177 = by_2_3 + by_2_4*x154;
		const double x178 = by_3_3 + by_3_4*x154;
		const double x179 = by_5_1 + by_5_2*x159 + by_5_3*x160 + by_5_4*x161;
		const double x180 = by_6_1 + by_6_2*x159 + by_6_3*x160 + by_6_4*x161;
		const double x181 = bx_1_3 + bx_1_4*x154;
		const double x182 = bx_2_3 + bx_2_4*x154;
		const double x183 = by_3_1 + by_3_2*x159 + by_3_3*x160 + by_3_4*x161;
		const double x184 = by_4_1 + by_4_2*x159 + by_4_3*x160 + by_4_4*x161;

		// Reduced expressions
		*Bx_out = bx_0_0 + bx_0_1*s + bx_0_2*x0 + bx_0_3*x1 + bx_0_4*x2 + x*(bx_1_0 + x3 + x4 + x5 + x6) + x11*x13 + x112*(720*by_1_0 + x101*x102 + x107*x108 + x109*x110 + x111*x84 + x85*x96 + 720*x88 + 720*x89 + 720*x90 + 720*x91) + x129*(bx_1_4*x114 + bx_2_4*x115 + bx_3_4*x116 + 5040*bx_4_0 + bx_4_4*x117 + bx_5_4*x118 + bx_6_4*x119 + x113 + 20160*x120 + x121*x122 + x123*x60 + x124*x61 + x125*x46 + x126*x32 + x127*x39 + 5040*x21 + 5040*x22 + 5040*x23 + 5040*x24) - x138*(720*by_3_0 + 720*x100 + x102*x109 + x107*x85 + x108*x84 + 1440*x130 + x131*x132 + x133*x75 + x134*x76 + x135*x51 + x136*x52 + 720*x97 + 720*x98 + 720*x99) - x150*(5040*bx_2_0 + 5040*x10 + x120*x142 + x121*x143 + x126*x18 + x127*x25 + 10080*x139 + x140*x141 + x144*x60 + x145*x61 + x146*x46 + x147*x32 + x148*x39 + 5040*x7 + 5040*x8 + 5040*x9) + x18*x20 + x25*x27 + x32*x34 + x39*x41 + x47*(725760*bx_5_4*x + bx_6_4*x43 + x42 + 40320*x46) - x55*(51840*by_4_4*x + by_5_4*x49 + by_6_4*x50 + x48 + 4320*x51 + x52*x53) - x65*(362880*bx_3_4*x + bx_4_4*x57 + bx_5_4*x58 + 5040*bx_6_0 + bx_6_4*x59 + 5040*x35 + 5040*x36 + 5040*x37 + 5040*x38 + x46*x63 + x56 + 30240*x60 + x61*x62) + x87*(17280*by_2_4*x + by_3_4*x68 + by_4_4*x69 + 720*by_5_0 + by_5_4*x73 + by_6_4*x74 + x51*x78 + x52*x79 + x66 + 720*x67 + 720*x70 + 720*x71 + 720*x72 + 2880*x75 + x76*x77 + x84*x85);
		*By_out = by_0_0 + by_0_1*s + by_0_2*x0 + by_0_3*x1 + by_0_4*x2 + x*(by_1_0 + x88 + x89 + x90 + x91) + x101*x20 + x107*x27 + x109*x34 + (1.0/17280.0)*x128*(17280*by_0_4 + by_2_4*x68 + by_3_4*x69 + 720*by_4_0 + by_4_4*x73 + by_5_4*x74 + 24*by_6_4*x40 + x*x66 + x102*x84 + 720*x103 + 720*x104 + 720*x105 + 720*x106 + x109*x85 + 2880*x131 + 120*x26*x52 + x51*x79 + x75*x77 + x76*x78) + x13*x96 + (1.0/30240.0)*x137*(30240*bs_3 + 30240*bs_4*x154 + bx_1_4*x115 + bx_2_4*x116 + 5040*bx_3_0 + bx_3_4*x117 + bx_4_4*x118 + bx_5_4*x119 + 24*bx_6_4*x153 + x*x113 + x120*x122 + x121*x123 + x124*x60 + x125*x61 + x126*x25 + x127*x32 + 5040*x14 + 20160*x140 + x147*x39 + 5040*x15 + 5040*x16 + 5040*x17 + 168*x33*x46) - 1.0/1440.0*x149*(1440*by_0_2 + 1440*by_0_3*x44 + 1440*by_0_4*x45 + 720*by_2_0 + x101*x85 + x102*x107 + x108*x109 + x110*x84 + x130*x132 + x131*x133 + x134*x75 + x135*x76 + x136*x51 + 2*x40*x52 + 720*x92 + 720*x93 + 720*x94 + 720*x95) + x41*x84 + (1.0/25401600.0)*x54*(725760*bx_3_4 + bx_5_4*x43 + 120960*bx_6_4*x19 + x*x42 + 40320*x*x46 + 40320*x61) - 1.0/518400.0*x64*(51840*by_2_4 + by_4_4*x49 + by_5_4*x50 + 720*by_6_0 + 2160*by_6_4*x26 + x*x48 + x151*x52 + x51*x53 + 4320*x76 + 720*x80 + 720*x81 + 720*x82 + 720*x83) - 1.0/604800.0*x86*(362880*bx_1_4 + bx_3_4*x57 + bx_4_4*x58 + 5040*bx_5_0 + bx_5_4*x59 + 3024*bx_6_4*x33 + x*x56 + 30240*x121 + x126*x39 + x152*x46 + 5040*x28 + 5040*x29 + 5040*x30 + 5040*x31 + x60*x62 + x61*x63) - 1.0/5040.0*y*(5040*bs_1 + bs_2*x155 + bs_3*x156 + bs_4*x157 + 5040*bx_1_0 + x11*x126 + x120*x143 + x121*x144 + x127*x18 + x139*x141 + x140*x142 + x145*x60 + x146*x61 + x147*x25 + x148*x32 + 2*x153*x46 + x158*x39 + 5040*x3 + 5040*x4 + 5040*x5 + 5040*x6);
		*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2 + x*(bx_0_1 + bx_0_2*x159 + bx_0_3*x160 + bx_0_4*x161) + x112*(720*by_0_1 + by_0_2*x169 + by_0_3*x170 + by_0_4*x171 + x102*(by_2_1 + by_2_2*x159 + by_2_3*x160 + by_2_4*x161) + x108*x183 + x110*x184 + x111*x179 + x180*x40 + x85*(by_1_1 + by_1_2*x159 + by_1_3*x160 + by_1_4*x161)) + x129*(120960*bs_4 + 5040*bx_3_1 + bx_3_2*x155 + bx_3_3*x156 + bx_3_4*x157 + 60480*x*x182 + 30240*x12*x175 + x126*x164 + x127*x165 + x147*x166 + 2520*x167*x26 + 504*x168*x33 + 10080*x176*x19 + 60480*x181) + x13*(bx_1_1 + bx_1_2*x159 + bx_1_3*x160 + bx_1_4*x161) - x138*(4320*by_0_3 + 4320*by_0_4*x154 + 720*by_2_1 + by_2_2*x169 + by_2_3*x170 + by_2_4*x171 + x102*x184 + x108*x179 + x110*x180 + x151*x177 + 180*x172*x26 + 36*x173*x33 + 6*x174*x40 + 720*x178*x19 + x183*x85 + x53*(by_1_3 + by_1_4*x154)) - x150*(10080*bs_2 + 10080*bs_3*x44 + 10080*bs_4*x45 + 5040*bx_1_1 + bx_1_2*x155 + bx_1_3*x156 + bx_1_4*x157 + x126*x162 + x127*x163 + x147*x164 + x148*x165 + x152*x182 + 6*x153*x168 + x158*x166 + 42*x167*x40 + 1260*x175*x26 + 252*x176*x33 + x181*x63 + x62*(bx_0_3 + bx_0_4*x154)) + (1.0/5040.0)*x153*x166 + x162*x20 + x163*x27 + x164*x34 + x165*x41 + x47*(x114*x168 + 120960*x167) - x55*(720*by_6_1 + by_6_2*x169 + by_6_3*x170 + by_6_4*x171 + 12960*x*x173 + 6480*x12*x174 + 12960*x172) - x65*(5040*bx_5_1 + bx_5_2*x155 + bx_5_3*x156 + bx_5_4*x157 + 90720*x*x176 + 45360*x12*x167 + x126*x166 + 15120*x168*x19 + 90720*x175) + x87*(720*by_4_1 + by_4_2*x169 + by_4_3*x170 + by_4_4*x171 + 8640*x*x178 + x102*x180 + 4320*x12*x172 + 1440*x173*x19 + 360*x174*x26 + 8640*x177 + x179*x85);
		return;

	}
	default{
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
