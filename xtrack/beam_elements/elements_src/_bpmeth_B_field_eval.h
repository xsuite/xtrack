#include <stddef.h>
#include <stdio.h>

#ifndef XSUITE_BPMETH_B_FIELD_EVAL_H
#define XSUITE_BPMETH_B_FIELD_EVAL_H

// Auto-generated symbolic field expressions for B
void evaluate_B(const double x_array[], const double y_array[], const double s_array[], size_t n, const double *params_flat, const int multipole_order, double Bx_out[], double By_out[], double Bs_out[]){

	switch (multipole_order) {
	case 1:
		for (size_t ii = 0; ii < n; ++ii, ++x_array, ++y_array, ++s_array, ++Bx_out, ++By_out, ++Bs_out) {
			const double x = *x_array;
			const double y = *y_array;
			const double s = *s_array;

			// Parameter List
			const double *p = params_flat + ii * 15;
			const double a_1_0 = p[0];
			const double a_1_1 = p[1];
			const double a_1_2 = p[2];
			const double a_1_3 = p[3];
			const double a_1_4 = p[4];
			const double b_1_0 = p[5];
			const double b_1_1 = p[6];
			const double b_1_2 = p[7];
			const double b_1_3 = p[8];
			const double b_1_4 = p[9];
			const double bs_0 = p[10];
			const double bs_1 = p[11];
			const double bs_2 = p[12];
			const double bs_3 = p[13];
			const double bs_4 = p[14];

			// Common sub-expressions
			const double x0 = s*s;
			const double x1 = s*s*s;
			const double x2 = s*s*s*s;

			// Reduced expressions
			*Bx_out = a_1_0 + a_1_1*s + a_1_2*x0 + a_1_3*x1 + a_1_4*x2;
			*By_out = b_1_0 + b_1_1*s + b_1_2*x0 + b_1_3*x1 + b_1_4*x2;
			*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2;
		}
		return;

	case 2:
		for (size_t ii = 0; ii < n; ++ii, ++x_array, ++y_array, ++s_array, ++Bx_out, ++By_out, ++Bs_out) {
			const double x = *x_array;
			const double y = *y_array;
			const double s = *s_array;

			// Parameter List
			const double *p = params_flat + ii * 25;
			const double a_1_0 = p[0];
			const double a_1_1 = p[1];
			const double a_1_2 = p[2];
			const double a_1_3 = p[3];
			const double a_1_4 = p[4];
			const double a_2_0 = p[5];
			const double a_2_1 = p[6];
			const double a_2_2 = p[7];
			const double a_2_3 = p[8];
			const double a_2_4 = p[9];
			const double b_1_0 = p[10];
			const double b_1_1 = p[11];
			const double b_1_2 = p[12];
			const double b_1_3 = p[13];
			const double b_1_4 = p[14];
			const double b_2_0 = p[15];
			const double b_2_1 = p[16];
			const double b_2_2 = p[17];
			const double b_2_3 = p[18];
			const double b_2_4 = p[19];
			const double bs_0 = p[20];
			const double bs_1 = p[21];
			const double bs_2 = p[22];
			const double bs_3 = p[23];
			const double bs_4 = p[24];

			// Common sub-expressions
			const double x0 = s*s;
			const double x1 = s*s*s;
			const double x2 = s*s*s*s;
			const double x3 = b_2_0 + b_2_1*s + b_2_2*x0 + b_2_3*x1 + b_2_4*x2;

			// Reduced expressions
			*Bx_out = a_1_0 + a_1_1*s + a_1_2*x0 + a_1_3*x1 + a_1_4*x2 + x*(a_2_0 + a_2_1*s + a_2_2*x0 + a_2_3*x1 + a_2_4*x2) + x3*y;
			*By_out = b_1_0 + b_1_1*s + b_1_2*x0 + b_1_3*x1 + b_1_4*x2 + x*x3;
			*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2;
		}
		return;

	case 3:
		for (size_t ii = 0; ii < n; ++ii, ++x_array, ++y_array, ++s_array, ++Bx_out, ++By_out, ++Bs_out) {
			const double x = *x_array;
			const double y = *y_array;
			const double s = *s_array;

			// Parameter List
			const double *p = params_flat + ii * 35;
			const double a_1_0 = p[0];
			const double a_1_1 = p[1];
			const double a_1_2 = p[2];
			const double a_1_3 = p[3];
			const double a_1_4 = p[4];
			const double a_2_0 = p[5];
			const double a_2_1 = p[6];
			const double a_2_2 = p[7];
			const double a_2_3 = p[8];
			const double a_2_4 = p[9];
			const double a_3_0 = p[10];
			const double a_3_1 = p[11];
			const double a_3_2 = p[12];
			const double a_3_3 = p[13];
			const double a_3_4 = p[14];
			const double b_1_0 = p[15];
			const double b_1_1 = p[16];
			const double b_1_2 = p[17];
			const double b_1_3 = p[18];
			const double b_1_4 = p[19];
			const double b_2_0 = p[20];
			const double b_2_1 = p[21];
			const double b_2_2 = p[22];
			const double b_2_3 = p[23];
			const double b_2_4 = p[24];
			const double b_3_0 = p[25];
			const double b_3_1 = p[26];
			const double b_3_2 = p[27];
			const double b_3_3 = p[28];
			const double b_3_4 = p[29];
			const double bs_0 = p[30];
			const double bs_1 = p[31];
			const double bs_2 = p[32];
			const double bs_3 = p[33];
			const double bs_4 = p[34];

			// Common sub-expressions
			const double x0 = s*s;
			const double x1 = s*s*s;
			const double x2 = s*s*s*s;
			const double x3 = a_2_0 + a_2_1*s + a_2_2*x0 + a_2_3*x1 + a_2_4*x2;
			const double x4 = b_2_0 + b_2_1*s + b_2_2*x0 + b_2_3*x1 + b_2_4*x2;
			const double x5 = b_3_0 + b_3_1*s + b_3_2*x0 + b_3_3*x1 + b_3_4*x2;
			const double x6 = x*y;
			const double x7 = x*x;
			const double x8 = a_3_0 + a_3_1*s + a_3_2*x0 + a_3_3*x1 + a_3_4*x2;
			const double x9 = (1.0/2.0)*x8;

			// Reduced expressions
			*Bx_out = a_1_0 + a_1_1*s + a_1_2*x0 + a_1_3*x1 + a_1_4*x2 + x*x3 + x4*y + x5*x6 + x7*x9 - x9*y*y;
			*By_out = b_1_0 + b_1_1*s + b_1_2*x0 + b_1_3*x1 + b_1_4*x2 + x*x4 - x3*y + (1.0/2.0)*x5*x7 - x6*x8;
			*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2;
		}
		return;

	case 4:
		for (size_t ii = 0; ii < n; ++ii, ++x_array, ++y_array, ++s_array, ++Bx_out, ++By_out, ++Bs_out) {
			const double x = *x_array;
			const double y = *y_array;
			const double s = *s_array;

			// Parameter List
			const double *p = params_flat + ii * 45;
			const double a_1_0 = p[0];
			const double a_1_1 = p[1];
			const double a_1_2 = p[2];
			const double a_1_3 = p[3];
			const double a_1_4 = p[4];
			const double a_2_0 = p[5];
			const double a_2_1 = p[6];
			const double a_2_2 = p[7];
			const double a_2_3 = p[8];
			const double a_2_4 = p[9];
			const double a_3_0 = p[10];
			const double a_3_1 = p[11];
			const double a_3_2 = p[12];
			const double a_3_3 = p[13];
			const double a_3_4 = p[14];
			const double a_4_0 = p[15];
			const double a_4_1 = p[16];
			const double a_4_2 = p[17];
			const double a_4_3 = p[18];
			const double a_4_4 = p[19];
			const double b_1_0 = p[20];
			const double b_1_1 = p[21];
			const double b_1_2 = p[22];
			const double b_1_3 = p[23];
			const double b_1_4 = p[24];
			const double b_2_0 = p[25];
			const double b_2_1 = p[26];
			const double b_2_2 = p[27];
			const double b_2_3 = p[28];
			const double b_2_4 = p[29];
			const double b_3_0 = p[30];
			const double b_3_1 = p[31];
			const double b_3_2 = p[32];
			const double b_3_3 = p[33];
			const double b_3_4 = p[34];
			const double b_4_0 = p[35];
			const double b_4_1 = p[36];
			const double b_4_2 = p[37];
			const double b_4_3 = p[38];
			const double b_4_4 = p[39];
			const double bs_0 = p[40];
			const double bs_1 = p[41];
			const double bs_2 = p[42];
			const double bs_3 = p[43];
			const double bs_4 = p[44];

			// Common sub-expressions
			const double x0 = s*s;
			const double x1 = s*s*s;
			const double x2 = s*s*s*s;
			const double x3 = a_2_0 + a_2_1*s + a_2_2*x0 + a_2_3*x1 + a_2_4*x2;
			const double x4 = b_2_0 + b_2_1*s + b_2_2*x0 + b_2_3*x1 + b_2_4*x2;
			const double x5 = b_3_0 + b_3_1*s + b_3_2*x0 + b_3_3*x1 + b_3_4*x2;
			const double x6 = x*y;
			const double x7 = x*x;
			const double x8 = a_3_0 + a_3_1*s + a_3_2*x0 + a_3_3*x1 + a_3_4*x2;
			const double x9 = (1.0/2.0)*x8;
			const double x10 = x*x*x;
			const double x11 = a_4_0 + a_4_1*s + a_4_2*x0 + a_4_3*x1 + a_4_4*x2;
			const double x12 = y*y;
			const double x13 = b_4_0 + b_4_1*s + b_4_2*x0 + b_4_3*x1 + b_4_4*x2;
			const double x14 = (1.0/6.0)*x13;
			const double x15 = x*x12;
			const double x16 = (1.0/2.0)*x11;
			const double x17 = (1.0/2.0)*x13;
			const double x18 = x7*y;
			const double x19 = (1.0/2.0)*x5;

			// Reduced expressions
			*Bx_out = a_1_0 + a_1_1*s + a_1_2*x0 + a_1_3*x1 + a_1_4*x2 + x*x3 + (1.0/6.0)*x10*x11 - x12*x9 - x14*y*y*y - x15*x16 + x17*x18 + x4*y + x5*x6 + x7*x9;
			*By_out = b_1_0 + b_1_1*s + b_1_2*x0 + b_1_3*x1 + b_1_4*x2 + x*x4 + x10*x14 - x12*x19 - x15*x17 - x16*x18 + x19*x7 - x3*y - x6*x8;
			*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2;
		}
		return;

	case 5:
		for (size_t ii = 0; ii < n; ++ii, ++x_array, ++y_array, ++s_array, ++Bx_out, ++By_out, ++Bs_out) {
			const double x = *x_array;
			const double y = *y_array;
			const double s = *s_array;

			// Parameter List
			const double *p = params_flat + ii * 55;
			const double a_1_0 = p[0];
			const double a_1_1 = p[1];
			const double a_1_2 = p[2];
			const double a_1_3 = p[3];
			const double a_1_4 = p[4];
			const double a_2_0 = p[5];
			const double a_2_1 = p[6];
			const double a_2_2 = p[7];
			const double a_2_3 = p[8];
			const double a_2_4 = p[9];
			const double a_3_0 = p[10];
			const double a_3_1 = p[11];
			const double a_3_2 = p[12];
			const double a_3_3 = p[13];
			const double a_3_4 = p[14];
			const double a_4_0 = p[15];
			const double a_4_1 = p[16];
			const double a_4_2 = p[17];
			const double a_4_3 = p[18];
			const double a_4_4 = p[19];
			const double a_5_0 = p[20];
			const double a_5_1 = p[21];
			const double a_5_2 = p[22];
			const double a_5_3 = p[23];
			const double a_5_4 = p[24];
			const double b_1_0 = p[25];
			const double b_1_1 = p[26];
			const double b_1_2 = p[27];
			const double b_1_3 = p[28];
			const double b_1_4 = p[29];
			const double b_2_0 = p[30];
			const double b_2_1 = p[31];
			const double b_2_2 = p[32];
			const double b_2_3 = p[33];
			const double b_2_4 = p[34];
			const double b_3_0 = p[35];
			const double b_3_1 = p[36];
			const double b_3_2 = p[37];
			const double b_3_3 = p[38];
			const double b_3_4 = p[39];
			const double b_4_0 = p[40];
			const double b_4_1 = p[41];
			const double b_4_2 = p[42];
			const double b_4_3 = p[43];
			const double b_4_4 = p[44];
			const double b_5_0 = p[45];
			const double b_5_1 = p[46];
			const double b_5_2 = p[47];
			const double b_5_3 = p[48];
			const double b_5_4 = p[49];
			const double bs_0 = p[50];
			const double bs_1 = p[51];
			const double bs_2 = p[52];
			const double bs_3 = p[53];
			const double bs_4 = p[54];

			// Common sub-expressions
			const double x0 = s*s;
			const double x1 = s*s*s;
			const double x2 = s*s*s*s;
			const double x3 = a_2_0 + a_2_1*s + a_2_2*x0 + a_2_3*x1 + a_2_4*x2;
			const double x4 = b_2_0 + b_2_1*s + b_2_2*x0 + b_2_3*x1 + b_2_4*x2;
			const double x5 = b_3_0 + b_3_1*s + b_3_2*x0 + b_3_3*x1 + b_3_4*x2;
			const double x6 = x*y;
			const double x7 = x*x;
			const double x8 = a_3_0 + a_3_1*s + a_3_2*x0 + a_3_3*x1 + a_3_4*x2;
			const double x9 = (1.0/2.0)*x8;
			const double x10 = x*x*x;
			const double x11 = a_4_0 + a_4_1*s + a_4_2*x0 + a_4_3*x1 + a_4_4*x2;
			const double x12 = (1.0/6.0)*x11;
			const double x13 = x*x*x*x;
			const double x14 = a_5_0 + a_5_1*s + a_5_2*x0 + a_5_3*x1 + a_5_4*x2;
			const double x15 = (1.0/24.0)*x14;
			const double x16 = y*y;
			const double x17 = y*y*y;
			const double x18 = b_4_0 + b_4_1*s + b_4_2*x0 + b_4_3*x1 + b_4_4*x2;
			const double x19 = (1.0/6.0)*x18;
			const double x20 = x*x16;
			const double x21 = (1.0/2.0)*x11;
			const double x22 = b_5_0 + b_5_1*s + b_5_2*x0 + b_5_3*x1 + b_5_4*x2;
			const double x23 = (1.0/6.0)*x22;
			const double x24 = x*x17;
			const double x25 = (1.0/2.0)*x18;
			const double x26 = x7*y;
			const double x27 = x10*y;
			const double x28 = (1.0/4.0)*x16*x7;
			const double x29 = (1.0/2.0)*x5;
			const double x30 = (1.0/6.0)*x14;

			// Reduced expressions
			*Bx_out = a_1_0 + a_1_1*s + a_1_2*x0 + a_1_3*x1 + a_1_4*x2 + x*x3 + x10*x12 + x13*x15 - x14*x28 + x15*y*y*y*y - x16*x9 - x17*x19 - x20*x21 - x23*x24 + x23*x27 + x25*x26 + x4*y + x5*x6 + x7*x9;
			*By_out = b_1_0 + b_1_1*s + b_1_2*x0 + b_1_3*x1 + b_1_4*x2 + x*x4 + x10*x19 + x12*x17 + (1.0/24.0)*x13*x22 - x16*x29 - x20*x25 - x21*x26 - x22*x28 + x24*x30 - x27*x30 + x29*x7 - x3*y - x6*x8;
			*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2;
		}
		return;

	case 6:
		for (size_t ii = 0; ii < n; ++ii, ++x_array, ++y_array, ++s_array, ++Bx_out, ++By_out, ++Bs_out) {
			const double x = *x_array;
			const double y = *y_array;
			const double s = *s_array;

			// Parameter List
			const double *p = params_flat + ii * 65;
			const double a_1_0 = p[0];
			const double a_1_1 = p[1];
			const double a_1_2 = p[2];
			const double a_1_3 = p[3];
			const double a_1_4 = p[4];
			const double a_2_0 = p[5];
			const double a_2_1 = p[6];
			const double a_2_2 = p[7];
			const double a_2_3 = p[8];
			const double a_2_4 = p[9];
			const double a_3_0 = p[10];
			const double a_3_1 = p[11];
			const double a_3_2 = p[12];
			const double a_3_3 = p[13];
			const double a_3_4 = p[14];
			const double a_4_0 = p[15];
			const double a_4_1 = p[16];
			const double a_4_2 = p[17];
			const double a_4_3 = p[18];
			const double a_4_4 = p[19];
			const double a_5_0 = p[20];
			const double a_5_1 = p[21];
			const double a_5_2 = p[22];
			const double a_5_3 = p[23];
			const double a_5_4 = p[24];
			const double a_6_0 = p[25];
			const double a_6_1 = p[26];
			const double a_6_2 = p[27];
			const double a_6_3 = p[28];
			const double a_6_4 = p[29];
			const double b_1_0 = p[30];
			const double b_1_1 = p[31];
			const double b_1_2 = p[32];
			const double b_1_3 = p[33];
			const double b_1_4 = p[34];
			const double b_2_0 = p[35];
			const double b_2_1 = p[36];
			const double b_2_2 = p[37];
			const double b_2_3 = p[38];
			const double b_2_4 = p[39];
			const double b_3_0 = p[40];
			const double b_3_1 = p[41];
			const double b_3_2 = p[42];
			const double b_3_3 = p[43];
			const double b_3_4 = p[44];
			const double b_4_0 = p[45];
			const double b_4_1 = p[46];
			const double b_4_2 = p[47];
			const double b_4_3 = p[48];
			const double b_4_4 = p[49];
			const double b_5_0 = p[50];
			const double b_5_1 = p[51];
			const double b_5_2 = p[52];
			const double b_5_3 = p[53];
			const double b_5_4 = p[54];
			const double b_6_0 = p[55];
			const double b_6_1 = p[56];
			const double b_6_2 = p[57];
			const double b_6_3 = p[58];
			const double b_6_4 = p[59];
			const double bs_0 = p[60];
			const double bs_1 = p[61];
			const double bs_2 = p[62];
			const double bs_3 = p[63];
			const double bs_4 = p[64];

			// Common sub-expressions
			const double x0 = s*s;
			const double x1 = s*s*s;
			const double x2 = s*s*s*s;
			const double x3 = a_2_0 + a_2_1*s + a_2_2*x0 + a_2_3*x1 + a_2_4*x2;
			const double x4 = b_2_0 + b_2_1*s + b_2_2*x0 + b_2_3*x1 + b_2_4*x2;
			const double x5 = b_3_0 + b_3_1*s + b_3_2*x0 + b_3_3*x1 + b_3_4*x2;
			const double x6 = x*y;
			const double x7 = x*x;
			const double x8 = a_3_0 + a_3_1*s + a_3_2*x0 + a_3_3*x1 + a_3_4*x2;
			const double x9 = (1.0/2.0)*x8;
			const double x10 = x*x*x;
			const double x11 = a_4_0 + a_4_1*s + a_4_2*x0 + a_4_3*x1 + a_4_4*x2;
			const double x12 = (1.0/6.0)*x11;
			const double x13 = x*x*x*x;
			const double x14 = a_5_0 + a_5_1*s + a_5_2*x0 + a_5_3*x1 + a_5_4*x2;
			const double x15 = (1.0/24.0)*x14;
			const double x16 = a_6_0 + a_6_1*s + a_6_2*x0 + a_6_3*x1 + a_6_4*x2;
			const double x17 = (1.0/120.0)*x*x*x*x*x;
			const double x18 = y*y;
			const double x19 = y*y*y;
			const double x20 = b_4_0 + b_4_1*s + b_4_2*x0 + b_4_3*x1 + b_4_4*x2;
			const double x21 = (1.0/6.0)*x20;
			const double x22 = y*y*y*y;
			const double x23 = b_6_0 + b_6_1*s + b_6_2*x0 + b_6_3*x1 + b_6_4*x2;
			const double x24 = x*x18;
			const double x25 = (1.0/2.0)*x11;
			const double x26 = b_5_0 + b_5_1*s + b_5_2*x0 + b_5_3*x1 + b_5_4*x2;
			const double x27 = (1.0/6.0)*x26;
			const double x28 = x*x19;
			const double x29 = (1.0/24.0)*x16;
			const double x30 = x*x22;
			const double x31 = (1.0/2.0)*x20;
			const double x32 = x7*y;
			const double x33 = x10*y;
			const double x34 = x13*y;
			const double x35 = (1.0/24.0)*x23;
			const double x36 = (1.0/4.0)*x18*x7;
			const double x37 = (1.0/12.0)*x23;
			const double x38 = x19*x7;
			const double x39 = x10*x18;
			const double x40 = (1.0/12.0)*x16;
			const double x41 = (1.0/2.0)*x5;
			const double x42 = (1.0/24.0)*x26;
			const double x43 = (1.0/6.0)*x14;

			// Reduced expressions
			*Bx_out = a_1_0 + a_1_1*s + a_1_2*x0 + a_1_3*x1 + a_1_4*x2 + x*x3 + x10*x12 + x13*x15 - x14*x36 + x15*x22 + x16*x17 - x18*x9 - x19*x21 + (1.0/120.0)*x23*y*y*y*y*y - x24*x25 - x27*x28 + x27*x33 + x29*x30 + x31*x32 + x34*x35 - x37*x38 - x39*x40 + x4*y + x5*x6 + x7*x9;
			*By_out = b_1_0 + b_1_1*s + b_1_2*x0 + b_1_3*x1 + b_1_4*x2 + x*x4 + x10*x21 + x12*x19 + x13*x42 + x17*x23 - x18*x41 + x22*x42 - x24*x31 - x25*x32 - x26*x36 + x28*x43 - x29*x34 - x3*y + x30*x35 - x33*x43 - x37*x39 + x38*x40 + x41*x7 - x6*x8;
			*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2;
		}
		return;

	case 7:
		for (size_t ii = 0; ii < n; ++ii, ++x_array, ++y_array, ++s_array, ++Bx_out, ++By_out, ++Bs_out) {
			const double x = *x_array;
			const double y = *y_array;
			const double s = *s_array;

			// Parameter List
			const double *p = params_flat + ii * 75;
			const double a_1_0 = p[0];
			const double a_1_1 = p[1];
			const double a_1_2 = p[2];
			const double a_1_3 = p[3];
			const double a_1_4 = p[4];
			const double a_2_0 = p[5];
			const double a_2_1 = p[6];
			const double a_2_2 = p[7];
			const double a_2_3 = p[8];
			const double a_2_4 = p[9];
			const double a_3_0 = p[10];
			const double a_3_1 = p[11];
			const double a_3_2 = p[12];
			const double a_3_3 = p[13];
			const double a_3_4 = p[14];
			const double a_4_0 = p[15];
			const double a_4_1 = p[16];
			const double a_4_2 = p[17];
			const double a_4_3 = p[18];
			const double a_4_4 = p[19];
			const double a_5_0 = p[20];
			const double a_5_1 = p[21];
			const double a_5_2 = p[22];
			const double a_5_3 = p[23];
			const double a_5_4 = p[24];
			const double a_6_0 = p[25];
			const double a_6_1 = p[26];
			const double a_6_2 = p[27];
			const double a_6_3 = p[28];
			const double a_6_4 = p[29];
			const double a_7_0 = p[30];
			const double a_7_1 = p[31];
			const double a_7_2 = p[32];
			const double a_7_3 = p[33];
			const double a_7_4 = p[34];
			const double b_1_0 = p[35];
			const double b_1_1 = p[36];
			const double b_1_2 = p[37];
			const double b_1_3 = p[38];
			const double b_1_4 = p[39];
			const double b_2_0 = p[40];
			const double b_2_1 = p[41];
			const double b_2_2 = p[42];
			const double b_2_3 = p[43];
			const double b_2_4 = p[44];
			const double b_3_0 = p[45];
			const double b_3_1 = p[46];
			const double b_3_2 = p[47];
			const double b_3_3 = p[48];
			const double b_3_4 = p[49];
			const double b_4_0 = p[50];
			const double b_4_1 = p[51];
			const double b_4_2 = p[52];
			const double b_4_3 = p[53];
			const double b_4_4 = p[54];
			const double b_5_0 = p[55];
			const double b_5_1 = p[56];
			const double b_5_2 = p[57];
			const double b_5_3 = p[58];
			const double b_5_4 = p[59];
			const double b_6_0 = p[60];
			const double b_6_1 = p[61];
			const double b_6_2 = p[62];
			const double b_6_3 = p[63];
			const double b_6_4 = p[64];
			const double b_7_0 = p[65];
			const double b_7_1 = p[66];
			const double b_7_2 = p[67];
			const double b_7_3 = p[68];
			const double b_7_4 = p[69];
			const double bs_0 = p[70];
			const double bs_1 = p[71];
			const double bs_2 = p[72];
			const double bs_3 = p[73];
			const double bs_4 = p[74];

			// Common sub-expressions
			const double x0 = s*s;
			const double x1 = s*s*s;
			const double x2 = s*s*s*s;
			const double x3 = a_2_0 + a_2_1*s + a_2_2*x0 + a_2_3*x1 + a_2_4*x2;
			const double x4 = b_2_0 + b_2_1*s + b_2_2*x0 + b_2_3*x1 + b_2_4*x2;
			const double x5 = b_3_0 + b_3_1*s + b_3_2*x0 + b_3_3*x1 + b_3_4*x2;
			const double x6 = x*y;
			const double x7 = x*x;
			const double x8 = a_3_0 + a_3_1*s + a_3_2*x0 + a_3_3*x1 + a_3_4*x2;
			const double x9 = (1.0/2.0)*x8;
			const double x10 = x*x*x;
			const double x11 = a_4_0 + a_4_1*s + a_4_2*x0 + a_4_3*x1 + a_4_4*x2;
			const double x12 = (1.0/6.0)*x11;
			const double x13 = x*x*x*x;
			const double x14 = a_5_0 + a_5_1*s + a_5_2*x0 + a_5_3*x1 + a_5_4*x2;
			const double x15 = (1.0/24.0)*x14;
			const double x16 = x*x*x*x*x;
			const double x17 = a_6_0 + a_6_1*s + a_6_2*x0 + a_6_3*x1 + a_6_4*x2;
			const double x18 = (1.0/120.0)*x17;
			const double x19 = x*x*x*x*x*x;
			const double x20 = a_7_0 + a_7_1*s + a_7_2*x0 + a_7_3*x1 + a_7_4*x2;
			const double x21 = (1.0/720.0)*x20;
			const double x22 = y*y;
			const double x23 = y*y*y;
			const double x24 = b_4_0 + b_4_1*s + b_4_2*x0 + b_4_3*x1 + b_4_4*x2;
			const double x25 = (1.0/6.0)*x24;
			const double x26 = y*y*y*y;
			const double x27 = y*y*y*y*y;
			const double x28 = b_6_0 + b_6_1*s + b_6_2*x0 + b_6_3*x1 + b_6_4*x2;
			const double x29 = (1.0/120.0)*x28;
			const double x30 = x*x22;
			const double x31 = (1.0/2.0)*x11;
			const double x32 = b_5_0 + b_5_1*s + b_5_2*x0 + b_5_3*x1 + b_5_4*x2;
			const double x33 = (1.0/6.0)*x32;
			const double x34 = x*x23;
			const double x35 = (1.0/24.0)*x17;
			const double x36 = x*x26;
			const double x37 = b_7_0 + b_7_1*s + b_7_2*x0 + b_7_3*x1 + b_7_4*x2;
			const double x38 = (1.0/120.0)*x37;
			const double x39 = x*x27;
			const double x40 = (1.0/2.0)*x24;
			const double x41 = x7*y;
			const double x42 = x10*y;
			const double x43 = x13*y;
			const double x44 = (1.0/24.0)*x28;
			const double x45 = x16*y;
			const double x46 = (1.0/4.0)*x22*x7;
			const double x47 = (1.0/12.0)*x28;
			const double x48 = x23*x7;
			const double x49 = (1.0/48.0)*x20;
			const double x50 = x26*x7;
			const double x51 = x10*x22;
			const double x52 = (1.0/12.0)*x17;
			const double x53 = (1.0/36.0)*x10*x23;
			const double x54 = x13*x22;
			const double x55 = (1.0/2.0)*x5;
			const double x56 = (1.0/24.0)*x32;
			const double x57 = (1.0/6.0)*x14;
			const double x58 = (1.0/120.0)*x20;
			const double x59 = (1.0/48.0)*x37;

			// Reduced expressions
			*Bx_out = a_1_0 + a_1_1*s + a_1_2*x0 + a_1_3*x1 + a_1_4*x2 + x*x3 + x10*x12 + x13*x15 - x14*x46 + x15*x26 + x16*x18 + x19*x21 - x21*y*y*y*y*y*y - x22*x9 - x23*x25 + x27*x29 - x30*x31 - x33*x34 + x33*x42 + x35*x36 - x37*x53 + x38*x39 + x38*x45 + x4*y + x40*x41 + x43*x44 - x47*x48 + x49*x50 - x49*x54 + x5*x6 - x51*x52 + x7*x9;
			*By_out = b_1_0 + b_1_1*s + b_1_2*x0 + b_1_3*x1 + b_1_4*x2 + x*x4 + x10*x25 + x12*x23 + x13*x56 + x16*x29 - x18*x27 + (1.0/720.0)*x19*x37 + x20*x53 - x22*x55 + x26*x56 - x3*y - x30*x40 - x31*x41 - x32*x46 + x34*x57 - x35*x43 + x36*x44 - x39*x58 - x42*x57 - x45*x58 - x47*x51 + x48*x52 + x50*x59 - x54*x59 + x55*x7 - x6*x8;
			*Bs_out = bs_0 + bs_1*s + bs_2*x0 + bs_3*x1 + bs_4*x2;
		}
		return;

	default:
		printf("Error: Unsupported multipole order %d\n", multipole_order);
		printf("Supported orders are 0 to 7\n");
		printf("Setting field values to zero.\n");
		for (size_t ii = 0; ii < n; ++ii, ++Bx_out, ++By_out, ++Bs_out) {
			// Reduced expressions
			*Bx_out = 0;
			*By_out = 0;
			*Bs_out = 0;
		}
		return;
	}
}

#endif // XSUITE_BPMETH_B_FIELD_EVAL_H
