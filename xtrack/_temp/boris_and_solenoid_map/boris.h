#ifndef BORIS_H
#define BORIS_H

void boris_step(int64_t N_sub_steps, double Dtt,
		double* B_field, double* B_skew,
		double* xn1, double* yn1,  double* zn1,
		double* vxn1, double* vyn1, double* vzn1,
		double* Ex_n, double* Ey_n, 
		double* Bx_n_custom, double* By_n_custom, double* Bz_n_custom,
		int64_t custom_B,
		int64_t N_mp, int64_t N_multipoles,
		double charge, double* mass_array)
{
	int64_t p, isub, order;
	double Ex_np, Ey_np;
	double Bx_n, By_n, Bz_n;
	double rexy, imxy, rexy_0;
	double tBx, tBy, tBz, tBsq;
	double sBx, sBy, sBz;
	double vx_prime, vy_prime, vz_prime;
	double vx_min, vy_min, vz_min;
	double vx_plus, vy_plus, vz_plus;
	double vxn1p, vyn1p, vzn1p;
	double xn1p, yn1p, zn1p;


	for(p=0; p<N_mp; p++)
	{
        double qm = charge/mass_array[p];
		Ex_np = Ex_n[p];
		Ey_np = Ey_n[p];

		vxn1p = vxn1[p];
		vyn1p = vyn1[p];
		vzn1p = vzn1[p];

		xn1p = xn1[p];
		yn1p = yn1[p];
		zn1p = zn1[p];

		for (isub=0; isub<N_sub_steps; isub++)
		{
			rexy = 1.;
			imxy = 0.;
			By_n = B_field[0];
			Bx_n = B_skew[0];
			Bz_n = 0.;

			for(order = 1; order < N_multipoles; order++)
			{
				/* rexy, imxy correspond to real, imaginary part of (x+iy)^(n-1) */
				rexy_0 = rexy;
				rexy = rexy_0*xn1p - imxy*yn1p;
				imxy = imxy*xn1p + rexy_0*yn1p;

				/*
				* Bx +iBy = sum[ (k + ik')(x + iy)^(n-1) ]
				* where k, k' are the strengths and skew strengths of the magnet
				*/
				By_n += (B_field[order]*rexy - B_skew[order]*imxy);
				Bx_n += (B_field[order]*imxy + B_skew[order]*rexy);
			}
			
			if (custom_B){
				Bx_n += Bx_n_custom[p];
				By_n += By_n_custom[p]; 
				Bz_n += Bz_n_custom[p]; 
			}

			tBx = 0.5*qm*Dtt*Bx_n;
			tBy = 0.5*qm*Dtt*By_n;
			tBz = 0.5*qm*Dtt*Bz_n;
			tBsq = tBx*tBx + tBy*tBy + tBz*tBz;

			sBx = 2.*tBx/(1.+tBsq);
			sBy = 2.*tBy/(1.+tBsq);
			sBz = 2.*tBz/(1.+tBsq);

			vx_min = vxn1p + 0.5*qm*Ex_np*Dtt;
			vy_min = vyn1p + 0.5*qm*Ey_np*Dtt;
			vz_min = vzn1p;

			//v_prime = v_min + cross(v_min, tB)
			vx_prime = vy_min*tBz - vz_min*tBy + vx_min;
			vy_prime = vz_min*tBx - vx_min*tBz + vy_min;
			vz_prime = vx_min*tBy - vy_min*tBx + vz_min;

			//v_plus = v_min + cross(v_prime, sB)
			vx_plus = vy_prime*sBz - vz_prime*sBy + vx_min;
			vy_plus = vz_prime*sBx - vx_prime*sBz+ vy_min;
			vz_plus = vx_prime*sBy - vy_prime*sBx + vz_min;

			vxn1p = vx_plus + 0.5*qm*Ex_np*Dtt;
			vyn1p = vy_plus + 0.5*qm*Ey_np*Dtt;
			vzn1p = vz_plus;

			xn1p = xn1p + vxn1p * Dtt;
			yn1p = yn1p + vyn1p * Dtt;
			zn1p = zn1p + vzn1p * Dtt;
		}

		xn1[p] = xn1p;
		yn1[p] = yn1p;
		zn1[p] = zn1p;

		vxn1[p] = vxn1p;
		vyn1[p] = vyn1p;
		vzn1[p] = vzn1p;
	}
}

#endif