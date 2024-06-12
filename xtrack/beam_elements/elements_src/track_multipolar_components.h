// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_TRACK_MULTIPOLAR_COMPONENTS_H
#define XTRACK_TRACK_MULTIPOLAR_COMPONENTS_H

/*gpufun*/
void track_multipolar_kick_bend(
    LocalParticle* part, int64_t order, double inv_factorial_order,
    /*gpuglmem*/ const double* knl,
    /*gpuglmem*/ const double* ksl,
    double const factor_knl_ksl,
    double kick_weight, double k0, double k1, double h, double length){

    double const chi = LocalParticle_get_chi(part);

    double const k1l = chi * k1 * length * kick_weight;
    double const k0l = chi * k0 * length * kick_weight;

    // dipole kick
    double dpx = -k0l;
    double dpy = 0;

    // quadrupole kick
    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);
    dpx += -k1l * x;
    dpy +=  k1l * y;

    // k0h correction can be computed from this term in the hamiltonian
    // H = 1/2 h k0 x^2
    // (see MAD 8 physics manual, eq. 5.15, and apply Hamilton's eq. dp/ds = -dH/dx)
    double const k0l_mult = chi * knl[0] * factor_knl_ksl * kick_weight;
    dpx += -(k0l + k0l_mult) * h * x;

    // k1h correction can be computed from this term in the hamiltonian
    // H = 1/3 hk1 x^3 - 1/2 hk1 xy^2
    // (see MAD 8 physics manual, eq. 5.15, and apply Hamilton's eq. dp/ds = -dH/dx)
    double const k1l_mult = chi * knl[1] * factor_knl_ksl * kick_weight;
    dpx += h * (k1l + k1l_mult) * (-x * x + 0.5 * y * y);
    dpy += h * (k1l + k1l_mult) * x * y;

    LocalParticle_add_to_px(part, dpx);
    LocalParticle_add_to_py(part, dpy);


    // multipolar kick
    int64_t index = order;
    double inv_factorial = inv_factorial_order;

    double dpx_mul = chi * knl[index] * factor_knl_ksl * inv_factorial;
    double dpy_mul = chi * ksl[index] * factor_knl_ksl * inv_factorial;

    while( index > 0 )
    {
        double const zre = dpx_mul * x - dpy_mul * y;
        double const zim = dpx_mul * y + dpy_mul * x;

        inv_factorial *= index;
        index -= 1;

        double this_knl = chi * knl[index] * factor_knl_ksl;
        double this_ksl = chi * ksl[index] * factor_knl_ksl;

        dpx_mul = this_knl * inv_factorial + zre;
        dpy_mul = this_ksl * inv_factorial + zim;
    }

    dpx_mul = -dpx_mul; // rad

    LocalParticle_add_to_px(part, kick_weight * dpx_mul);
    LocalParticle_add_to_py(part, kick_weight * dpy_mul);

}

#endif