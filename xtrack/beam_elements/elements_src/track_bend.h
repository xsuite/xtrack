// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_TRACK_BEND_H
#define XTRACK_TRACK_BEND_H


// model = 0: adaptive
// model = 1: full (for backward compatibility)
// model = 2: bend-kick-bend
// model = 3: rot-kick-rot
// model = 4: expanded

/*gpufun*/
void track_multipolar_kick_bend(
    LocalParticle* part, int64_t order, double inv_factorial_order,
    const double* knl,
    const double* ksl,
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

#define N_KICKS_YOSHIDA 7

/*gpufun*/
void Bend_track_local_particle_from_params(LocalParticle* part0,
                double length, double k0, double k1, double h,
                int64_t num_multipole_kicks, int64_t model,
                double const* knl, double const* ksl,
                int64_t order, double inv_factorial_order,
                double factor_knl_ksl){

    if (num_multipole_kicks == 0) { // num_multipole_kicks needs to be determined

        if (fabs(h) < 1e-8){
            num_multipole_kicks = 1; // straight magnet, one multipole kick in the middle
        }
        else{
            double b_circum = 2 * 3.14159 / fabs(h);
            num_multipole_kicks = fabs(length) / b_circum / 0.5e-3; // 0.5 mrad per kick (on average)
        }
    }

    if (model==0 || model==1 || model==2 || model==3){

            int64_t num_slices;
            if (num_multipole_kicks <= N_KICKS_YOSHIDA) {
                num_slices = 1;
            }
            else{
                num_slices = num_multipole_kicks / N_KICKS_YOSHIDA + 1;
            }

            const double slice_length = length / (num_slices);
            const double kick_weight = 1. / num_slices;
            const double d_yoshida[] =
                         // From MAD-NG
                         {3.922568052387799819591407413100e-01,
                          5.100434119184584780271052295575e-01,
                          -4.710533854097565531482416645304e-01,
                          6.875316825251809316199569366290e-02};
                        //  {0x1.91abc4988937bp-2, 0x1.052468fb75c74p-1, // same in hex
                        //  -0x1.e25bd194051b9p-2, 0x1.199cec1241558p-4 };
                        //  {1/8.0, 1/8.0, 1/8.0, 1/8.0}; // Uniform, for debugging
            const double k_yoshida[] =
                         // From MAD-NG
                         {7.845136104775599639182814826199e-01,
                          2.355732133593569921359289764951e-01,
                          -1.177679984178870098432412305556e+00,
                          1.315186320683906284756403692882e+00};
                        //  {0x1.91abc4988937bp-1, 0x1.e2743579895b4p-3, // same in hex
                        //  -0x1.2d7c6f7933b93p+0, 0x1.50b00cfb7be3ep+0 };
                        //  {1/7.0, 1/7.0, 1/7.0, 1/7.0}; // Uniform, for debugging

            // printf("num_slices = %ld\n", num_slices);
            // printf("slice_length = %e\n", slice_length);
            // printf("check = %d\n", check);

            double k0_kick = 0;
            double k0_drift = 0;
            if (model ==0 || model==1 || model==3){
                // Slice is short w.r.t. bending radius
                k0_kick = k0;
                k0_drift = 0;
            }
            else {
                // method is 2
                // Force bend-kick-bend
                k0_kick = 0;
                k0_drift = k0;
            }

            // printf("k0_kick = %e\n", k0_kick);

            // Check if it can be handled without slicing
            int no_slice_needed = 0;
            if (k0_kick == 0 && k1 == 0){
                int multip_present = 0;
                for (int mm=0; mm<=order; mm++){
                    if (knl[mm] != 0 || ksl[mm] != 0){
                        multip_present = 1;
                        break;
                    }
                }
                if (!multip_present){
                    no_slice_needed = 1;
                }
            }

            if (no_slice_needed){
                // printf("No slicing needed\n");
                //start_per_particle_block (part0->part)
                    track_thick_bend(part, length, k0_drift, h);
                //end_per_particle_block
            }
            else{
                for (int ii = 0; ii < num_slices; ii++) {
                    //start_per_particle_block (part0->part)
                        track_thick_bend(part, slice_length * d_yoshida[0], k0_drift, h);
                        track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl, factor_knl_ksl,
                            kick_weight * k_yoshida[0], k0_kick, k1, h, length);
                        track_thick_bend(part, slice_length * d_yoshida[1], k0_drift, h);
                        track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl, factor_knl_ksl,
                            kick_weight * k_yoshida[1], k0_kick, k1, h, length);
                        track_thick_bend(part, slice_length * d_yoshida[2], k0_drift, h);
                        track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl, factor_knl_ksl,
                            kick_weight * k_yoshida[2], k0_kick, k1, h, length);
                        track_thick_bend(part, slice_length * d_yoshida[3], k0_drift, h);
                        track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl, factor_knl_ksl,
                            kick_weight * k_yoshida[3], k0_kick, k1, h, length);
                        track_thick_bend(part, slice_length * d_yoshida[3], k0_drift, h);
                        track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl, factor_knl_ksl,
                            kick_weight * k_yoshida[2], k0_kick, k1, h, length);
                        track_thick_bend(part, slice_length * d_yoshida[2], k0_drift, h);
                        track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl, factor_knl_ksl,
                            kick_weight * k_yoshida[1], k0_kick, k1, h, length);
                        track_thick_bend(part, slice_length * d_yoshida[1], k0_drift, h);
                        track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl, factor_knl_ksl,
                            kick_weight * k_yoshida[0], k0_kick, k1, h, length);
                        track_thick_bend(part, slice_length * d_yoshida[0], k0_drift, h);
                    //end_per_particle_block
                }
            }

    }
    if (model==4){
        const double slice_length = length / (num_multipole_kicks + 1);
        const double kick_weight = 1. / num_multipole_kicks;
        //start_per_particle_block (part0->part)
            track_thick_cfd(part, slice_length, k0, k1, h);

            for (int ii = 0; ii < num_multipole_kicks; ii++) {
                track_multipolar_kick_bend(
                            part, order, inv_factorial_order, knl, ksl, factor_knl_ksl,
                            kick_weight, 0, 0, 0, 0);
                track_thick_cfd(part, slice_length, k0, k1, h);
            }
        //end_per_particle_block
    }
}

#endif