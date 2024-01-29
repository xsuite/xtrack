// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H
#define XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H

// model = 0: adaptive
// model = 1: full (for backward compatibility)
// model = 2: bend-kick-bend
// model = 3: rot-kick-rot
// model = 4: expanded

/*gpufun*/
void track_multipolar_kick_bend(
    LocalParticle* part, int64_t order, double inv_factorial_order,
    /*gpuglmem*/ const double* knl,
    /*gpuglmem*/ const double* ksl,
    double kick_weight, double k0, double k1, double h, double length){

    double const k1l = k1 * length * kick_weight;
    double const k0l = k0 * length * kick_weight;

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
    dpx += -k0l * h * x;

    // k1h correction can be computed from this term in the hamiltonian
    // H = 1/3 hk1 x^3 - 1/2 hk1 xy^2
    // (see MAD 8 physics manual, eq. 5.15, and apply Hamilton's eq. dp/ds = -dH/dx)

    dpx += h * k1l * (-x * x + 0.5 * y * y);
    dpy += h * k1l * x * y;
    LocalParticle_add_to_px(part, dpx);
    LocalParticle_add_to_py(part, dpy);

    multipolar_kick(part, order, inv_factorial_order, knl, ksl, kick_weight);
}


/*gpufun*/
void CombinedFunctionMagnet_track_local_particle(
        CombinedFunctionMagnetData el,
        LocalParticle* part0
) {
    double length = CombinedFunctionMagnetData_get_length(el);

    #ifdef XSUITE_BACKTRACK
        length = -length;
    #endif

    const double k0 = CombinedFunctionMagnetData_get_k0(el);
    const double k1 = CombinedFunctionMagnetData_get_k1(el);
    const double h = CombinedFunctionMagnetData_get_h(el);

    int64_t num_multipole_kicks = CombinedFunctionMagnetData_get_num_multipole_kicks(el);
    const int64_t order = CombinedFunctionMagnetData_get_order(el);
    const double inv_factorial_order = CombinedFunctionMagnetData_get_inv_factorial_order(el);

    const int64_t model = CombinedFunctionMagnetData_get_model(el);

    /*gpuglmem*/ const double *knl = CombinedFunctionMagnetData_getp1_knl(el, 0);
    /*gpuglmem*/ const double *ksl = CombinedFunctionMagnetData_getp1_ksl(el, 0);

    const double slice_length = length / (num_multipole_kicks + 1);
    const double kick_weight = 1. / num_multipole_kicks;


    if (model==0 || model==1 || model==2 || model==3){

            int64_t num_slices;
            if (num_multipole_kicks < 8) {
                num_slices = 1;
            }
            else{
                num_slices = num_multipole_kicks / 7 + 1;
            }

            const double slice_length = length / (num_slices);
            const double kick_weight = 1. / num_slices;
            const double d_yoshida[] =
                         {0x1.91abc4988937bp-2, 0x1.052468fb75c74p-1,
                         -0x1.e25bd194051b9p-2, 0x1.199cec1241558p-4 };
                        //  {1/8.0, 1/8.0, 1/8.0, 1/8.0}; // Uniform, for debugging
            const double k_yoshida[] =
                         {0x1.91abc4988937bp-1, 0x1.e2743579895b4p-3,
                         -0x1.2d7c6f7933b93p+0, 0x1.50b00cfb7be3ep+0 };
                        //  {1/7.0, 1/7.0, 1/7.0, 1/7.0}; // Uniform, for debugging

            double k0_kick, k0_drift;
            if (model ==0 || model==1){
                if (h / 6.28 * slice_length > 2e-2){
                    // Slice is long w.r.t. bending radius
                    //(more than 2 % of bending circumference)
                    k0_kick = 0;
                    k0_drift = k0;
                }
                else{
                    // Slice is short w.r.t. bending radius
                    k0_kick = k0;
                    k0_drift = 0;
                }
            }
            else if (model==2){
                // Force bend-kick-bend
                k0_kick = 0;
                k0_drift = k0;
            }
            else if (model==3){
                // Force drift-kick-drift
                k0_kick = k0;
                k0_drift = 0;
            }

            for (int ii = 0; ii < num_slices; ii++) {
                //start_per_particle_block (part0->part)
                    track_thick_bend(part, slice_length * d_yoshida[0], k0_drift, h);
                    track_multipolar_kick_bend(
                        part, order, inv_factorial_order, knl, ksl,
                        kick_weight * k_yoshida[0], k0_kick, k1, h, length);
                    track_thick_bend(part, slice_length * d_yoshida[1], k0_drift, h);
                    track_multipolar_kick_bend(
                        part, order, inv_factorial_order, knl, ksl,
                        kick_weight * k_yoshida[1], k0_kick, k1, h, length);
                    track_thick_bend(part, slice_length * d_yoshida[2], k0_drift, h);
                    track_multipolar_kick_bend(
                        part, order, inv_factorial_order, knl, ksl,
                        kick_weight * k_yoshida[2], k0_kick, k1, h, length);
                    track_thick_bend(part, slice_length * d_yoshida[3], k0_drift, h);
                    track_multipolar_kick_bend(
                        part, order, inv_factorial_order, knl, ksl,
                        kick_weight * k_yoshida[3], k0_kick, k1, h, length);
                    track_thick_bend(part, slice_length * d_yoshida[3], k0_drift, h);
                    track_multipolar_kick_bend(
                        part, order, inv_factorial_order, knl, ksl,
                        kick_weight * k_yoshida[2], k0_kick, k1, h, length);
                    track_thick_bend(part, slice_length * d_yoshida[2], k0_drift, h);
                    track_multipolar_kick_bend(
                        part, order, inv_factorial_order, knl, ksl,
                        kick_weight * k_yoshida[1], k0_kick, k1, h, length);
                    track_thick_bend(part, slice_length * d_yoshida[1], k0_drift, h);
                    track_multipolar_kick_bend(
                        part, order, inv_factorial_order, knl, ksl,
                        kick_weight * k_yoshida[0], k0_kick, k1, h, length);
                    track_thick_bend(part, slice_length * d_yoshida[0], k0_drift, h);
                //end_per_particle_block
            }

    }
    if (model==4){
        //start_per_particle_block (part0->part)
            track_thick_cfd(part, slice_length, k0, k1, h);

            for (int ii = 0; ii < num_multipole_kicks; ii++) {
                multipolar_kick(part, order, inv_factorial_order, knl, ksl, kick_weight);
                track_thick_cfd(part, slice_length, k0, k1, h);
            }
        //end_per_particle_block
    }

}

#endif // XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H