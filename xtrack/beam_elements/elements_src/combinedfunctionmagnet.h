// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H
#define XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H

// model = 0: thick combined function dipole (expanded hamiltonian)
// model = 1: thick combined function dipole (exact bend + quadrupole kicks
//                                            + k1h correction, yoshida integration)
//


/*gpufun*/
void track_multipolar_kick_bend(
    LocalParticle* part, int64_t order, double inv_factorial_order,
    /*gpuglmem*/ const double* knl,
    /*gpuglmem*/ const double* ksl,
    double kick_weight, double k1, double h, double length){

    double const k1l = k1 * length * kick_weight;

    // quadrupole kick
    double const x = LocalParticle_get_x(part);
    double const y = LocalParticle_get_y(part);
    double dpx = -k1l * x;
    double dpy =  k1l * y;

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

    if (model==0){
        //start_per_particle_block (part0->part)
            track_thick_cfd(part, slice_length, k0, k1, h);

            for (int ii = 0; ii < num_multipole_kicks; ii++) {
                multipolar_kick(part, order, inv_factorial_order, knl, ksl, kick_weight);
                track_thick_cfd(part, slice_length, k0, k1, h);
            }
        //end_per_particle_block
    }
    else if (model==10){

        if (fabs(k1) > 0 && num_multipole_kicks == 0) {
            num_multipole_kicks = 5; // default value
        }
        double const k1lslice = k1 * length / num_multipole_kicks;

        //start_per_particle_block (part0->part)
            track_thick_bend(part, slice_length, k0, h);
        //end_per_particle_block

        for (int ii = 0; ii < num_multipole_kicks; ii++) {
            if ((fabs(h) > 0) && (fabs(k1) > 0)) {
                //start_per_particle_block (part0->part)
                    double const x = LocalParticle_get_x(part);
                    double const y = LocalParticle_get_y(part);
                    double dpx = -k1lslice * x;
                    double dpy =  k1lslice * y;
                    if (model == 10){
                        dpx += h * k1lslice * (-x * x + 0.5 * y * y);
                        dpy += h * k1lslice * x * y;
                    }
                    LocalParticle_add_to_px(part, dpx);
                    LocalParticle_add_to_py(part, dpy);
                //end_per_particle_block
            }
            //start_per_particle_block (part0->part)
                multipolar_kick(part, order, inv_factorial_order, knl, ksl, kick_weight);
                track_thick_bend(part, slice_length, k0, h);
            //end_per_particle_block
        }
    }
    else{

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

            for (int ii = 0; ii < num_slices; ii++) {
                //start_per_particle_block (part0->part)
                    track_thick_bend(part, slice_length * d_yoshida[0], k0, h);
                    track_multipolar_kick_bend(
                        part, order, inv_factorial_order, knl, ksl,
                        kick_weight * k_yoshida[0], k1, h, length);
                    track_thick_bend(part, slice_length * d_yoshida[1], k0, h);
                    track_multipolar_kick_bend(
                        part, order, inv_factorial_order, knl, ksl,
                        kick_weight * k_yoshida[1], k1, h, length);
                    track_thick_bend(part, slice_length * d_yoshida[2], k0, h);
                    track_multipolar_kick_bend(
                        part, order, inv_factorial_order, knl, ksl,
                        kick_weight * k_yoshida[2], k1, h, length);
                    track_thick_bend(part, slice_length * d_yoshida[3], k0, h);
                    track_multipolar_kick_bend(
                        part, order, inv_factorial_order, knl, ksl,
                        kick_weight * k_yoshida[3], k1, h, length);
                    track_thick_bend(part, slice_length * d_yoshida[3], k0, h);
                    track_multipolar_kick_bend(
                        part, order, inv_factorial_order, knl, ksl,
                        kick_weight * k_yoshida[2], k1, h, length);
                    track_thick_bend(part, slice_length * d_yoshida[2], k0, h);
                    track_multipolar_kick_bend(
                        part, order, inv_factorial_order, knl, ksl,
                        kick_weight * k_yoshida[1], k1, h, length);
                    track_thick_bend(part, slice_length * d_yoshida[1], k0, h);
                    track_multipolar_kick_bend(
                        part, order, inv_factorial_order, knl, ksl,
                        kick_weight * k_yoshida[0], k1, h, length);
                    track_thick_bend(part, slice_length * d_yoshida[0], k0, h);
                //end_per_particle_block
            }

    }

}

#endif // XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H