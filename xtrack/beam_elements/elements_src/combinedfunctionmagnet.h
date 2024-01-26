// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H
#define XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H

// model = 0: thick combined function dipole (expanded hamiltonian)
// model = 1: thick combined function dipole (exact bend + quadrupole kicks + k1h correction)
// model = 2: thick combined function dipole (exact bend + quadrupole kicks, no k1h correction)

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
    else if (model==1 || model==2){

        #ifdef XSUITE_BACKTRACK
                LocalParticle_kill_particle(part, -30);
        #else

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
                        if (model == 1){
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
        #endif
    }
    else{
        #ifdef XSUITE_BACKTRACK
                LocalParticle_kill_particle(part, -30);
        #else

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
                        if (model == 1){
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

}

#endif // XTRACK_THICKCOMBINEDFUNCTIONDIPOLE_H