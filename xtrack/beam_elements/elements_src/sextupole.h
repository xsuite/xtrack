// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_SEXTUPOLE_H
#define XTRACK_SEXTUPOLE_H

/*gpufun*/
void Sextupole_track_local_particle(
        SextupoleData el,
        LocalParticle* part0
) {
    double length = SextupoleData_get_length(el);
    double backtrack_sign = 1;

    #ifdef XSUITE_BACKTRACK
        length = -length;
        backtrack_sign = -1;
    #endif

    double const k2 = SextupoleData_get_k2(el);
    double const k2s = SextupoleData_get_k2s(el);

    double const knl_sext[3] = {0., 0., backtrack_sign * k2 * length};
    double const ksl_sext[3] = {0., 0., backtrack_sign * k2s * length};

    const int64_t order = SextupoleData_get_order(el);
    const double inv_factorial_order = SextupoleData_get_inv_factorial_order(el);
    /*gpuglmem*/ const double *knl = SextupoleData_getp1_knl(el, 0);
    /*gpuglmem*/ const double *ksl = SextupoleData_getp1_ksl(el, 0);

    //start_per_particle_block (part0->part)

        // Drift
        Drift_single_particle(part, length / 2.);

        Multipole_track_single_particle(part,
            0., length, 1, // weight 1
            knl, ksl, order, inv_factorial_order,
            knl_sext, ksl_sext, 2, 0.5,
            backtrack_sign,
            0, 0,
            NULL, NULL, NULL,
            NULL, NULL, NULL,
            NULL, NULL);

        // Drift
        Drift_single_particle(part, length / 2.);


    //end_per_particle_block


}

#endif // XTRACK_SEXTUPOLE_H