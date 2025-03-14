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

    const uint8_t edge_entry_active = SextupoleData_get_edge_entry_active(el);
    const uint8_t edge_exit_active = SextupoleData_get_edge_exit_active(el);
    const double combined_kn[3] = {0, 0, k2};
    const double combined_ks[3] = {0, 0, k2s};

    #define SEXT_KICK(part, weight) \
        Multipole_track_single_particle(part, \
            0., length * (weight), (weight), \
            knl, ksl, order, inv_factorial_order, \
            knl_sext, ksl_sext, 2, 0.5, \
            backtrack_sign, \
            0, 0, \
            NULL, NULL, NULL, \
            NULL, NULL, NULL, \
            NULL, NULL)

    #define SEXT_DRIFT(pp, ww) \
        Drift_single_particle(pp, ((length) * (ww)))

    #define SEXT_FRINGE(part, side) \
        MultFringe_track_single_particle( \
            part, \
            combined_kn, \
            combined_ks, \
            /* k_order */ 2, \
            /* knl */ NULL, \
            /* ksl */ NULL, \
            /* kl_order */ -1, \
            length, \
            side, \
            /* min_order */ 0 \
        )

    // TEAPOT weights
    int64_t num_multipole_kicks = SextupoleData_get_num_multipole_kicks(el);
    if (num_multipole_kicks == 0) { // auto mode
        num_multipole_kicks = 1;
    }
    const double kick_weight = 1. / num_multipole_kicks;
    double edge_drift_weight = 0.5;
    double inside_drift_weight = 0;
    if (num_multipole_kicks > 1) {
        edge_drift_weight = 1. / (2 * (1 + num_multipole_kicks));
        inside_drift_weight = (
            ((float) num_multipole_kicks)
                / ((float)(num_multipole_kicks*num_multipole_kicks) - 1));
    }

    // TRACKING
    // Entry fringe
    if (edge_entry_active) {
        //start_per_particle_block (part0->part)
            SEXT_FRINGE(part, 0);
        //end_per_particle_block
    }

    // TEAPOT body
    //start_per_particle_block (part0->part)
        SEXT_DRIFT(part, edge_drift_weight);
        for (int i_kick=0; i_kick<num_multipole_kicks - 1; i_kick++) {
            SEXT_KICK(part, kick_weight);
            SEXT_DRIFT(part, inside_drift_weight);
        }
        SEXT_KICK(part, kick_weight);
        SEXT_DRIFT(part, edge_drift_weight);
    //end_per_particle_block

    // Entry fringe
    if (edge_exit_active) {
        //start_per_particle_block (part0->part)
            SEXT_FRINGE(part, 1);
        //end_per_particle_block
    }

    #undef SEXT_KICK
    #undef SEXT_DRIFT
    #undef SEXT_FRINGE
}

#endif // XTRACK_SEXTUPOLE_H