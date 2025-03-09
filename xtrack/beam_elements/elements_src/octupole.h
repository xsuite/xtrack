// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_OCTUPOLE_H
#define XTRACK_OCTUPOLE_H

/*gpufun*/
void Octupole_track_local_particle(
        OctupoleData el,
        LocalParticle* part0
) {
    double length = OctupoleData_get_length(el);
    double backtrack_sign = 1;

    #ifdef XSUITE_BACKTRACK
        length = -length;
        backtrack_sign = -1;
    #endif

    double const k3 = OctupoleData_get_k3(el);
    double const k3s = OctupoleData_get_k3s(el);

    double const knl_oct[4] = {0., 0., 0., backtrack_sign * k3 * length};
    double const ksl_oct[4] = {0., 0., 0., backtrack_sign * k3s * length};

    const int64_t order = OctupoleData_get_order(el);
    const double inv_factorial_order = OctupoleData_get_inv_factorial_order(el);
    /*gpuglmem*/ const double *knl = OctupoleData_getp1_knl(el, 0);
    /*gpuglmem*/ const double *ksl = OctupoleData_getp1_ksl(el, 0);

    const uint8_t edge_entry_active = OctupoleData_get_edge_entry_active(el);
    const uint8_t edge_exit_active = OctupoleData_get_edge_exit_active(el);
    const double combined_kn[4] = {0, 0, 0, k3 / 6};
    const double combined_ks[4] = {0, 0, 0, k3s / 6};

    #define OCT_KICK(part, weight) \
        Multipole_track_single_particle(part, \
            0., length * (weight), (weight), \
            knl, ksl, order, inv_factorial_order, \
            knl_oct, ksl_oct, 3, 1./6., \
            backtrack_sign, \
            0, 0, \
            NULL, NULL, NULL, \
            NULL, NULL, NULL, \
            NULL, NULL)

    #define OCT_DRIFT(pp, ww) \
        Drift_single_particle(pp, ((length) * (ww)))

    #define OCT_FRINGE(part, side)\
        MultFringe_track_single_particle(\
            combined_kn,\
            combined_ks,\
            side,\
            4,\
            part\
        )

    // TEAPOT weights
    int64_t num_multipole_kicks = OctupoleData_get_num_multipole_kicks(el);
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
            OCT_FRINGE(part, 0);
        //end_per_particle_block
    }

    // TEAPOT body
    //start_per_particle_block (part0->part)
        OCT_DRIFT(part, edge_drift_weight);
        for (int i_kick=0; i_kick<num_multipole_kicks - 1; i_kick++) {
            OCT_KICK(part, kick_weight);
            OCT_DRIFT(part, inside_drift_weight);
        }
        OCT_KICK(part, kick_weight);
        OCT_DRIFT(part, edge_drift_weight);
    //end_per_particle_block

    // Entry fringe
    if (edge_exit_active) {
        //start_per_particle_block (part0->part)
            OCT_FRINGE(part, 1);
        //end_per_particle_block
    }

    #undef OCT_KICK
    #undef OCT_DRIFT
    #undef OCT_FRINGE
}

#endif // XTRACK_OCTUPOLE_H