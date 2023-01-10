// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_DRIFT_H
#define XTRACK_DRIFT_H

/*gpufun*/
void Drift_track_local_particle(DriftData el, LocalParticle* part0){

    double const length = DriftData_get_length(el);

    //start_per_particle_block (part0->part)

    #ifndef XTRACK_USE_EXACT_DRIFTS

        double const rpp    = LocalParticle_get_rpp(part);
        double const rv0v    = 1./LocalParticle_get_rvv(part);
        double const xp     = LocalParticle_get_px(part) * rpp;
        double const yp     = LocalParticle_get_py(part) * rpp;
        double const dzeta  = 1 - rv0v * ( 1. + ( xp*xp + yp*yp ) / 2. );

        LocalParticle_add_to_x(part, xp * length );
        LocalParticle_add_to_y(part, yp * length );
        LocalParticle_add_to_s(part, length);
        LocalParticle_add_to_zeta(part, length * dzeta );

    #else

        double const px = LocalParticle_get_px(part);
        double const py = LocalParticle_get_py(part);
        double const rv0v    = 1./LocalParticle_get_rvv(part);
        double const one_plus_delta = 1. + LocalParticle_get_delta(part);

        double const one_over_pz = 1./sqrt(one_plus_delta*one_plus_delta
                                           - px * px - py * py);
        double const dzeta = 1 - rv0v * one_plus_delta * one_over_pz;

        LocalParticle_add_to_x(part, px * one_over_pz * length);
        LocalParticle_add_to_y(part, py * one_over_pz * length);
        LocalParticle_add_to_zeta(part, dzeta * length);
        LocalParticle_add_to_s(part, length);

    #endif
    //end_per_particle_block

}

#endif
